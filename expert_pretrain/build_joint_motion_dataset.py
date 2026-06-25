#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from schema import (
    CANONICAL_IMUS,
    CANONICAL_JOINTS,
    DIP_REDUCED_IMUS,
    SMPL_JOINTS,
    XSENS_JOINTS,
    canonical_name,
)


def _rotation_to_9d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 9:
        return x.float()
    if x.shape[-1] == 4:
        # Local import keeps this script usable for metadata checks without importing articulate early.
        import articulate as art
        return art.math.quaternion_to_rotation_matrix(x.float()).view(*x.shape[:-1], 9)
    raise ValueError(f"Unsupported orientation shape: {tuple(x.shape)}")


def _zeroed_imu(acc: torch.Tensor, ori: torch.Tensor, names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    t = acc.shape[0]
    out = torch.zeros(t, len(CANONICAL_IMUS), 12, dtype=torch.float32)
    mask = torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)
    ori9 = _rotation_to_9d(ori)

    index = {name: i for i, name in enumerate(CANONICAL_IMUS)}
    for src_i, raw_name in enumerate(names):
        name = canonical_name(raw_name)
        if name not in index:
            continue
        dst_i = index[name]
        out[:, dst_i, :9] = ori9[:, src_i]
        out[:, dst_i, 9:] = acc[:, src_i].float()
        mask[dst_i] = True
    return out, mask


def _zeroed_joints(position: torch.Tensor, names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    t = position.shape[0]
    out = torch.zeros(t, len(CANONICAL_JOINTS), 3, dtype=torch.float32)
    mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    index = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
    for src_i, raw_name in enumerate(names):
        name = canonical_name(raw_name)
        if name not in index:
            continue
        dst_i = index[name]
        out[:, dst_i] = position[:, src_i].float()
        mask[dst_i] = True
    return out, mask


def _motion_targets(position: torch.Tensor, fps: float) -> Dict[str, torch.Tensor]:
    # Root-relative absolute position is kept only for analysis; the training target is movement.
    root = position[:, :1].clone()
    root_relative = position - root
    delta = torch.zeros_like(root_relative)
    delta[1:] = root_relative[1:] - root_relative[:-1]
    velocity = delta * fps
    displacement = root_relative - root_relative[:1]
    distance = delta.norm(dim=-1, keepdim=True)
    return {
        "joint_position_root_relative": root_relative,
        "joint_delta": delta,
        "joint_velocity": velocity,
        "joint_displacement": displacement,
        "joint_step_distance": distance,
    }


def _windows(t: int, seq_len: int, stride: int) -> Iterable[Tuple[int, int]]:
    if t <= seq_len:
        yield 0, t
        return
    for start in range(0, t - seq_len + 1, stride):
        yield start, start + seq_len
    last = t - seq_len
    if last % stride != 0:
        yield last, t


def _load_extract_file(path: str):
    data = torch.load(path, map_location="cpu")
    acc = data["imu"].get("free acceleration")
    ori = data["imu"].get("calibrated orientation", data["imu"].get("orientation"))
    imu_names = data["imu"].get("name", [])
    pos = data["joint"]["position"]
    joint_names = data["joint"].get("name", XSENS_JOINTS if pos.shape[1] == 23 else SMPL_JOINTS)
    fps = float(data.get("framerate", 60.0))
    return acc, ori, imu_names, pos, joint_names, fps


def _load_work_file(path: str):
    data = torch.load(path, map_location="cpu")
    imu = data["imu"]["imu"].float().view(-1, 6, 12)
    pos = data["joint"]["position"].float()
    names = SMPL_JOINTS if pos.shape[1] == 24 else XSENS_JOINTS
    acc = imu[:, :, 9:]
    ori = imu[:, :, :9]
    return acc, ori, DIP_REDUCED_IMUS, pos, names, 60.0


def _source_files(root: Path, mode: str, datasets: List[str]) -> List[Tuple[str, str, str]]:
    files = []
    if mode == "extract":
        for dataset in datasets:
            for path in sorted(glob.glob(str(root / "datasets" / "extract" / dataset / "*.pt"))):
                files.append((dataset, "all", path))
    elif mode == "work":
        for split in ["train", "test"]:
            for dataset in datasets:
                for path in sorted(glob.glob(str(root / "datasets" / "work" / split / dataset / "*.pt"))):
                    files.append((dataset, split, path))
    else:
        raise ValueError(mode)
    return files


def build(args):
    root = Path(args.root)
    out_root = Path(args.output)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    files = _source_files(root, args.mode, datasets)
    out_root.mkdir(parents=True, exist_ok=True)

    shard = []
    shard_idx = 0
    total = 0

    loader = _load_extract_file if args.mode == "extract" else _load_work_file
    for dataset, split, path in files:
        try:
            acc, ori, imu_names, pos, joint_names, fps = loader(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue

        imu, imu_mask = _zeroed_imu(acc, ori, imu_names)
        joints, joint_mask = _zeroed_joints(pos, joint_names)
        targets = _motion_targets(joints, fps)
        length = imu.shape[0]

        for start, end in _windows(length, args.seq_len, args.stride):
            if end - start < args.min_len:
                continue
            sample = {
                "imu": imu[start:end],
                "imu_mask": imu_mask,
                "joint_mask": joint_mask,
                "fps": fps,
                "source": dataset,
                "split": split,
                "motion": os.path.basename(path),
                "start": start,
                "end": end,
            }
            for key, value in targets.items():
                sample[key] = value[start:end]
            shard.append(sample)
            total += 1

            if len(shard) >= args.shard_size:
                _save_shard(out_root, shard_idx, shard, args)
                shard_idx += 1
                shard = []

    if shard:
        _save_shard(out_root, shard_idx, shard, args)
    _save_meta(out_root, args, total)
    print(f"built {total} windows into {out_root}")


def _save_shard(out_root: Path, shard_idx: int, shard: List[Dict], args):
    path = out_root / f"shard_{shard_idx:05d}.pt"
    torch.save({"samples": shard}, path)
    print(f"[save] {path} ({len(shard)} samples)")


def _save_meta(out_root: Path, args, total: int):
    torch.save({
        "canonical_imus": CANONICAL_IMUS,
        "canonical_joints": CANONICAL_JOINTS,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "samples": total,
        "target": "joint_delta / joint_displacement sequence, root-relative",
    }, out_root / "metadata.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="datasets/expert_pretrain")
    parser.add_argument("--mode", choices=["extract", "work"], default="extract")
    parser.add_argument("--datasets", default="andy,emokine,unipd,virginia,cip,dip")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=256)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
