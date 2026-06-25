#!/usr/bin/env python3
"""Compare all-expert full-pose outputs with MobilePoser-style metrics.

This script is for the root-relative vs self pose-change ablation:

  root-relative expert target: joint_orient_r6d,joint_delta
  self-change expert target:  joint_rot_delta_r6d,joint_delta_local

Both models are evaluated in the all-experts regime: all 28 experts run, their
orientation outputs are converted to a full canonical pose trajectory, and the
result is compared to GT with the MobilePoser-style FK pose/mesh/jitter metrics.
Root-current model outputs are first un-normalized with the pelvis/root IMU
orientation. Self-change model outputs are integrated from the first global GT pose
of each window before metric computation.
"""
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval_wholebody import MobilePoserMetricAccumulator
from masked_dataset import MaskedMotionDataset, apply_visible_imu_sampling, build_fixed_visible_sets, collate
from multiexpert_model import MultiExpert
from rotation_utils import rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys, target_dim, target_slices

PELVIS_IMU_IDX = CANONICAL_IMUS.index("Pelvis")


def matrix_to_r6d(rot: torch.Tensor) -> torch.Tensor:
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def dedupe(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_model(path: str, device):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    keys = split_keys(args.get("target", "joint_orient_r6d,joint_delta"))
    dim = int(args.get("target_dim", target_dim(keys)))
    model = MultiExpert(
        len(CANONICAL_IMUS),
        len(CANONICAL_JOINTS),
        hidden_dim=int(args.get("hidden", 128)),
        nhead=int(args.get("nhead", 4)),
        num_layers=int(args.get("layers", 4)),
        target_dim=dim,
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    if slice_for(keys, "joint_orient_r6d") is not None:
        mode = "root_current"
    elif slice_for(keys, "joint_rot_delta_r6d") is not None:
        mode = "self_delta"
    else:
        raise ValueError(f"{path} has no supported orientation target: {keys}")
    return {"path": path, "ckpt": ckpt, "args": args, "keys": keys, "model": model, "mode": mode}


def integrate_self_rotation(delta6: torch.Tensor, init_orient6: torch.Tensor) -> torch.Tensor:
    """Integrate R[t] = R[t-1] @ dR[t] from the first global GT pose of the window."""
    delta = rotation_6d_to_matrix(delta6.float())
    init = rotation_6d_to_matrix(init_orient6[:, 0].float())
    b, t, j = delta.shape[:3]
    out = torch.empty(b, t, j, 3, 3, dtype=delta.dtype, device=delta.device)
    out[:, 0] = init
    for step in range(1, t):
        out[:, step] = out[:, step - 1] @ delta[:, step]
    return matrix_to_r6d(out).to(delta6.dtype)


def unnormalize_root_current(rel6: torch.Tensor, full_imu: torch.Tensor) -> torch.Tensor:
    """Undo root normalization: R_global_joint = R_root @ R_root_relative_joint."""
    rel = rotation_6d_to_matrix(rel6.float())
    root = full_imu[:, :, PELVIS_IMU_IDX, :9].float().view(full_imu.shape[0], full_imu.shape[1], 3, 3)
    global_rot = root.unsqueeze(2) @ rel
    return matrix_to_r6d(global_rot).to(rel6.dtype)


def geodesic_sums_per_joint(pred6, tgt6, mask, lengths):
    """Per-joint global-pose rotation error sums in degrees."""
    Rp = rotation_6d_to_matrix(pred6.float())
    Rt = rotation_6d_to_matrix(tgt6.float())
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / 3.14159265)
    b, t, j = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t, 1)
    valid = mask.view(b, 1, j) & (steps < lengths.view(b, 1, 1))
    return (ang * valid).sum(dim=(0, 1)), valid.sum(dim=(0, 1))


def xyz_dist_sums_per_joint(pred_xyz, tgt_xyz, mask, lengths):
    """Per-joint movement-vector error sums in centimetres."""
    dist = (pred_xyz.float() - tgt_xyz.float()).norm(dim=-1) * 100.0
    b, t, j = dist.shape
    steps = torch.arange(t, device=dist.device).view(1, t, 1)
    valid = mask.view(b, 1, j) & (steps < lengths.view(b, 1, 1))
    return (dist * valid).sum(dim=(0, 1)), valid.sum(dim=(0, 1))


def init_error_sums(device):
    joints = len(CANONICAL_JOINTS)
    return {
        "angle_num": torch.zeros(joints, device=device),
        "angle_den": torch.zeros(joints, device=device),
        "xyz_num": torch.zeros(joints, device=device),
        "xyz_den": torch.zeros(joints, device=device),
    }


def add_error_sums(total, angle_num, angle_den, xyz_num, xyz_den):
    total["angle_num"] += angle_num
    total["angle_den"] += angle_den
    total["xyz_num"] += xyz_num
    total["xyz_den"] += xyz_den


def summarize_error_sums(name: str, sums):
    angle_den = sums["angle_den"].detach().cpu()
    xyz_den = sums["xyz_den"].detach().cpu()
    angle = sums["angle_num"].detach().cpu() / angle_den.clamp_min(1.0)
    xyz = sums["xyz_num"].detach().cpu() / xyz_den.clamp_min(1.0)
    angle_valid = angle_den > 0
    xyz_valid = xyz_den > 0
    if angle_valid.any():
        angle_macro = float(angle[angle_valid].mean())
        angle_pooled = float(sums["angle_num"].sum() / sums["angle_den"].sum().clamp_min(1.0))
    else:
        angle_macro = float("nan")
        angle_pooled = float("nan")
    if xyz_valid.any():
        xyz_macro = float(xyz[xyz_valid].mean())
        xyz_pooled = float(sums["xyz_num"].sum() / sums["xyz_den"].sum().clamp_min(1.0))
    else:
        xyz_macro = float("nan")
        xyz_pooled = float("nan")
    print(
        f"  [{name}] direct_errors "
        f"pose_angle={angle_macro:.2f}deg pooled={angle_pooled:.2f}deg | "
        f"movement_xyz={xyz_macro:.3f}cm pooled={xyz_pooled:.3f}cm"
    )


def predict_outputs(entry, imu, imu_mask, lengths, gt_global_orient6, full_imu):
    pred = entry["model"](imu, imu_mask, lengths)
    slices = target_slices(entry["keys"])
    if entry["mode"] == "root_current":
        a, b = slices["joint_orient_r6d"]
        orient6 = unnormalize_root_current(pred[..., a:b], full_imu)
        motion_key = "joint_delta"
    else:
        a, b = slices["joint_rot_delta_r6d"]
        orient6 = integrate_self_rotation(pred[..., a:b], gt_global_orient6)
        motion_key = "joint_delta_local"
    ma, mb = slices[motion_key]
    return orient6, pred[..., ma:mb], motion_key


def summarize_mobileposer(name: str, metric: MobilePoserMetricAccumulator):
    summary = metric.summary()
    print(f"\n[{name}] windows={len(metric.rows)} skipped={metric.skipped}")
    if summary is None:
        print("  no windows had the required SMPL-24 supervised joints")
        return
    for metric_name, row in zip(metric.__class__.__module__ and [
        "SIP Error (deg)",
        "Angular Error (deg)",
        "Masked Angular Error (deg)",
        "Positional Error (cm)",
        "Masked Positional Error (cm)",
        "Mesh Error (cm)",
        "Jitter Error (100m/s^3)",
        "Distance Error (cm)",
    ], summary):
        if torch.isnan(row[0]):
            print(f"  {metric_name}: n/a")
        else:
            print(f"  {metric_name}: {float(row[0]):.2f} (+/- {float(row[1]):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pengfei/Downloads/poser_mle_posechange_60hz")
    ap.add_argument("--manifest", default="", help="default: <data>/splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--root-ckpt", default="", help="checkpoint trained with joint_orient_r6d,joint_delta")
    ap.add_argument("--self-ckpt", default="", help="checkpoint trained with joint_rot_delta_r6d,joint_delta_local")
    ap.add_argument("--ks", default="2,3,4,5", help="active IMU counts to evaluate")
    ap.add_argument("--include-kw", default="",
                    help="optional comma-separated motion-name keywords for subset eval, "
                         "e.g. totalcapture, dip, or imuposer")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mobileposer-smpl",
                    default="/home/pengfei/Downloads/mobileposer_official/mobileposer/smpl/basicmodel_m.pkl")
    args = ap.parse_args()

    ckpt_paths = OrderedDict()
    if args.root_ckpt:
        ckpt_paths["root_current"] = args.root_ckpt
    if args.self_ckpt:
        ckpt_paths["self_delta"] = args.self_ckpt
    if not ckpt_paths:
        raise ValueError("Pass at least one of --root-ckpt or --self-ckpt")
    if not Path(args.mobileposer_smpl).exists():
        raise FileNotFoundError(f"MobilePoser SMPL file not found: {args.mobileposer_smpl}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    entries = OrderedDict((name, load_model(path, device)) for name, path in ckpt_paths.items())
    eval_keys = dedupe(["joint_global_orient_r6d"] + [key for e in entries.values() for key in e["keys"]])
    manifest = args.manifest or f"{args.data.rstrip('/')}/splits.json"
    slices = target_slices(eval_keys)
    if "joint_global_orient_r6d" not in slices:
        raise ValueError("evaluation target must include joint_global_orient_r6d")

    include_kw = [k.strip().lower() for k in args.include_kw.split(",") if k.strip()]
    val_ds = MaskedMotionDataset(args.data, augment=False, target_key=",".join(eval_keys),
                                 mask_key="auto", manifest=manifest, split="val", require_min_imus=True)
    eval_ds = MaskedMotionDataset(args.data, augment=False, target_key=",".join(eval_keys),
                                  mask_key="auto", manifest=manifest, split=args.split, require_min_imus=True,
                                  include_only_kw=include_kw or None)
    dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate, pin_memory=(device.type == "cuda"))

    subset = f" include_kw={include_kw}" if include_kw else ""
    print(f"[data] split={args.split}{subset} windows={len(eval_ds)} target_keys={eval_keys}")
    for name, e in entries.items():
        print(f"[ckpt:{name}] mode={e['mode']} path={e['path']} epoch={e['ckpt'].get('epoch')}")

    oa, ob = slices["joint_global_orient_r6d"]
    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        visible_sets = build_fixed_visible_sets([val_ds, eval_ds], seed=args.seed + k, min_k=k, max_k=k, split="val")
        metrics = OrderedDict(
            (name, MobilePoserMetricAccumulator(args.mobileposer_smpl, device, args.fps))
            for name in entries
        )
        error_sums = OrderedDict((name, init_error_sums(device)) for name in entries)
        with torch.no_grad():
            for batch in dl:
                # Keep a copy of the full IMU signal for undoing root normalization.
                # The sparse sampled imu below is still what the experts actually see.
                full_imu = batch["imu"].to(device, non_blocking=True)
                batch = apply_visible_imu_sampling(
                    batch, epoch=0, batch_idx=0, seed=args.seed + k, min_k=k, max_k=k, split="val",
                    fixed_per_dataset=True, visible_sets=visible_sets,
                )
                imu = batch["imu"].to(device, non_blocking=True)
                imu_mask = batch["imu_mask"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                gt_orient6 = target[..., oa:ob]
                for name, entry in entries.items():
                    pred_orient6, pred_motion, motion_key = predict_outputs(
                        entry, imu, imu_mask, lengths, gt_orient6, full_imu
                    )
                    metrics[name].add_batch(pred_orient6.float(), gt_orient6.float(), mask, lengths)
                    angle_num, angle_den = geodesic_sums_per_joint(pred_orient6, gt_orient6, mask, lengths)
                    ma, mb = slices[motion_key]
                    xyz_num, xyz_den = xyz_dist_sums_per_joint(pred_motion, target[..., ma:mb], mask, lengths)
                    add_error_sums(error_sums[name], angle_num, angle_den, xyz_num, xyz_den)
        print(f"\n=== active IMUs k={k} visible_sets={visible_sets} ===")
        for name, sums in error_sums.items():
            summarize_error_sums(name, sums)
        for name, metric in metrics.items():
            summarize_mobileposer(name, metric)


if __name__ == "__main__":
    main()
