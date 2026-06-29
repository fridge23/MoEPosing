#!/usr/bin/env python3
"""Preprocess raw TotalCapture + DIP_IMU data into training shards.

Produces shard files compatible with MaskedMotionDataset:
  imu [T, 15, 12], imu_mask [15], joint_orient_r6d [T, 28, 6],
  orient_mask [28], joint_delta [T, 28, 3], joint_mask [28], ...
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import CANONICAL_IMUS, CANONICAL_JOINTS, SMPL_JOINTS
from synth_imu import synthesize_imu

IMU_INDEX = {name: i for i, name in enumerate(CANONICAL_IMUS)}
JOINT_INDEX = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# TotalCapture raw text file joint names -> canonical joint names
TC_RAW_JOINT_MAP = {
    "Hips": "Pelvis", "Spine": "L5", "Spine1": "Spine1", "Spine2": "Spine2",
    "Spine3": "Spine3", "Neck": "Neck", "Head": "Head",
    "RightShoulder": "RightShoulder", "RightArm": "RightUpperArm",
    "RightForeArm": "RightForeArm", "RightHand": "RightHand",
    "LeftShoulder": "LeftShoulder", "LeftArm": "LeftUpperArm",
    "LeftForeArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightUpLeg": "RightUpperLeg", "RightLeg": "RightLowerLeg",
    "RightFoot": "RightFoot", "LeftUpLeg": "LeftUpperLeg",
    "LeftLeg": "LeftLowerLeg", "LeftFoot": "LeftFoot",
}

# TotalCapture IMU pkl: 6 sensors in MobilePoser order
TC_IMU_NAMES = ["LeftForeArm", "RightForeArm", "LeftUpperLeg", "RightUpperLeg", "Head", "Pelvis"]

# DIP_IMU: 17 sensor order (from DIP-IMU paper / standard Xsens MVN layout)
DIP_17_IMU_NAMES = [
    "Head", "T8", "Pelvis",
    "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm",
    "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "LeftHand", "RightHand",
    "LeftToe_skip", "RightToe_skip",
]


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).view(*q.shape[:-1], 3, 3)


def mat_to_r6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(*aa.shape[:-1], 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    return I * cos_a + sin_a * K + (1 - cos_a) * (axis.unsqueeze(-1) @ axis.unsqueeze(-2))


def smpl_local_to_global(local_rotmats: torch.Tensor) -> torch.Tensor:
    out = []
    for j, p in enumerate(SMPL_PARENTS[:local_rotmats.shape[1]]):
        out.append(local_rotmats[:, j] if p < 0 else out[p] @ local_rotmats[:, j])
    return torch.stack(out, dim=1)


def root_normalized_r6d(rotmats: torch.Tensor, src_names: List[str]):
    t = rotmats.shape[0]
    orient = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t, len(CANONICAL_JOINTS), 1, 1)
    mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    root_i = None
    for i, n in enumerate(src_names):
        cn = n if n in JOINT_INDEX else None
        if cn == "Pelvis":
            root_i = i
            break
    if root_i is None:
        return mat_to_r6d(orient), mask
    root_t = rotmats[:, root_i].transpose(-1, -2)
    for i, n in enumerate(src_names[:rotmats.shape[1]]):
        cn = n if n in JOINT_INDEX else None
        if cn is None:
            continue
        dst = JOINT_INDEX[cn]
        orient[:, dst] = root_t @ rotmats[:, i]
        mask[dst] = True
    return mat_to_r6d(orient), mask


def motion_targets(position: torch.Tensor, fps: float):
    root = position[:, :1]
    root_relative = position - root
    delta = torch.zeros_like(root_relative)
    delta[1:] = root_relative[1:] - root_relative[:-1]
    return {
        "joint_position_root_relative": root_relative,
        "joint_delta": delta,
        "joint_velocity": delta * float(fps),
        "joint_displacement": root_relative - root_relative[:1],
        "joint_step_distance": delta.norm(dim=-1, keepdim=True),
    }


def windows(t: int, seq_len: int, stride: int):
    if t <= seq_len:
        yield 0, t
        return
    for start in range(0, t - seq_len + 1, stride):
        yield start, start + seq_len
    last = t - seq_len
    if last % stride:
        yield last, t


class ShardWriter:
    def __init__(self, out_root: Path, shard_size: int = 128):
        self.out_root = out_root
        self.shard_size = shard_size
        self.samples = []
        out_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_root.glob("shard_*.pt"))
        self.shard_idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
        self.total = 0
        self.source_counts = {}

    def add(self, seq: Dict, seq_len: int, stride: int, min_len: int):
        t = int(seq["joint_delta"].shape[0])
        for start, end in windows(t, seq_len, stride):
            if end - start < min_len:
                continue
            sample = {}
            for key, value in seq.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == t:
                    sample[key] = value[start:end].contiguous()
                else:
                    sample[key] = value
            sample["start"] = start
            sample["end"] = end
            self.samples.append(sample)
            self.total += 1
            src = sample.get("source", "unknown")
            self.source_counts[src] = self.source_counts.get(src, 0) + 1
            if len(self.samples) >= self.shard_size:
                self._flush()

    def _flush(self):
        if not self.samples:
            return
        path = self.out_root / f"shard_{self.shard_idx:05d}.pt"
        torch.save({"samples": self.samples}, path)
        print(f"  [save] {path} ({len(self.samples)} samples)")
        self.samples = []
        self.shard_idx += 1

    def close(self):
        self._flush()


def parse_tc_positions(path: str) -> Tuple[torch.Tensor, List[str]]:
    with open(path) as f:
        header = f.readline().strip().split("\t")
        header = [h for h in header if h]
        rows = []
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) >= len(header) * 3:
                row = [float(v) for v in vals[:len(header) * 3]]
                rows.append(row)
    data = torch.tensor(rows, dtype=torch.float32)
    positions = data.view(-1, len(header), 3) / 100.0  # cm -> meters
    return positions, header


def parse_tc_orientations(path: str) -> Tuple[torch.Tensor, List[str]]:
    with open(path) as f:
        header = f.readline().strip().split("\t")
        header = [h for h in header if h]
        rows = []
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) >= len(header) * 4:
                row = [float(v) for v in vals[:len(header) * 4]]
                rows.append(row)
    data = torch.tensor(rows, dtype=torch.float32)
    quats = data.view(-1, len(header), 4)  # [w, x, y, z]
    rotmats = quat_to_mat(quats)
    return rotmats, header


def process_totalcapture(data_root: Path, writer: ShardWriter, args):
    tc_raw = data_root / "raw_data" / "raw" / "TotalCapture" / "raw"
    tc_imu = data_root / "raw_data" / "raw" / "TotalCapture" / "IMU"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    for subject_dir in sorted(tc_raw.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name  # S1, S2, S5
        subject_lower = subject.lower()  # s1, s2, s5

        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name  # acting1, walking1, etc.
            pos_file = seq_dir / "gt_skel_gbl_pos.txt"
            ori_file = seq_dir / "gt_skel_gbl_ori.txt"

            if not pos_file.exists() or not ori_file.exists():
                stats["skipped"] += 1
                continue

            imu_pkl = tc_imu / f"{subject_lower}_{seq_name}.pkl"
            has_real_imu = imu_pkl.exists()

            positions, pos_names = parse_tc_positions(str(pos_file))
            rotmats, ori_names = parse_tc_orientations(str(ori_file))

            t = min(positions.shape[0], rotmats.shape[0])
            positions = positions[:t]
            rotmats = rotmats[:t]

            # Map to canonical joints
            joints = torch.zeros(t, len(CANONICAL_JOINTS), 3)
            joint_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
            for src_i, raw_name in enumerate(pos_names):
                cn = TC_RAW_JOINT_MAP.get(raw_name)
                if cn and cn in JOINT_INDEX:
                    dst = JOINT_INDEX[cn]
                    joints[:, dst] = positions[:, src_i]
                    joint_mask[dst] = True

            # Map orientations to canonical joints
            canonical_names_for_ori = []
            for raw_name in ori_names:
                cn = TC_RAW_JOINT_MAP.get(raw_name, raw_name)
                canonical_names_for_ori.append(cn)

            orient6, orient_mask = root_normalized_r6d(rotmats, canonical_names_for_ori)

            # IMU data
            imu = torch.zeros(t, len(CANONICAL_IMUS), 12)
            imu_mask = torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)

            if has_real_imu:
                with open(imu_pkl, "rb") as f:
                    imu_data = pickle.load(f, encoding="latin1")
                acc = torch.tensor(imu_data["acc"], dtype=torch.float32)
                ori = torch.tensor(imu_data["ori"], dtype=torch.float32)
                t_imu = min(t, acc.shape[0], ori.shape[0])
                t = t_imu
                joints = joints[:t]
                orient6 = orient6[:t]
                imu = torch.zeros(t, len(CANONICAL_IMUS), 12)

                for src_i, name in enumerate(TC_IMU_NAMES[:acc.shape[1]]):
                    if name in IMU_INDEX:
                        dst = IMU_INDEX[name]
                        imu[:, dst, :9] = ori[:t, src_i].reshape(t, 9)
                        imu[:, dst, 9:] = acc[:t, src_i]
                        imu_mask[dst] = True
            else:
                imu, imu_mask = synthesize_imu(joints, joint_mask, 60.0)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            targets = motion_targets(joints, 60.0)
            seq = {
                "imu": imu,
                "imu_mask": imu_mask,
                "joint_mask": joint_mask,
                "joint_orient_r6d": orient6,
                "orient_mask": orient_mask,
                "source": "totalcapture",
                "motion": f"{subject}_{seq_name}",
                "fps": 60.0,
            }
            seq.update(targets)
            writer.add(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t
            print(f"  [tc] {subject}/{seq_name}: {t} frames, "
                  f"imus={int(imu_mask.sum())}, joints={int(joint_mask.sum())}")

    return stats


def process_dip_imu(data_root: Path, writer: ShardWriter, args):
    dip_root = data_root / "raw_data" / "raw" / "DIP_IMU"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    for subject_dir in sorted(dip_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_file in sorted(subject_dir.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as e:
                print(f"  [skip] {pkl_file}: {e}")
                stats["skipped"] += 1
                continue

            imu_raw = torch.tensor(data["imu"], dtype=torch.float32)  # [T, 17, 12]
            gt_pose = torch.tensor(data["gt"], dtype=torch.float32)  # [T, 72]
            t = min(imu_raw.shape[0], gt_pose.shape[0])
            imu_raw = imu_raw[:t]
            gt_pose = gt_pose[:t]

            # Map 17 DIP IMUs -> 15 canonical IMU slots
            imu = torch.zeros(t, len(CANONICAL_IMUS), 12)
            imu_mask = torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)
            for src_i, name in enumerate(DIP_17_IMU_NAMES[:imu_raw.shape[1]]):
                if name.endswith("_skip"):
                    continue
                if name in IMU_INDEX:
                    dst = IMU_INDEX[name]
                    imu[:, dst] = imu_raw[:, src_i]
                    imu_mask[dst] = True

            # SMPL pose -> global rotations
            local_aa = gt_pose.view(t, 24, 3)
            local_rot = axis_angle_to_matrix(local_aa)
            global_rot = smpl_local_to_global(local_rot)
            orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])

            # For joint positions: use SMPL FK with standard neutral body offsets
            # We compute positions from global rotations + standard bone vectors
            joint_pos, joint_mask = smpl_fk_positions(global_rot, t)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            targets = motion_targets(joint_pos, 60.0)
            seq = {
                "imu": imu,
                "imu_mask": imu_mask,
                "joint_mask": joint_mask,
                "joint_orient_r6d": orient6,
                "orient_mask": orient_mask,
                "source": "dip",
                "motion": f"{subject_dir.name}/{pkl_file.stem}",
                "fps": 60.0,
            }
            seq.update(targets)
            writer.add(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t
            print(f"  [dip] {subject_dir.name}/{pkl_file.stem}: {t} frames, "
                  f"imus={int(imu_mask.sum())}")

    return stats


# Standard SMPL neutral body zero-shape T-pose joint positions (meters)
# Extracted from the SMPL neutral body model at beta=0, pose=0
SMPL_TPOSE_JOINTS = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],  # 0 Pelvis
    [ 0.0611, -0.0907, -0.0100],  # 1 L_Hip
    [-0.0611, -0.0907, -0.0100],  # 2 R_Hip
    [ 0.0028,  0.1208, -0.0267],  # 3 Spine1
    [ 0.0759, -0.4879, -0.0318],  # 4 L_Knee
    [-0.0759, -0.4879, -0.0318],  # 5 R_Knee
    [ 0.0029,  0.2398, -0.0122],  # 6 Spine2
    [ 0.0916, -0.8724, -0.0093],  # 7 L_Ankle
    [-0.0916, -0.8724, -0.0093],  # 8 R_Ankle
    [ 0.0023,  0.3628,  0.0036],  # 9 Spine3
    [ 0.0803, -0.9319,  0.0970],  # 10 L_Foot
    [-0.0803, -0.9319,  0.0970],  # 11 R_Foot
    [ 0.0044,  0.5165,  0.0035],  # 12 Neck
    [ 0.0644,  0.5014, -0.0190],  # 13 L_Collar
    [-0.0644,  0.5014, -0.0190],  # 14 R_Collar
    [ 0.0031,  0.5972,  0.0329],  # 15 Head
    [ 0.1901,  0.4832, -0.0255],  # 16 L_Shoulder
    [-0.1901,  0.4832, -0.0255],  # 17 R_Shoulder
    [ 0.4194,  0.4567, -0.0320],  # 18 L_Elbow
    [-0.4194,  0.4567, -0.0320],  # 19 R_Elbow
    [ 0.6456,  0.4500, -0.0073],  # 20 L_Wrist
    [-0.6456,  0.4500, -0.0073],  # 21 R_Wrist
    [ 0.6862,  0.4404,  0.0345],  # 22 L_Hand
    [-0.6862,  0.4404,  0.0345],  # 23 R_Hand
], dtype=torch.float32)


def smpl_fk_positions(global_rot: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute joint positions from SMPL global rotations + standard bone vectors."""
    bone_vec = torch.zeros(24, 3)
    for j in range(24):
        p = SMPL_PARENTS[j]
        if p < 0:
            bone_vec[j] = SMPL_TPOSE_JOINTS[j]
        else:
            bone_vec[j] = SMPL_TPOSE_JOINTS[j] - SMPL_TPOSE_JOINTS[p]

    positions = torch.zeros(t, 24, 3)
    for j in range(24):
        p = SMPL_PARENTS[j]
        if p < 0:
            positions[:, j] = 0.0
        else:
            positions[:, j] = positions[:, p] + (global_rot[:, p] @ bone_vec[j].unsqueeze(-1)).squeeze(-1)

    # Map SMPL 24 joints -> 28 canonical joints
    canonical_pos = torch.zeros(t, len(CANONICAL_JOINTS), 3)
    canonical_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    for src_i, name in enumerate(SMPL_JOINTS[:24]):
        dst = JOINT_INDEX.get(name)
        if dst is not None:
            canonical_pos[:, dst] = positions[:, src_i]
            canonical_mask[dst] = True
    return canonical_pos, canonical_mask


def process_imuposer(data_root: Path, writer: ShardWriter, args):
    ip_root = data_root / "raw_data" / "raw" / "IMUPoser"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    # IMUPoser uses 10 IMUs: pose [T,72], imu [T,60] (10*6: 3 acc + 3 gyro per sensor)
    # Order from IMUPoser paper: head, chest, pelvis, l_upper_arm, r_upper_arm,
    #   l_lower_arm, r_lower_arm, l_upper_leg, r_upper_leg, l_lower_leg, r_lower_leg
    # But the pkl has imu [T, 60] = 10 sensors * 6 (orientation not stored as rotmat)
    # IMUPoser stores: acc[3] + gyro[3] per sensor, no orientation rotmat
    # Without orientation rotmat, we can only use synthesized IMU from FK positions
    # We'll use SMPL pose -> FK positions -> synthesize IMU

    for subject_dir in sorted(ip_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_file in sorted(subject_dir.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as e:
                print(f"  [skip] {pkl_file}: {e}")
                stats["skipped"] += 1
                continue

            pose = data["pose"]
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose)
            pose = pose.float()
            t = pose.shape[0]

            local_aa = pose.view(t, 24, 3)
            local_rot = axis_angle_to_matrix(local_aa)
            global_rot = smpl_local_to_global(local_rot)
            orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])
            joint_pos, joint_mask = smpl_fk_positions(global_rot, t)

            imu, imu_mask = synthesize_imu(joint_pos, joint_mask, 60.0)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            targets = motion_targets(joint_pos, 60.0)
            seq = {
                "imu": imu,
                "imu_mask": imu_mask,
                "joint_mask": joint_mask,
                "joint_orient_r6d": orient6,
                "orient_mask": orient_mask,
                "source": "imuposer",
                "motion": f"{subject_dir.name}/{pkl_file.stem}",
                "fps": 60.0,
            }
            seq.update(targets)
            writer.add(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t
            print(f"  [imuposer] {subject_dir.name}/{pkl_file.stem}: {t} frames")

    return stats


def process_amass(data_root: Path, writer: ShardWriter, args):
    amass_root = data_root / "raw_data" / "raw" / "AMASS"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    for subset_dir in sorted(amass_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        for npz_path in sorted(subset_dir.rglob("*.npz")):
            try:
                data = dict(np.load(str(npz_path), allow_pickle=True))
            except Exception as e:
                stats["skipped"] += 1
                continue

            poses = data.get("poses")
            if poses is None:
                stats["skipped"] += 1
                continue
            poses = torch.from_numpy(poses).float()
            fps = float(data.get("mocap_framerate", 60.0))
            t = poses.shape[0]
            if t < args.min_len:
                stats["skipped"] += 1
                continue

            # SMPLH has 52 joints; we only use first 24 (SMPL body)
            local_aa = poses[:, :72].view(t, 24, 3)
            local_rot = axis_angle_to_matrix(local_aa)
            global_rot = smpl_local_to_global(local_rot)
            orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])
            joint_pos, joint_mask = smpl_fk_positions(global_rot, t)

            # Resample to 60 Hz if needed
            if abs(fps - 60.0) > 1.0:
                out_t = max(2, int(round(t * 60.0 / fps)))
                if out_t < args.min_len:
                    stats["skipped"] += 1
                    continue
                j28, d3 = joint_pos.shape[1], joint_pos.shape[2]
                flat_pos = joint_pos.reshape(t, -1).T.unsqueeze(0)  # [1, 28*3, T]
                joint_pos = torch.nn.functional.interpolate(
                    flat_pos, size=out_t, mode="linear", align_corners=False
                ).squeeze(0).T.reshape(out_t, j28, d3)
                j28o, d6 = orient6.shape[1], orient6.shape[2]
                flat_ori = orient6.reshape(t, -1).T.unsqueeze(0)  # [1, 28*6, T]
                orient6 = torch.nn.functional.interpolate(
                    flat_ori, size=out_t, mode="linear", align_corners=False
                ).squeeze(0).T.reshape(out_t, j28o, d6)
                t = out_t

            imu, imu_mask = synthesize_imu(joint_pos, joint_mask, 60.0)

            targets = motion_targets(joint_pos, 60.0)
            rel_path = npz_path.relative_to(amass_root)
            seq = {
                "imu": imu,
                "imu_mask": imu_mask,
                "joint_mask": joint_mask,
                "joint_orient_r6d": orient6,
                "orient_mask": orient_mask,
                "source": f"amass_{subset_dir.name}",
                "motion": str(rel_path),
                "fps": 60.0,
            }
            seq.update(targets)
            writer.add(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t

        if stats["sequences"] % 50 == 0 and stats["sequences"] > 0:
            print(f"  [amass] {subset_dir.name}: {stats['sequences']} seqs so far")

    return stats


def process_hps_mvnx(data_root: Path, writer: ShardWriter, args):
    """HPS dataset: Xsens MVNX files with 23 segments + 17 real IMU sensors."""
    import xml.etree.ElementTree as ET

    mvnx_root = data_root / "unzipped" / "hps_mvnx" / "hps_mvnx"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    ns = {'mvnx': 'http://www.xsens.com/mvn/mvnx'}

    # Xsens 17 sensor names -> canonical IMU names
    XSENS_SENSOR_NAMES = [
        "Pelvis", "T8", "Head", "RightShoulder", "RightUpperArm",
        "RightForeArm", "RightHand", "LeftShoulder", "LeftUpperArm",
        "LeftForeArm", "LeftHand", "RightUpperLeg", "RightLowerLeg",
        "RightFoot", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    ]
    # Xsens 23 segment names
    XSENS_SEGMENT_NAMES = [
        "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
        "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
        "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe",
    ]

    for mvnx_path in sorted(mvnx_root.glob("*.mvnx")):
        try:
            tree = ET.parse(str(mvnx_path))
        except Exception as e:
            print(f"  [skip] {mvnx_path.name}: {e}")
            stats["skipped"] += 1
            continue

        root_el = tree.getroot()
        subj = root_el.find('.//mvnx:subject', ns)
        if subj is None:
            stats["skipped"] += 1
            continue

        fps_str = subj.get("frameRate", "30")
        fps = float(fps_str)
        frames_elem = subj.find('.//mvnx:frames', ns)
        frames = frames_elem.findall('mvnx:frame', ns)

        ori_list, pos_list, sensor_ori_list, sensor_acc_list = [], [], [], []
        for frame in frames:
            if frame.get("type") != "normal":
                continue
            ori_text = frame.findtext('mvnx:orientation', '', ns)
            pos_text = frame.findtext('mvnx:position', '', ns)
            s_ori_text = frame.findtext('mvnx:sensorOrientation', '', ns)
            s_acc_text = frame.findtext('mvnx:sensorFreeAcceleration', '', ns)
            if not ori_text or not pos_text:
                continue
            ori_list.append([float(x) for x in ori_text.split()])
            pos_list.append([float(x) for x in pos_text.split()])
            if s_ori_text:
                sensor_ori_list.append([float(x) for x in s_ori_text.split()])
            if s_acc_text:
                sensor_acc_list.append([float(x) for x in s_acc_text.split()])

        if not ori_list:
            stats["skipped"] += 1
            continue

        t = len(ori_list)
        n_seg = 23

        seg_ori = torch.tensor(ori_list, dtype=torch.float32).view(t, n_seg, 4)
        seg_pos = torch.tensor(pos_list, dtype=torch.float32).view(t, n_seg, 3)
        seg_rotmats = quat_to_mat(seg_ori)

        # Map 23 Xsens segments -> 28 canonical joints
        joints = torch.zeros(t, len(CANONICAL_JOINTS), 3)
        joint_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
        for src_i, name in enumerate(XSENS_SEGMENT_NAMES):
            if name in JOINT_INDEX:
                dst = JOINT_INDEX[name]
                joints[:, dst] = seg_pos[:, src_i]
                joint_mask[dst] = True

        orient6, orient_mask = root_normalized_r6d(seg_rotmats, XSENS_SEGMENT_NAMES)

        # Map 17 Xsens sensors -> 15 canonical IMU slots
        imu = torch.zeros(t, len(CANONICAL_IMUS), 12)
        imu_mask = torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)

        if sensor_ori_list and sensor_acc_list:
            s_ori = torch.tensor(sensor_ori_list, dtype=torch.float32)
            s_acc = torch.tensor(sensor_acc_list, dtype=torch.float32)
            t_s = min(t, s_ori.shape[0], s_acc.shape[0])
            s_ori = s_ori[:t_s].view(t_s, 17, 4)
            s_acc = s_acc[:t_s].view(t_s, 17, 3)
            s_rotmats = quat_to_mat(s_ori)
            for src_i, name in enumerate(XSENS_SENSOR_NAMES):
                if name in IMU_INDEX:
                    dst = IMU_INDEX[name]
                    imu[:t_s, dst, :9] = s_rotmats[:, src_i].reshape(t_s, 9)
                    imu[:t_s, dst, 9:] = s_acc[:, src_i]
                    imu_mask[dst] = True

        # Resample to 60 Hz if needed
        if abs(fps - 60.0) > 1.0:
            out_t = max(2, int(round(t * 60.0 / fps)))
            if out_t < args.min_len:
                stats["skipped"] += 1
                continue
            j28, d3 = joints.shape[1], joints.shape[2]
            flat_pos = joints.reshape(t, -1).T.unsqueeze(0)
            joints = torch.nn.functional.interpolate(
                flat_pos, size=out_t, mode="linear", align_corners=False
            ).squeeze(0).T.reshape(out_t, j28, d3)
            j28o, d6 = orient6.shape[1], orient6.shape[2]
            flat_ori = orient6.reshape(t, -1).T.unsqueeze(0)
            orient6 = torch.nn.functional.interpolate(
                flat_ori, size=out_t, mode="linear", align_corners=False
            ).squeeze(0).T.reshape(out_t, j28o, d6)
            n15, d12 = imu.shape[1], imu.shape[2]
            flat_imu = imu.reshape(t, -1).T.unsqueeze(0)
            imu = torch.nn.functional.interpolate(
                flat_imu, size=out_t, mode="linear", align_corners=False
            ).squeeze(0).T.reshape(out_t, n15, d12)
            t = out_t

        if t < args.min_len:
            stats["skipped"] += 1
            continue

        targets = motion_targets(joints, 60.0)
        seq = {
            "imu": imu,
            "imu_mask": imu_mask,
            "joint_mask": joint_mask,
            "joint_orient_r6d": orient6,
            "orient_mask": orient_mask,
            "source": "hps",
            "motion": mvnx_path.stem,
            "fps": 60.0,
        }
        seq.update(targets)
        writer.add(seq, args.seq_len, args.stride, args.min_len)
        stats["sequences"] += 1
        stats["frames"] += t
        print(f"  [hps] {mvnx_path.stem}: {t} frames, "
              f"imus={int(imu_mask.sum())}, joints={int(joint_mask.sum())}")

    return stats


def process_czumhad(data_root: Path, writer: ShardWriter, args):
    """CZU-MHAD: skeleton (.mat) with 22 joints. No real IMU -> synthesized."""
    import scipy.io as sio

    mhad_root = data_root / "unzipped" / "CZU-MHAD" / "CZU-MHAD"
    skel_dir = mhad_root / "skeleton_mat"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    if not skel_dir.exists():
        print("  [skip] CZU-MHAD skeleton_mat not found")
        return stats

    # CZU-MHAD skeleton: [88, T] = 22 joints * 4 values (x, y, z, confidence?)
    for mat_path in sorted(skel_dir.glob("*.mat")):
        try:
            d = sio.loadmat(str(mat_path))
            skel = torch.tensor(d["skeleton"], dtype=torch.float32)
        except Exception as e:
            stats["skipped"] += 1
            continue

        # skel shape: [88, T] => reshape to [T, 22, 4]
        n_vals, t = skel.shape
        n_joints = n_vals // 4
        if n_joints < 20 or t < args.min_len:
            stats["skipped"] += 1
            continue

        skel = skel.T.reshape(t, n_joints, 4)
        positions = skel[:, :, :3] / 100.0  # likely in cm -> meters
        # confidence = skel[:, :, 3]

        # CZU-MHAD uses Kinect skeleton with 22 joints
        # Only a subset maps to our canonical joints; use positions for synthesized IMU
        # Map the most common Kinect joints to canonical
        KINECT_MAP = {
            0: "Pelvis", 1: "L5", 2: "Neck", 3: "Head",
            4: "LeftShoulder", 5: "LeftUpperArm", 6: "LeftForeArm", 7: "LeftHand",
            8: "RightShoulder", 9: "RightUpperArm", 10: "RightForeArm", 11: "RightHand",
            12: "LeftUpperLeg", 13: "LeftLowerLeg", 14: "LeftFoot",
            16: "RightUpperLeg", 17: "RightLowerLeg", 18: "RightFoot",
        }
        joints = torch.zeros(t, len(CANONICAL_JOINTS), 3)
        joint_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
        for src_i, cn in KINECT_MAP.items():
            if src_i < n_joints and cn in JOINT_INDEX:
                dst = JOINT_INDEX[cn]
                joints[:, dst] = positions[:, src_i]
                joint_mask[dst] = True

        orient_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
        orient6 = mat_to_r6d(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(t, len(CANONICAL_JOINTS), 1, 1))

        # Resample 30fps -> 60fps
        out_t = t * 2
        j28, d3 = joints.shape[1], joints.shape[2]
        flat_pos = joints.reshape(t, -1).T.unsqueeze(0)
        joints = torch.nn.functional.interpolate(
            flat_pos, size=out_t, mode="linear", align_corners=False
        ).squeeze(0).T.reshape(out_t, j28, d3)
        j28o, d6 = orient6.shape[1], orient6.shape[2]
        flat_ori = orient6.reshape(t, -1).T.unsqueeze(0)
        orient6 = torch.nn.functional.interpolate(
            flat_ori, size=out_t, mode="linear", align_corners=False
        ).squeeze(0).T.reshape(out_t, j28o, d6)
        t = out_t

        imu, imu_mask = synthesize_imu(joints, joint_mask, 60.0)

        targets = motion_targets(joints, 60.0)
        seq = {
            "imu": imu,
            "imu_mask": imu_mask,
            "joint_mask": joint_mask,
            "joint_orient_r6d": orient6,
            "orient_mask": orient_mask,
            "source": "czumhad",
            "motion": mat_path.stem,
            "fps": 60.0,
        }
        seq.update(targets)
        writer.add(seq, args.seq_len, args.stride, args.min_len)
        stats["sequences"] += 1
        stats["frames"] += t

    print(f"  [czumhad] {stats['sequences']} seqs, {stats['frames']} frames")
    return stats


def process_mmfit(data_root: Path, writer: ShardWriter, args):
    """MM-Fit: 3D poses + IMU from smartwatch/earbud/smartphone."""
    mmfit_root = data_root / "unzipped" / "mm-fit" / "mm-fit"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    if not mmfit_root.exists():
        print("  [skip] MM-Fit not found")
        return stats

    # MM-Fit: 18 joints, acc+gyro from 4 sensor locations
    # Sensors: sw_l (smartwatch left=LeftHand), sw_r (right=RightHand),
    #          eb_l (earbud left=Head), sp_r (smartphone right=RightUpperLeg)
    MMFIT_SENSOR_MAP = {
        "sw_l": "LeftHand", "sw_r": "RightHand",
        "eb_l": "Head", "sp_r": "RightUpperLeg",
    }
    # MM-Fit 18 joints: 0=nose,1=neck,2=RShoulder,3=RElbow,4=RWrist,5=LShoulder,
    # 6=LElbow,7=LWrist,8=MidHip,9=RHip,10=RKnee,11=RAnkle,12=LHip,13=LKnee,14=LAnkle,
    # 15=REye,16=LEye,17=REar
    MMFIT_JOINT_MAP = {
        1: "Neck", 2: "RightShoulder", 3: "RightForeArm", 4: "RightHand",
        5: "LeftShoulder", 6: "LeftForeArm", 7: "LeftHand",
        8: "Pelvis", 9: "RightUpperLeg", 10: "RightLowerLeg", 11: "RightFoot",
        12: "LeftUpperLeg", 13: "LeftLowerLeg", 14: "LeftFoot",
    }

    for workout_dir in sorted(mmfit_root.iterdir()):
        if not workout_dir.is_dir():
            continue
        prefix = workout_dir.name
        pose_file = workout_dir / f"{prefix}_pose_3d.npy"
        if not pose_file.exists():
            continue

        try:
            pose_3d = np.load(str(pose_file))  # [3, T, 18] or [T, 18, 3]
        except Exception:
            stats["skipped"] += 1
            continue

        if pose_3d.ndim == 3 and pose_3d.shape[0] == 3:
            pose_3d = pose_3d.transpose(1, 2, 0)  # [T, 18, 3]
        t = pose_3d.shape[0]
        n_j = pose_3d.shape[1]

        positions = torch.tensor(pose_3d, dtype=torch.float32)
        joints = torch.zeros(t, len(CANONICAL_JOINTS), 3)
        joint_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
        for src_i, cn in MMFIT_JOINT_MAP.items():
            if src_i < n_j and cn in JOINT_INDEX:
                dst = JOINT_INDEX[cn]
                joints[:, dst] = positions[:, src_i]
                joint_mask[dst] = True

        orient_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
        orient6 = mat_to_r6d(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(t, len(CANONICAL_JOINTS), 1, 1))

        imu, imu_mask = synthesize_imu(joints, joint_mask, 30.0)

        # Load real IMU acc where available (overwrite synth)
        for sensor_key, can_name in MMFIT_SENSOR_MAP.items():
            acc_file = workout_dir / f"{prefix}_{sensor_key}_acc.npy"
            if acc_file.exists() and can_name in IMU_INDEX:
                try:
                    acc = np.load(str(acc_file))  # [N, 5] (timestamp, x, y, z, magnitude?)
                    if acc.shape[1] >= 4:
                        acc_xyz = torch.tensor(acc[:, 1:4], dtype=torch.float32)
                        # Resample IMU to match pose fps
                        if acc_xyz.shape[0] != t:
                            acc_flat = acc_xyz.T.unsqueeze(0)
                            acc_xyz = torch.nn.functional.interpolate(
                                acc_flat, size=t, mode="linear", align_corners=False
                            ).squeeze(0).T
                        dst = IMU_INDEX[can_name]
                        imu[:, dst, 9:] = acc_xyz[:t]
                        imu_mask[dst] = True
                except Exception:
                    pass

        # Resample 30fps -> 60fps
        out_t = t * 2
        if out_t < args.min_len:
            stats["skipped"] += 1
            continue
        j28, d3 = joints.shape[1], joints.shape[2]
        flat_pos = joints.reshape(t, -1).T.unsqueeze(0)
        joints = torch.nn.functional.interpolate(
            flat_pos, size=out_t, mode="linear", align_corners=False
        ).squeeze(0).T.reshape(out_t, j28, d3)
        j28o, d6 = orient6.shape[1], orient6.shape[2]
        flat_ori = orient6.reshape(t, -1).T.unsqueeze(0)
        orient6 = torch.nn.functional.interpolate(
            flat_ori, size=out_t, mode="linear", align_corners=False
        ).squeeze(0).T.reshape(out_t, j28o, d6)
        n15, d12 = imu.shape[1], imu.shape[2]
        flat_imu = imu.reshape(t, -1).T.unsqueeze(0)
        imu = torch.nn.functional.interpolate(
            flat_imu, size=out_t, mode="linear", align_corners=False
        ).squeeze(0).T.reshape(out_t, n15, d12)
        t = out_t

        targets = motion_targets(joints, 60.0)
        seq = {
            "imu": imu,
            "imu_mask": imu_mask,
            "joint_mask": joint_mask,
            "joint_orient_r6d": orient6,
            "orient_mask": orient_mask,
            "source": "mmfit",
            "motion": prefix,
            "fps": 60.0,
        }
        seq.update(targets)
        writer.add(seq, args.seq_len, args.stride, args.min_len)
        stats["sequences"] += 1
        stats["frames"] += t

    print(f"  [mmfit] {stats['sequences']} seqs, {stats['frames']} frames")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data60hz")
    parser.add_argument("--sources", default="totalcapture,dip,imuposer,amass,hps,czumhad,mmfit")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()

    out_root = Path(args.output)
    writer = ShardWriter(out_root, args.shard_size)
    data_root = Path(args.data_root)
    sources = {s.strip().lower() for s in args.sources.split(",")}
    all_stats = {}

    if "totalcapture" in sources:
        print("[convert] TotalCapture (raw positions + orientations + real IMU)")
        all_stats["totalcapture"] = process_totalcapture(data_root, writer, args)
        print(f"  -> {all_stats['totalcapture']}")

    if "dip" in sources:
        print("[convert] DIP_IMU (17 real IMUs + SMPL GT)")
        all_stats["dip"] = process_dip_imu(data_root, writer, args)
        print(f"  -> {all_stats['dip']}")

    if "imuposer" in sources:
        print("[convert] IMUPoser (SMPL GT + synthesized IMU)")
        all_stats["imuposer"] = process_imuposer(data_root, writer, args)
        print(f"  -> {all_stats['imuposer']}")

    if "amass" in sources:
        print("[convert] AMASS (SMPL GT + synthesized IMU)")
        all_stats["amass"] = process_amass(data_root, writer, args)
        print(f"  -> {all_stats['amass']}")

    if "hps" in sources:
        print("[convert] HPS (Xsens MVNX: 17 real IMUs + 23-segment skeleton)")
        all_stats["hps"] = process_hps_mvnx(data_root, writer, args)
        print(f"  -> {all_stats['hps']}")

    if "czumhad" in sources:
        print("[convert] CZU-MHAD (Kinect skeleton + synthesized IMU)")
        all_stats["czumhad"] = process_czumhad(data_root, writer, args)
        print(f"  -> {all_stats['czumhad']}")

    if "mmfit" in sources:
        print("[convert] MM-Fit (3D pose + IMU sensors)")
        all_stats["mmfit"] = process_mmfit(data_root, writer, args)
        print(f"  -> {all_stats['mmfit']}")

    writer.close()

    # Save metadata
    meta = {
        "canonical_imus": CANONICAL_IMUS,
        "canonical_joints": CANONICAL_JOINTS,
        "window": {"seq_len": args.seq_len, "stride": args.stride, "min_len": args.min_len},
        "total_windows": writer.total,
        "source_window_counts": writer.source_counts,
        "stats": all_stats,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    torch.save(meta, out_root / "metadata.pt")

    print(f"\n[done] {writer.total} windows -> {out_root}")
    for src, cnt in sorted(writer.source_counts.items()):
        print(f"  {src}: {cnt} windows")


if __name__ == "__main__":
    main()
