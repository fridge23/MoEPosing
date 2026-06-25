#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from schema import CANONICAL_IMUS, CANONICAL_JOINTS, SMPL_JOINTS, canonical_name
from synth_imu import synthesize_imu


IMU_INDEX = {name: i for i, name in enumerate(CANONICAL_IMUS)}
JOINT_INDEX = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
TARGET_FPS: Optional[float] = None


VIDIMU_JOINT_ALIASES = {
    "pelvis": "Pelvis",
    "left_hip": "LeftUpperLeg",
    "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg",
    "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "left_big_toe": "LeftToe",
    "right_big_toe": "RightToe",
    "torso": "T8",
    "neck": "Neck",
    "nose": "Head",
    "left_shoulder": "LeftShoulder",
    "right_shoulder": "RightShoulder",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
    "left_index_knuckle": "LeftHandIndex",
    "right_index_knuckle": "RightHandIndex",
}

VIDIMU_IMU_ALIASES = {
    "qsBACK": "T8",
    "qsLUA": "LeftUpperArm",
    "qsRUA": "RightUpperArm",
    "qsLLA": "LeftForeArm",
    "qsRLA": "RightForeArm",
}

ULTRA_MARKER_ALIASES = {
    "HEAD": "Head",
    "C7": "Neck",
    "CLAV": "T8",
    "STUR": "T8",
    "LSHO": "LeftShoulder",
    "RSHO": "RightShoulder",
    "LHAR2": "LeftUpperArm",
    "RHAR2": "RightUpperArm",
    "LHME": "LeftForeArm",
    "RHME": "RightForeArm",
    "LMCM": "LeftHand",
    "RMCM": "RightHand",
    "LASI": "Pelvis",
    "RASI": "Pelvis",
    "LPSI": "Pelvis",
    "RPSI": "Pelvis",
    "LLFC": "LeftUpperLeg",
    "RLFC": "RightUpperLeg",
    "LKNE": "LeftLowerLeg",
    "RKNE": "RightLowerLeg",
    "LLAN": "LeftFoot",
    "RLAN": "RightFoot",
    "LHEE": "LeftFoot",
    "RHEE": "RightFoot",
    "LTOE": "LeftToe",
    "RTOE": "RightToe",
}

SOURCE_UNIT_ASSUMPTIONS = {
    "mobileposer": {
        "input_position": "processed SMPL joint tensor; treated as meters",
        "output_position": "meters, root-relative targets after motion_targets()",
        "imu_acceleration": "processed MobilePoser acceleration tensor; not rescaled here",
        "orientation": "9D rotation matrix",
        "fps": "args.mobileposer_fps, default 60 Hz (MobilePoser process.py TARGET_FPS=60)",
        "confidence": "assumed from MobilePoser preprocessing",
    },
    "dynaip": {
        "input_position": "DynaIP extracted Xsens/CIP joint position; treated as meters",
        "output_position": "meters, root-relative targets after motion_targets()",
        "imu_acceleration": "Xsens/CIP free acceleration from DynaIP extract; not rescaled here",
        "orientation": "quaternion converted to 9D rotation matrix when needed",
        "fps": "per-file framerate, resampled by --target-fps when set",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "ceti_age_kinematics": {
        "input_position": "CeTI *_motion.tsv position columns in centimeters",
        "conversion": "position_cm / 100.0",
        "output_position": "meters, root-relative targets after motion_targets()",
        "imu_acceleration": "synthesized world-frame free acceleration from joint positions (2nd time-difference * fps^2)",
        "orientation": "synthesized per-segment orientation built from joint positions (bone direction + reference plane)",
        "fps": "inferred from Timestamp milliseconds, normally 100 Hz; resampled by --target-fps when set",
        "confidence": "explicit units from *_channels.tsv",
    },
    "vidimu": {
        "input_position": "video/IK-derived CSV joint coordinates in millimeters",
        "conversion": "position_mm / 1000.0; then 180deg-about-X (Y-down camera frame -> world Y-up)",
        "output_position": "meters, root-relative targets after motion_targets()",
        "imu_acceleration": "synthesized world-frame free acceleration from joint positions (2nd time-difference * fps^2)",
        "orientation": "synthesized per-segment orientation built from positions; real .raw quaternions NOT used (unregistered IMU frame)",
        "fps": "50 Hz",
        "confidence": "scale inferred from file magnitudes and OpenSim/VIDIMU format",
    },
    "ultra_mocap": {
        "input_position": "Vicon .trc marker positions in millimeters",
        "conversion": "marker_mm / 1000.0; mapped markers for each canonical joint are averaged",
        "output_position": "meters, root-relative targets after motion_targets()",
        "imu_acceleration": "synthesized world-frame free acceleration from marker joint positions; raw Delsys IMU (gravity-containing) NOT used",
        "orientation": "synthesized per-segment orientation built from positions; raw IMU orientation unavailable",
        "fps": ".trc DataRate, normally 100 Hz; resampled by --target-fps when set",
        "confidence": "marker units explicit in .trc; IMU is synthetic",
    },
}


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )


def mat_to_r6d(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [...,3,3] -> 6D (first two columns), the standard r6d encoding."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def canonical_rotmats(rotmats: torch.Tensor, src_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map source global rotations into canonical joint order, identity-filled for missing joints."""
    t = rotmats.shape[0]
    orient = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t, len(CANONICAL_JOINTS), 1, 1)
    mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    for i, n in enumerate(src_names[: rotmats.shape[1]]):
        dst = JOINT_INDEX.get(canonical_name(str(n)))
        if dst is None:
            continue
        orient[:, dst] = rotmats[:, i].float()
        mask[dst] = True
    return orient, mask


def global_r6d(rotmats: torch.Tensor, src_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-joint global rotation matrices -> canonical global r6d orientation target."""
    orient, mask = canonical_rotmats(rotmats, src_names)
    return mat_to_r6d(orient), mask


def root_normalized_rotmats(rotmats: torch.Tensor, src_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map source global rotations into canonical root-normalized matrices."""
    t = rotmats.shape[0]
    orient = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t, len(CANONICAL_JOINTS), 1, 1)
    mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    root_i = next((i for i, n in enumerate(src_names) if canonical_name(str(n)) == "Pelvis"), None)
    if root_i is None:
        return orient, mask
    root_t = rotmats[:, root_i].transpose(-1, -2)  # [T,3,3]
    for i, n in enumerate(src_names[: rotmats.shape[1]]):
        dst = JOINT_INDEX.get(canonical_name(str(n)))
        if dst is None:
            continue
        orient[:, dst] = root_t @ rotmats[:, i]
        mask[dst] = True
    return orient, mask


def root_normalized_r6d(rotmats: torch.Tensor, src_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-joint global rotation matrices [T,Jsrc,3,3] (+ source joint names) ->
    root(pelvis)-normalized r6d orientation target [T,28,6] over CANONICAL_JOINTS,
    and orient_mask [28]. Root-normalization (R_pelvis^T @ R_joint) makes the target
    invariant to global heading and (being rotation) to body size."""
    orient, mask = root_normalized_rotmats(rotmats, src_names)
    return mat_to_r6d(orient), mask


# SMPL 24-joint kinematic-tree parents (matches SMPL_JOINTS ordering in schema.py).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def smpl_local_to_global(local_rotmats: torch.Tensor) -> torch.Tensor:
    """SMPL *local* (relative-to-parent) joint rotations [T,24,3,3] -> *global* rotations.

    MobilePoser stores SMPL pose (per-joint, relative to parent); the orientation /
    pose-change targets expect global rotations (DynaIP feeds global Xsens orientations).
    R_global[j] = R_global[parent[j]] @ R_local[j], parents precede children so a single
    forward pass suffices."""
    local = local_rotmats.float()
    out: List[torch.Tensor] = []
    for j, p in enumerate(SMPL_PARENTS[: local.shape[1]]):
        out.append(local[:, j] if p < 0 else out[p] @ local[:, j])
    return torch.stack(out, dim=1)


def self_pose_change_targets(
    position: torch.Tensor,
    joint_mask: torch.Tensor,
    rotmats: torch.Tensor,
    src_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Adjacent-frame pose-change targets in each joint's own previous frame.

    Unlike joint_orient_r6d, these targets are not root-normalized. The rotation
    target is the joint's own global pose change:

        delta_R[t] = R_joint[t-1].T @ R_joint[t]

    Starting from the first frame's global joint pose, integrating these deltas
    reconstructs the full global pose trajectory. The translation target mirrors
    this convention:

        delta_x_local[t] = R_joint[t-1].T @ (x_joint[t] - x_joint[t-1])

    so the movement vector is expressed in the joint's own previous coordinate
    frame and does not require root un-normalization at inference time.
    """
    t = min(position.shape[0], rotmats.shape[0])
    n_joints = len(CANONICAL_JOINTS)
    global_rot, orient_mask = canonical_rotmats(rotmats[:t].float(), src_names)
    rot_delta = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t, n_joints, 1, 1)
    if t > 1 and orient_mask.any():
        rot_delta[1:, orient_mask] = global_rot[:-1, orient_mask].transpose(-1, -2) @ global_rot[1:, orient_mask]

    pos = position[:t].float()
    local_delta = torch.zeros(t, n_joints, 3, dtype=torch.float32)
    local_mask = joint_mask.bool() & orient_mask
    if t > 1 and local_mask.any():
        step = pos[1:, local_mask] - pos[:-1, local_mask]
        local_delta[1:, local_mask] = (global_rot[:-1, local_mask].transpose(-1, -2) @ step.unsqueeze(-1)).squeeze(-1)

    return {
        "joint_global_orient_r6d": mat_to_r6d(global_rot),
        "global_orient_mask": orient_mask,
        "joint_rot_delta_r6d": mat_to_r6d(rot_delta),
        "rot_delta_mask": orient_mask,
        "joint_delta_local": local_delta,
        "joint_delta_local_mask": local_mask,
    }

def identity_9d(t: int, n: int) -> torch.Tensor:
    eye = torch.eye(3, dtype=torch.float32).reshape(9)
    return eye.view(1, 1, 9).repeat(t, n, 1)


def zeros_imu(t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(t, len(CANONICAL_IMUS), 12), torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)


def zeros_joints(t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(t, len(CANONICAL_JOINTS), 3), torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)


def _with_orient(extra: Dict, orient6, orient_mask) -> Dict:
    """Attach r6d orientation target + its mask to a sequence's extra dict when present."""
    if orient6 is not None and orient_mask is not None and bool(orient_mask.any()):
        extra = dict(extra)
        extra["joint_orient_r6d"] = orient6
        extra["orient_mask"] = orient_mask
    return extra


def _with_pose_targets(extra: Dict, orient6, orient_mask, pose_change: Optional[Dict] = None) -> Dict:
    extra = _with_orient(extra, orient6, orient_mask)
    if pose_change:
        extra = dict(extra)
        extra.update(pose_change)
    return extra


def motion_targets(position: torch.Tensor, fps: float) -> Dict[str, torch.Tensor]:
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




def resample_time_tensor(x: torch.Tensor, fps_in: float, fps_out: float) -> torch.Tensor:
    if x.shape[0] < 2 or fps_in <= 0 or fps_out <= 0 or abs(fps_in - fps_out) < 1e-6:
        return x
    out_t = max(2, int(round(x.shape[0] * float(fps_out) / float(fps_in))))
    old_shape = x.shape
    flat = x.float().reshape(old_shape[0], -1).T.unsqueeze(0)
    y = torch.nn.functional.interpolate(flat, size=out_t, mode="linear", align_corners=False)
    return y.squeeze(0).T.reshape(out_t, *old_shape[1:])

def windows(t: int, seq_len: int, stride: int) -> Iterable[Tuple[int, int]]:
    if t <= seq_len:
        yield 0, t
        return
    for start in range(0, t - seq_len + 1, stride):
        yield start, start + seq_len
    last = t - seq_len
    if last % stride:
        yield last, t


def finite_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x.flatten(1)).all(dim=1)


class ShardWriter:
    def __init__(self, out_root: Path, shard_size: int):
        self.out_root = out_root
        self.shard_size = shard_size
        self.samples: List[Dict] = []
        out_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_root.glob("shard_*.pt"))
        self.shard_idx = 0
        if existing:
            self.shard_idx = int(existing[-1].stem.split("_")[-1]) + 1
        self.total = 0
        self.source_counts: Dict[str, int] = {}

    def add_sequence(self, seq: Dict, seq_len: int, stride: int, min_len: int):
        t = int(seq["joint_delta"].shape[0])
        for start, end in windows(t, seq_len, stride):
            if end - start < min_len:
                continue
            sample = {}
            for key, value in seq.items():
                if isinstance(value, torch.Tensor) and value.shape[:1] == (t,):
                    sample[key] = value[start:end].contiguous()
                else:
                    sample[key] = value
            sample["start"] = start
            sample["end"] = end
            self.samples.append(sample)
            self.total += 1
            self.source_counts[sample["source"]] = self.source_counts.get(sample["source"], 0) + 1
            if len(self.samples) >= self.shard_size:
                self.flush()

    def flush(self):
        if not self.samples:
            return
        path = self.out_root / f"shard_{self.shard_idx:05d}.pt"
        torch.save({"samples": self.samples}, path)
        print(f"[save] {path} ({len(self.samples)} samples)")
        self.samples = []
        self.shard_idx += 1

    def close(self):
        self.flush()


def build_sequence(
    source: str,
    split: str,
    motion: str,
    fps: float,
    imu: torch.Tensor,
    imu_mask: torch.Tensor,
    joints: torch.Tensor,
    joint_mask: torch.Tensor,
    quality: str,
    notes: str,
    extra: Optional[Dict] = None,
) -> Optional[Dict]:
    t = min(imu.shape[0], joints.shape[0])
    if t < 2:
        return None
    imu = imu[:t].float()
    joints = joints[:t].float()
    valid = finite_rows(imu) & finite_rows(joints)
    if valid.sum().item() < 2:
        return None
    imu = imu[valid]
    joints = joints[valid]
    extra_values: Dict = {}
    if extra:
        for key, value in extra.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (t,):
                extra_values[key] = value[:t][valid].float()
            else:
                extra_values[key] = value

    effective_fps = float(fps)
    target_fps = TARGET_FPS
    if target_fps is not None and abs(effective_fps - float(target_fps)) > 1e-6:
        before_t = imu.shape[0]
        imu = resample_time_tensor(imu, effective_fps, float(target_fps))
        joints = resample_time_tensor(joints, effective_fps, float(target_fps))
        for key, value in list(extra_values.items()):
            if isinstance(value, torch.Tensor) and value.shape[:1] == (before_t,):
                extra_values[key] = resample_time_tensor(value, effective_fps, float(target_fps))
        effective_fps = float(target_fps)

    targets = motion_targets(joints, effective_fps)
    seq = {
        "imu": imu,
        "imu_mask": imu_mask.bool(),
        "joint_mask": joint_mask.bool(),
        "fps": float(effective_fps),
        "original_fps": float(fps),
        "source": source,
        "split": split,
        "motion": motion,
        "quality": quality,
        "notes": notes,
    }
    seq.update(targets)
    for key, value in extra_values.items():
        seq[key] = value
    return seq


def convert_mobileposer(data_root: Path, writer: ShardWriter, args) -> Dict:
    import torch

    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    # eval/totalcapture.pt and eval/TotalCapture.pt duplicate the top-level
    # TotalCapture.pt (which also has joint positions); skip the redundant copies.
    redundant = {"eval/totalcapture.pt", "eval/TotalCapture.pt"}
    for path in sorted((data_root / "mobileposer").glob("**/*.pt")):
        obj = torch.load(path, map_location="cpu")
        rel = path.relative_to(data_root / "mobileposer").as_posix()
        # Need IMU (acc/ori) + pose for the orientation target; 'joint' is optional
        # (the benchmark eval files DIP/IMUPoser/TotalCapture have no joint positions).
        if not all(k in obj for k in ["acc", "ori", "pose"]) or rel in redundant:
            stats["skipped"] += 1
            continue
        poses = obj["pose"]
        joint_seqs = obj.get("joint")
        for idx in range(len(obj["acc"])):
            acc, ori = obj["acc"][idx], obj["ori"][idx]
            t = min(acc.shape[0], ori.shape[0])
            if joint_seqs is not None:
                t = min(t, joint_seqs[idx].shape[0])
            imu, imu_mask = zeros_imu(t)
            orient6 = orient_mask = None
            pose = poses[idx]
            pose_global = None
            if pose.dim() == 4 and tuple(pose.shape[-2:]) == (3, 3):  # [T,24,3,3] LOCAL SMPL pose
                pose_global = smpl_local_to_global(pose[:t].float())  # -> global, matches target convention
                orient6, orient_mask = root_normalized_r6d(pose_global, SMPL_JOINTS)
            if orient_mask is None or not orient_mask.any():
                stats["skipped"] += 1
                continue
            # MobilePoser source IMU order (process.py vi_mask/ji_mask, config.all_imu_ids):
            # 0 left wrist, 1 right wrist, 2 left thigh, 3 right thigh, 4 head, 5 pelvis.
            # Legs are THIGH (UpperLeg, verts 876/4362), not LowerLeg; arms map to ForeArm
            # (ji_mask 18/19 = elbow). imuposer files have only the first 5 (no pelvis).
            names = ["LeftForeArm", "RightForeArm", "LeftUpperLeg", "RightUpperLeg", "Head", "Pelvis"]
            for src_i, name in enumerate(names[: acc.shape[1]]):
                dst = IMU_INDEX[name]
                imu[:, dst, :9] = ori[:t, src_i].reshape(t, 9).float()
                imu[:, dst, 9:] = acc[:t, src_i].float()
                imu_mask[dst] = True
            joints, joint_mask = zeros_joints(t)
            if joint_seqs is not None:
                joint = joint_seqs[idx]
                for src_i, name in enumerate(SMPL_JOINTS[: joint.shape[1]]):
                    dst = JOINT_INDEX.get(canonical_name(name))
                    if dst is not None:
                        joints[:, dst] = joint[:t, src_i].float()
                        joint_mask[dst] = True
            seq = build_sequence(
                "mobileposer",
                "eval" if rel.startswith("eval/") else "train",
                f"{rel}:{idx}",
                args.mobileposer_fps,
                imu,
                imu_mask,
                joints,
                joint_mask,
                "primary",
                "processed synthetic/real sparse IMU with SMPL 24-joint positions",
                _with_pose_targets(
                    {"unit_info": SOURCE_UNIT_ASSUMPTIONS["mobileposer"]},
                    orient6,
                    orient_mask,
                    self_pose_change_targets(joints, joint_mask, pose_global, SMPL_JOINTS),
                ),
            )
            if seq:
                writer.add_sequence(seq, args.seq_len, args.stride, args.min_len)
                stats["sequences"] += 1
                stats["frames"] += t
    return stats




def convert_dynaip_extract(data_root: Path, writer: ShardWriter, args) -> Dict:
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    extract_root = Path('datasets/extract')
    for path in sorted(extract_root.glob('**/*.pt')):
        if 'cip' in path.parts and 'cip_full' not in path.parts:
            continue
        dataset = 'cip' if path.parent.name == 'cip_full' else path.parent.name
        try:
            data = torch.load(path, map_location='cpu')
            acc = data['imu']['free acceleration'].float()
            ori = data['imu'].get('calibrated orientation', data['imu'].get('orientation')).float()
            imu_names = data['imu'].get('name') or []
            pos = data['joint']['position'].float()
            joint_names = data['joint'].get('name') or []
            fps = dynaip_true_fps(data, pos.shape[0])
        except Exception as exc:
            stats['skipped'] += 1
            print(f'[skip] dynaip {path}: {type(exc).__name__}: {exc}')
            continue
        t = min(acc.shape[0], ori.shape[0], pos.shape[0])
        imu, imu_mask = zeros_imu(t)
        ori9 = quat_to_mat(ori[:t].reshape(-1, 4)).view(t, ori.shape[1], 9) if ori.shape[-1] == 4 else ori[:t].reshape(t, ori.shape[1], 9)
        for src_i, raw_name in enumerate(imu_names[:acc.shape[1]]):
            name = canonical_name(str(raw_name))
            dst = IMU_INDEX.get(name)
            if dst is None:
                continue
            imu[:, dst, :9] = ori9[:, src_i]
            imu[:, dst, 9:] = acc[:t, src_i]
            imu_mask[dst] = True
        joints, joint_mask = zeros_joints(t)
        if not joint_names:
            joint_names = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'RightShoulder', 'RightUpperArm',
                           'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                           'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe', 'LeftUpperLeg', 'LeftLowerLeg',
                           'LeftFoot', 'LeftToe']
        for src_i, raw_name in enumerate(joint_names[:pos.shape[1]]):
            name = canonical_name(str(raw_name))
            dst = JOINT_INDEX.get(name)
            if dst is None:
                continue
            joints[:, dst] = pos[:t, src_i]
            joint_mask[dst] = True
        if not imu_mask.any() or not joint_mask.any():
            stats['skipped'] += 1
            continue
        orient6 = orient_mask = None
        jo = data['joint'].get('orientation')
        if jo is not None and jo.shape[-1] == 4:
            jr = quat_to_mat(jo[:t].reshape(-1, 4)).view(t, jo.shape[1], 3, 3)
            orient6, orient_mask = root_normalized_r6d(jr, joint_names)
        seq = build_sequence(
            'dynaip_' + dataset,
            'all',
            path.name,
            fps,
            imu,
            imu_mask,
            joints,
            joint_mask,
            'primary',
            'DynaIP extracted Xsens/CIP data with calibrated IMU orientation and full-body joint positions',
            _with_pose_targets(
                {'unit_info': SOURCE_UNIT_ASSUMPTIONS['dynaip']},
                orient6,
                orient_mask,
                self_pose_change_targets(joints, joint_mask, jr, joint_names) if jo is not None and jo.shape[-1] == 4 else None,
            ),
        )
        if seq:
            writer.add_sequence(seq, args.seq_len, args.stride, args.min_len)
            stats['sequences'] += 1
            stats['frames'] += t
    return stats


def convert_ceti(data_root: Path, writer: ShardWriter, args) -> Dict:
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    root = data_root / "CeTI-Age-Kinematics" / "ceti-age-kinematics"
    for path in sorted(root.glob("**/*_motion.tsv")):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
        except Exception:
            stats["skipped"] += 1
            continue
        if not rows:
            continue
        t = len(rows)
        joints, joint_mask = zeros_joints(t)
        accum: Dict[str, List[torch.Tensor]] = {}
        for raw_name in JOINT_INDEX:
            cols = [f"{raw_name}_position_x", f"{raw_name}_position_y", f"{raw_name}_position_z"]
            if cols[0] not in rows[0]:
                continue
            vals = torch.tensor([[float(r[c]) for c in cols] for r in rows], dtype=torch.float32) / 100.0
            accum.setdefault(raw_name, []).append(vals)
        for name, vals in accum.items():
            joints[:, JOINT_INDEX[name]] = torch.stack(vals).mean(dim=0)
            joint_mask[JOINT_INDEX[name]] = True
        if joint_mask.sum().item() == 0:
            stats["skipped"] += 1
            continue
        fps = infer_fps([float(r["Timestamp"]) for r in rows[: min(100, t)]], scale=0.001) or 100.0
        imu, imu_mask = synthesize_imu(joints, joint_mask, fps)
        seq = build_sequence(
            "ceti_age_kinematics",
            "all",
            path.relative_to(root).as_posix(),
            fps,
            imu,
            imu_mask,
            joints,
            joint_mask,
            "synthetic_imu",
            "Rokoko full-body kinematics; IMU synthesized from joint positions (free-acc + position-built orientation)",
            {"unit_info": SOURCE_UNIT_ASSUMPTIONS["ceti_age_kinematics"]},
        )
        if seq:
            writer.add_sequence(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t
    return stats


def dynaip_true_fps(data: Dict, n_stored: int, default: float = 60.0) -> float:
    """True fps of the STORED dynaip-extract frames.

    read_mvnx downsamples joint/imu arrays to ~60 Hz (step = frameRate//60) but keeps
    the original `framerate` field (e.g. 240) and an un-downsampled `timestamp ms`,
    so `data['framerate']` is wrong for 240 Hz sources. Recover the real rate from the
    number of stored frames over the timestamp span; fall back to 60 when no timestamp
    (virginia clips / cip xlsx are saved without one and are already 60 Hz).
    """
    ts = data.get('timestamp ms')
    try:
        if ts is not None and len(ts) > 1:
            span_s = (float(ts[-1]) - float(ts[0])) / 1000.0
            if span_s > 0:
                fps = (n_stored - 1) / span_s
                if 20.0 <= fps <= 120.0:
                    return float(fps)
    except Exception:
        pass
    return default


def infer_fps(times: List[float], scale: float = 1.0) -> Optional[float]:
    if len(times) < 3:
        return None
    diffs = [(b - a) * scale for a, b in zip(times, times[1:]) if b > a]
    if not diffs:
        return None
    diffs.sort()
    dt = diffs[len(diffs) // 2]
    return 1.0 / dt if dt > 0 else None


def read_vidimu_raw(path: Path, out_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    imu, mask = zeros_imu(out_len)
    if not path.exists():
        return imu, mask
    by_sensor: Dict[str, List[Tuple[float, List[float]]]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            sensor = row[0]
            if sensor not in VIDIMU_IMU_ALIASES:
                continue
            try:
                quat = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                ts = float(row[5])
            except ValueError:
                continue
            by_sensor.setdefault(sensor, []).append((ts, quat))
    for sensor, values in by_sensor.items():
        dst_name = VIDIMU_IMU_ALIASES[sensor]
        dst = IMU_INDEX[dst_name]
        values.sort(key=lambda x: x[0])
        if not values:
            continue
        src = torch.tensor([q for _, q in values], dtype=torch.float32)
        idx = torch.linspace(0, len(src) - 1, out_len).round().long().clamp(0, len(src) - 1)
        imu[:, dst, :9] = quat_to_mat(src[idx])
        mask[dst] = True
    return imu, mask


def convert_vidimu(data_root: Path, writer: ShardWriter, args) -> Dict:
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    root = data_root / "vidimu" / "dataset" / "videoandimusync"
    for path in sorted(root.glob("**/*.csv")):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        t = len(rows)
        joints, joint_mask = zeros_joints(t)
        for src, dst_name in VIDIMU_JOINT_ALIASES.items():
            cols = [f"{src}_x", f"{src}_y", f"{src}_z"]
            cols = [c if c in rows[0] else f" {c}" for c in cols]
            if not all(c in rows[0] for c in cols):
                continue
            vals = torch.tensor([[float(r[c]) for c in cols] for r in rows], dtype=torch.float32) / 1000.0
            # BodyTrack CSV is a camera frame (Y-down, Z = depth-from-camera). Rotate
            # 180 deg about X -> world Y-up, right-handed, consistent with other sources.
            vals[:, 1] = -vals[:, 1]
            vals[:, 2] = -vals[:, 2]
            dst = JOINT_INDEX[dst_name]
            joints[:, dst] = vals
            joint_mask[dst] = True
        if joint_mask.sum().item() == 0:
            stats["skipped"] += 1
            continue
        imu, imu_mask = synthesize_imu(joints, joint_mask, 50.0)
        seq = build_sequence(
            "vidimu",
            "all",
            path.relative_to(root).as_posix(),
            50.0,
            imu,
            imu_mask,
            joints,
            joint_mask,
            "synthetic_imu",
            "video/IK joints (camera->world frame corrected, meters); IMU synthesized from positions (free-acc + position-built orientation)",
            {"unit_info": SOURCE_UNIT_ASSUMPTIONS["vidimu"]},
        )
        if seq:
            writer.add_sequence(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += t
    return stats


def parse_ultra_trc(path: Path) -> Tuple[torch.Tensor, torch.Tensor, float]:
    with open(path) as f:
        lines = [next(f).rstrip("\n") for _ in range(6)]
        rest = f.readlines()
    rates = lines[2].split("\t")
    fps = float(rates[0]) if rates and rates[0] else 100.0
    markers = [x for x in lines[3].split("\t")[2:] if x]
    data = []
    for line in rest:
        parts = line.strip().split("\t")
        if len(parts) < 2 + 3 * len(markers):
            continue
        try:
            data.append([float(x) for x in parts[2 : 2 + 3 * len(markers)]])
        except ValueError:
            continue
    raw = torch.tensor(data, dtype=torch.float32).view(-1, len(markers), 3) / 1000.0
    joints, joint_mask = zeros_joints(raw.shape[0])
    buckets: Dict[str, List[torch.Tensor]] = {}
    for i, marker in enumerate(markers):
        name = ULTRA_MARKER_ALIASES.get(marker)
        if name:
            buckets.setdefault(name, []).append(raw[:, i])
    for name, vals in buckets.items():
        joints[:, JOINT_INDEX[name]] = torch.stack(vals).mean(dim=0)
        joint_mask[JOINT_INDEX[name]] = True
    return joints, joint_mask, fps


def read_ultra_imu(path: Path, out_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imu, mask = zeros_imu(out_len)
    gyro = torch.zeros(out_len, len(CANONICAL_IMUS), 3)
    names = ["Pelvis", "T8", "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm"]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return imu, mask, gyro
    idx = torch.linspace(0, len(rows) - 1, out_len).round().long().tolist()
    imu[:, :, :9] = identity_9d(out_len, len(CANONICAL_IMUS))
    for src_i, name in enumerate(names, start=1):
        dst = IMU_INDEX[name]
        acc_cols = [f"ACCX{src_i}", f"ACCY{src_i}", f"ACCZ{src_i}"]
        gyro_cols = [f"GYROX{src_i}", f"GYROY{src_i}", f"GYROZ{src_i}"]
        if not all(c in rows[0] for c in acc_cols + gyro_cols):
            continue
        imu[:, dst, 9:] = torch.tensor([[float(rows[i][c]) for c in acc_cols] for i in idx]) / 1000.0
        gyro[:, dst] = torch.tensor([[float(rows[i][c]) for c in gyro_cols] for i in idx])
        mask[dst] = True
    return imu, mask, gyro


def convert_ultra(data_root: Path, writer: ShardWriter, args) -> Dict:
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    root = data_root / "ULTRA-MoCap"
    for imu_path in sorted(root.glob("ULTra-MoCap-raw-*/*/*/IMU/*.csv")):
        marker_path = Path(str(imu_path).replace("/IMU/", "/Raw Marker/").replace(".csv", ".trc"))
        if not marker_path.exists():
            stats["skipped"] += 1
            continue
        joints, joint_mask, fps = parse_ultra_trc(marker_path)
        if joints.shape[0] < args.min_len or joint_mask.sum().item() == 0:
            stats["skipped"] += 1
            continue
        # Synthesize IMU from Vicon-marker joint positions instead of the raw Delsys
        # IMU CSV (its acceleration contains gravity and its orientation is unknown),
        # keeping one consistent free-acc + position-built-orientation convention.
        imu, imu_mask = synthesize_imu(joints, joint_mask, fps)
        seq = build_sequence(
            "ultra_mocap",
            "all",
            imu_path.relative_to(root).as_posix(),
            fps,
            imu,
            imu_mask,
            joints,
            joint_mask,
            "synthetic_imu",
            "Vicon-marker joints; IMU synthesized from positions (free-acc + position-built orientation)",
            {"unit_info": SOURCE_UNIT_ASSUMPTIONS["ultra_mocap"]},
        )
        if seq:
            writer.add_sequence(seq, args.seq_len, args.stride, args.min_len)
            stats["sequences"] += 1
            stats["frames"] += joints.shape[0]
    return stats


def inspect_skipped_sources(data_root: Path) -> Dict:
    return {
        "keraal": {
            "path": str(data_root / "KeraalDataset"),
            "reason": "Vicon/Kinect/OpenPose/BlazePose data found, but no IMU files were found.",
            "raw_units": {
                "kinect": "3D positions in meters plus quaternions",
                "vicon": "3D positions appear millimeter-scale plus quaternions",
                "blazepose": "normalized image-space x/y/z, not metric",
                "openpose": "2D image-space x/y",
            },
        },
        "rehab_12112951": {
            "path": str(data_root / "rehab_data" / "12112951"),
            "reason": "8 Delsys IMUs at 148.148 Hz found, but no full-body joint-position GT.",
            "raw_units": {
                "emg": "millivolts",
                "accelerometer": "G",
                "gyroscope": "deg/s",
                "emg_fps": 1259.2592592592594,
                "imu_fps": 148.14814814814815,
            },
        },
        "rehab_cg_archives": {
            "path": str(data_root / "rehab_data"),
            "reason": "Kinect JointPosition/JointOrientation archives found; no synchronized IMU confirmed yet.",
            "raw_units": {
                "kinect_joint_position": "Kinect camera coordinates, likely meters",
                "kinect_joint_orientation": "quaternions",
            },
        },
        "dynaip_raw": {
            "path": "datasets/raw",
            "reason": "Raw downloads are still in progress or not extracted; run DynaIP extract/process before adding these shards.",
        },
    }


def summarize_shards(out_root: Path) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    total = 0
    source_counts: Dict[str, int] = {}
    quality_counts: Dict[str, int] = {}
    for path in sorted(out_root.glob("shard_*.pt")):
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            continue
        for sample in obj.get("samples", []):
            total += 1
            source = str(sample.get("source", "unknown"))
            quality = str(sample.get("quality", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
    return total, source_counts, quality_counts


def save_metadata(out_root: Path, writer: ShardWriter, args, stats: Dict):
    total_windows, source_counts, quality_counts = summarize_shards(out_root)
    meta = {
        "schema_version": "poser_mle_unified_v1",
        "canonical_imus": CANONICAL_IMUS,
        "canonical_joints": CANONICAL_JOINTS,
        "imu_feature": "shape [T, N, 12], columns are orientation_9d + acceleration_xyz when available; missing sensors are zero-filled",
        "targets": [
            "joint_orient_r6d",
            "joint_global_orient_r6d",
            "joint_rot_delta_r6d",
            "joint_position_root_relative",
            "joint_delta",
            "joint_delta_local",
            "joint_velocity",
            "joint_displacement",
            "joint_step_distance",
        ],
        "quality_meaning": {
            "primary": "Real IMU and full-body canonical joint positions; directly usable for main training.",
            "synthetic_imu": "IMU is synthesized from joint positions (free-acc + position-built orientation); GT positions are real (mocap) or video/IK-derived.",
            "weak_gt": "IMU exists but GT is video/IK-derived or partial.",
            "auxiliary": "Useful motion data but sensors/GT are not the same modality as the primary set.",
            "motion_only": "No raw IMU found; IMU is all zeros and imu_mask is false.",
        },
        "source_unit_assumptions": SOURCE_UNIT_ASSUMPTIONS,
        "window": {"seq_len": args.seq_len, "stride": args.stride, "min_len": args.min_len, "target_fps": args.target_fps},
        "new_windows_this_run": writer.total,
        "total_windows": total_windows,
        "source_window_counts": source_counts,
        "quality_window_counts": quality_counts,
        "source_stats": stats,
        "skipped_sources": inspect_skipped_sources(Path(args.data_root)),
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    torch.save(meta, out_root / "metadata.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data")
    parser.add_argument("--sources", default="mobileposer,dynaip,vidimu,ultra,ceti")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--mobileposer-fps", type=float, default=60.0)  # MobilePoser reprocessed at 60 Hz
    parser.add_argument("--target-fps", type=float, default=None)
    args = parser.parse_args()

    global TARGET_FPS
    TARGET_FPS = args.target_fps

    out_root = Path(args.output)
    writer = ShardWriter(out_root, args.shard_size)
    data_root = Path(args.data_root)
    stats = {}
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

    if "mobileposer" in sources:
        print("[convert] mobileposer")
        stats["mobileposer"] = convert_mobileposer(data_root, writer, args)
    if "dynaip" in sources:
        print("[convert] dynaip")
        stats["dynaip"] = convert_dynaip_extract(data_root, writer, args)
    if "vidimu" in sources:
        print("[convert] vidimu")
        stats["vidimu"] = convert_vidimu(data_root, writer, args)
    if "ultra" in sources:
        print("[convert] ultra")
        stats["ultra"] = convert_ultra(data_root, writer, args)
    if "ceti" in sources:
        print("[convert] ceti")
        stats["ceti"] = convert_ceti(data_root, writer, args)

    writer.close()
    save_metadata(out_root, writer, args, stats)
    print(f"[done] {writer.total} windows written to {out_root}")


if __name__ == "__main__":
    main()
