#!/usr/bin/env python3
"""Process new datasets into per-dataset .pt files for MoEPosing training.

Datasets:
  HPS       -- hps_mvnx.zip: Xsens MVN 30fps, 17 real IMU sensors + 23-segment mocap.
                Quality: "primary". 36 recordings, 6 subjects, in-the-wild activities.
  CZU-MHAD  -- CZU-MHAD.zip: Kinect V2 25-joint skeleton + 3 Shimmer3 IMU sensors.
                Quality: "synthetic_imu" (orientation synthesized from skeleton positions).
  mm-fit    -- mm-fit.zip: Vicon 17-joint 3D pose (30fps) + 4 wearable IMUs (acc+gyr only).
                Quality: "synthetic_imu" (orientation synthesized; real accel used where available).
  MRI       -- mri.zip: Triangulated COCO-17 3D keypoints + 6 real IMU sensors (quat+accel).
                Quality: "primary".

NOT processed:
  RecGym    -- CSV stored with DEFLATE64 compression; Python zipfile cannot read it.
  Fit3D     -- test set only (videos, no pose annotations).

Output:
  data_processed/hps.pt
  data_processed/czumhad.pt
  data_processed/mmfit.pt
  data_processed/mri.pt

  Each file: {"samples": [...], "dataset_name": str,
              "canonical_imus": CANONICAL_IMUS, "canonical_joints": CANONICAL_JOINTS}
  Each sample matches the shard format consumed by MaskedMotionDataset.

Usage:
  python process_new_datasets.py --data-root data --output data_processed --sources all
  python process_new_datasets.py --data-root data --output data_processed --sources hps,mri
"""

import argparse
import csv
import io
import pickle
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse utilities from the existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from schema import CANONICAL_IMUS, CANONICAL_JOINTS, XSENS_JOINTS, SMPL_JOINTS, canonical_name
from synth_imu import synthesize_imu
from build_unified_pose_data import (
    quat_to_mat,
    root_normalized_r6d,
    build_sequence, zeros_imu, zeros_joints,
    self_pose_change_targets,
    windows,
)

JOINT_INDEX = {n: i for i, n in enumerate(CANONICAL_JOINTS)}
IMU_INDEX = {n: i for i, n in enumerate(CANONICAL_IMUS)}

# ---------------------------------------------------------------------------
# DIP_IMU -- from data/raw_data/DIP_IMU/
# ---------------------------------------------------------------------------

# DIP-IMU paper sensor order (17 Xsens sensors, 2 toe sensors skipped).
DIP_17_IMU_NAMES = [
    "Head", "T8", "Pelvis",
    "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm",
    "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "LeftHand", "RightHand",
    None, None,  # LeftToe / RightToe — not in CANONICAL_IMUS
]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

SMPL_TPOSE_JOINTS = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],
    [ 0.0611, -0.0907, -0.0100],
    [-0.0611, -0.0907, -0.0100],
    [ 0.0028,  0.1208, -0.0267],
    [ 0.0759, -0.4879, -0.0318],
    [-0.0759, -0.4879, -0.0318],
    [ 0.0029,  0.2398, -0.0122],
    [ 0.0916, -0.8724, -0.0093],
    [-0.0916, -0.8724, -0.0093],
    [ 0.0023,  0.3628,  0.0036],
    [ 0.0803, -0.9319,  0.0970],
    [-0.0803, -0.9319,  0.0970],
    [ 0.0044,  0.5165,  0.0035],
    [ 0.0644,  0.5014, -0.0190],
    [-0.0644,  0.5014, -0.0190],
    [ 0.0031,  0.5972,  0.0329],
    [ 0.1901,  0.4832, -0.0255],
    [-0.1901,  0.4832, -0.0255],
    [ 0.4194,  0.4567, -0.0320],
    [-0.4194,  0.4567, -0.0320],
    [ 0.6456,  0.4500, -0.0073],
    [-0.6456,  0.4500, -0.0073],
    [ 0.6862,  0.4404,  0.0345],
    [-0.6862,  0.4404,  0.0345],
], dtype=torch.float32)


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(*aa.shape[:-1], 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    return I * cos_a + sin_a * K + (1 - cos_a) * (axis.unsqueeze(-1) @ axis.unsqueeze(-2))


def _smpl_local_to_global(local_rotmats: torch.Tensor) -> torch.Tensor:
    out = []
    for j, p in enumerate(SMPL_PARENTS[:local_rotmats.shape[1]]):
        out.append(local_rotmats[:, j] if p < 0 else out[p] @ local_rotmats[:, j])
    return torch.stack(out, dim=1)


def _smpl_fk_positions(global_rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """SMPL FK: global rotations [T,24,3,3] → canonical joint positions [T,28,3]."""
    t, n = global_rot.shape[:2]
    bone_vec = torch.zeros(n, 3)
    for j in range(n):
        p = SMPL_PARENTS[j]
        bone_vec[j] = SMPL_TPOSE_JOINTS[j] if p < 0 else SMPL_TPOSE_JOINTS[j] - SMPL_TPOSE_JOINTS[p]

    positions = torch.zeros(t, n, 3)
    for j in range(n):
        p = SMPL_PARENTS[j]
        if p < 0:
            positions[:, j] = 0.0
        else:
            positions[:, j] = positions[:, p] + (global_rot[:, p] @ bone_vec[j].unsqueeze(-1)).squeeze(-1)

    canon_pos = torch.zeros(t, len(CANONICAL_JOINTS), 3)
    canon_mask = torch.zeros(len(CANONICAL_JOINTS), dtype=torch.bool)
    for src_i, name in enumerate(SMPL_JOINTS[:n]):
        dst = JOINT_INDEX.get(name)
        if dst is not None:
            canon_pos[:, dst] = positions[:, src_i]
            canon_mask[dst] = True
    return canon_pos, canon_mask


def convert_dip(raw_data_root: Path, writer, args) -> Dict:
    """Process DIP_IMU raw pickle files: 17 real Xsens IMUs + SMPL GT pose."""
    dip_root = raw_data_root / "DIP_IMU"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    if not dip_root.exists():
        print(f"  [DIP] {dip_root} not found; skipping.")
        return stats

    for subject_dir in sorted(dip_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_file in sorted(subject_dir.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as e:
                print(f"  [skip] {pkl_file.name}: {e}")
                stats["skipped"] += 1
                continue

            imu_raw = torch.tensor(data["imu"], dtype=torch.float32)   # [T, 17, 12]
            gt_pose = torch.tensor(data["gt"],  dtype=torch.float32)   # [T, 72]
            t = min(imu_raw.shape[0], gt_pose.shape[0])
            imu_raw = imu_raw[:t]
            gt_pose = gt_pose[:t]

            # Map 17 DIP IMU sensors → 15 CANONICAL_IMUS.
            imu, imu_mask = zeros_imu(t)
            for src_i, name in enumerate(DIP_17_IMU_NAMES):
                if name is None:
                    continue
                dst = IMU_INDEX.get(name)
                if dst is None:
                    continue
                imu[:, dst] = imu_raw[:, src_i]
                imu_mask[dst] = True

            # SMPL axis-angle → global rotation matrices → FK positions.
            local_aa  = gt_pose.view(t, 24, 3)
            local_rot = _axis_angle_to_matrix(local_aa)
            global_rot = _smpl_local_to_global(local_rot)
            orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])
            joints, joint_mask = _smpl_fk_positions(global_rot)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "dip",
                "all",
                f"{subject_dir.name}/{pkl_file.stem}",
                60.0,
                imu,
                imu_mask,
                joints,
                joint_mask,
                "primary",
                "DIP-IMU: 17 Xsens real IMUs + SMPL GT pose at 60Hz",
                {"joint_orient_r6d": orient6, "orient_mask": orient_mask},
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += t

    return stats


# ---------------------------------------------------------------------------
# HPS (Xsens MVN) -- from hps_mvnx.zip
# ---------------------------------------------------------------------------

# HPS MVNX has exactly 17 sensors matching these labels (order from XML <sensors>):
# Pelvis, T8, Head, RightShoulder, RightUpperArm, RightForeArm, RightHand,
# LeftShoulder,  LeftUpperArm,  LeftForeArm,  LeftHand,
# RightUpperLeg, RightLowerLeg, RightFoot, LeftUpperLeg, LeftLowerLeg, LeftFoot
# Of these, RightShoulder and LeftShoulder have no slot in CANONICAL_IMUS → skipped.
HPS_SENSOR_TO_IMU = {
    "Pelvis":        "Pelvis",
    "T8":            "T8",
    "Head":          "Head",
    "RightUpperArm": "RightUpperArm",
    "RightForeArm":  "RightForeArm",
    "RightHand":     "RightHand",
    "LeftUpperArm":  "LeftUpperArm",
    "LeftForeArm":   "LeftForeArm",
    "LeftHand":      "LeftHand",
    "RightUpperLeg": "RightUpperLeg",
    "RightLowerLeg": "RightLowerLeg",
    "RightFoot":     "RightFoot",
    "LeftUpperLeg":  "LeftUpperLeg",
    "LeftLowerLeg":  "LeftLowerLeg",
    "LeftFoot":      "LeftFoot",
    # RightShoulder, LeftShoulder → not in CANONICAL_IMUS, skipped
}

MVNX_NS = "http://www.xsens.com/mvn/mvnx"


def _parse_floats(text: Optional[str]) -> List[float]:
    if text is None:
        return []
    return [float(x) for x in text.split()]


def convert_hps_mvnx(zip_path: Path, writer, args) -> Dict:
    """Parse all .mvnx files from hps_mvnx.zip."""
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    with zipfile.ZipFile(zip_path) as zf:
        mvnx_entries = [n for n in zf.namelist() if n.endswith(".mvnx")]
        for entry in sorted(mvnx_entries):
            with zf.open(entry) as f:
                xml_bytes = f.read()
            seq_name = Path(entry).stem
            result = _parse_one_mvnx(xml_bytes, seq_name, args)
            if result is None:
                stats["skipped"] += 1
                continue
            seq = result
            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += int(seq["joint_delta"].shape[0])
    return stats


def _parse_one_mvnx(xml_bytes: bytes, name: str, args) -> Optional[Dict]:
    """Parse one MVNX file → unified sequence dict."""
    try:
        root_el = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  [skip] {name}: XML parse error: {e}")
        return None

    ns = MVNX_NS
    subject_el = root_el.find(f"{{{ns}}}subject")
    if subject_el is None:
        print(f"  [skip] {name}: no <subject>")
        return None

    fps = float(subject_el.get("frameRate", "30"))
    n_segs = int(subject_el.get("segmentCount", "23"))

    # Parse segment order (matches XSENS_JOINTS order in schema.py by default).
    seg_labels: List[str] = []
    for seg in subject_el.findall(f".//{{{ns}}}segment"):
        seg_labels.append(seg.get("label", ""))
    if not seg_labels:
        seg_labels = XSENS_JOINTS[:n_segs]

    # Parse sensor order from <sensors> block.
    sensor_labels: List[str] = []
    for s in subject_el.findall(f".//{{{ns}}}sensor"):
        sensor_labels.append(s.get("label", ""))

    n_sensors = len(sensor_labels)

    frames_el = subject_el.find(f"{{{ns}}}frames")
    if frames_el is None:
        print(f"  [skip] {name}: no <frames>")
        return None

    # Preallocate: we don't know T ahead of time so accumulate lists then stack.
    all_orient_q: List[List[float]] = []   # each row: n_segs × 4 quaternion (w x y z)
    all_pos:       List[List[float]] = []  # each row: n_segs × 3 positions (m)
    all_acc:       List[List[float]] = []  # each row: n_segs × 3 free-acc (m/s^2) per segment

    for frame_el in frames_el:
        if frame_el.get("type") not in ("normal", None, ""):
            continue
        orient_text = frame_el.findtext(f"{{{ns}}}orientation")
        pos_text    = frame_el.findtext(f"{{{ns}}}position")
        acc_text    = frame_el.findtext(f"{{{ns}}}acceleration")
        if not orient_text or not pos_text:
            continue
        orient_vals = _parse_floats(orient_text)
        pos_vals    = _parse_floats(pos_text)
        if len(orient_vals) < n_segs * 4 or len(pos_vals) < n_segs * 3:
            continue
        all_orient_q.append(orient_vals[:n_segs * 4])
        all_pos.append(pos_vals[:n_segs * 3])
        acc_vals = _parse_floats(acc_text) if acc_text else []
        all_acc.append(acc_vals[:n_segs * 3] if len(acc_vals) >= n_segs * 3 else [])

    t = len(all_orient_q)
    if t < 10:
        print(f"  [skip] {name}: only {t} frames")
        return None

    orient_np = np.array(all_orient_q, dtype=np.float32).reshape(t, n_segs, 4)
    pos_np    = np.array(all_pos,     dtype=np.float32).reshape(t, n_segs, 3)
    has_acc   = any(len(r) >= n_segs * 3 for r in all_acc)
    if has_acc:
        acc_np = np.array([r if len(r) >= n_segs * 3 else [0.0] * (n_segs * 3)
                           for r in all_acc], dtype=np.float32).reshape(t, n_segs, 3)
    else:
        acc_np = None

    orient_q  = torch.from_numpy(orient_np)  # [T, n_segs, 4]  (w x y z)
    positions = torch.from_numpy(pos_np)     # [T, n_segs, 3]
    # Convert quaternion → 9D rotation matrix for each segment.
    orient_mat = quat_to_mat(orient_q.reshape(-1, 4)).reshape(t, n_segs, 3, 3)

    # --- Build per-canonical-joint positions ---
    joints, joint_mask = zeros_joints(t)
    for src_i, label in enumerate(seg_labels[:n_segs]):
        dst = JOINT_INDEX.get(canonical_name(label))
        if dst is None:
            dst = JOINT_INDEX.get(label)
        if dst is None:
            continue
        joints[:, dst] = positions[:, src_i]
        joint_mask[dst] = True

    # --- Build per-canonical-IMU data ---
    imu, imu_mask = zeros_imu(t)
    seg_by_label = {lab: i for i, lab in enumerate(seg_labels)}
    for sensor_name, canonical_imu in HPS_SENSOR_TO_IMU.items():
        src_i = seg_by_label.get(sensor_name)
        if src_i is None:
            continue
        dst_imu = IMU_INDEX.get(canonical_imu)
        if dst_imu is None:
            continue
        # Orientation: take 9D from segment rotation matrix (row-major flatten).
        imu[:, dst_imu, :9] = orient_mat[:, src_i].reshape(t, 9)
        # Free acceleration: use <acceleration> tag if available, else synthesize.
        if acc_np is not None:
            imu[:, dst_imu, 9:] = torch.from_numpy(acc_np[:, src_i])
        imu_mask[dst_imu] = True

    if acc_np is None:
        # Synthesize free acceleration from positions for IMU slots we have.
        imu_synth, _ = synthesize_imu(joints, joint_mask, fps)
        for slot in range(len(CANONICAL_IMUS)):
            if imu_mask[slot]:
                # Overwrite only acceleration (keep real orientation).
                imu[:, slot, 9:] = imu_synth[:, slot, 9:]

    if not joint_mask.any() or not imu_mask.any():
        print(f"  [skip] {name}: empty joint or IMU mask")
        return None

    # --- Orientation targets ---
    orient6, orient_mask = root_normalized_r6d(orient_mat, seg_labels)
    pose_change = self_pose_change_targets(joints, joint_mask, orient_mat, seg_labels)

    seq = build_sequence(
        "hps",
        "all",
        name,
        fps,
        imu,
        imu_mask,
        joints,
        joint_mask,
        "primary",
        "Xsens MVN full-body: 17 real IMU sensors + 23-segment global orientation/position at 30Hz",
        {
            "joint_orient_r6d": orient6,
            "orient_mask": orient_mask,
            **pose_change,
        },
    )
    return seq


# ---------------------------------------------------------------------------
# CZU-MHAD -- from CZU-MHAD.zip
# ---------------------------------------------------------------------------

# Kinect V2 25-joint order (MATLAB SDK output).
KINECT_JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]

KINECT_TO_CANONICAL = {
    "SpineBase":    "Pelvis",
    "SpineMid":     "T8",           # approximate (mid spine ≈ T8 for our skeleton)
    "SpineShoulder":"Neck",
    "Neck":         "Neck",
    "Head":         "Head",
    "ShoulderLeft": "LeftShoulder",
    "ElbowLeft":    "LeftForeArm",
    "WristLeft":    "LeftHand",
    "HandLeft":     "LeftHandIndex",
    "ShoulderRight":"RightShoulder",
    "ElbowRight":   "RightForeArm",
    "WristRight":   "RightHand",
    "HandRight":    "RightHandIndex",
    "HipLeft":      "LeftUpperLeg",
    "KneeLeft":     "LeftLowerLeg",
    "AnkleLeft":    "LeftFoot",
    "FootLeft":     "LeftToe",
    "HipRight":     "RightUpperLeg",
    "KneeRight":    "RightLowerLeg",
    "AnkleRight":   "RightFoot",
    "FootRight":    "RightToe",
}

# CZU-MHAD action IDs 1-22.
CZU_ACTIONS = {
    "1": "swipe_left", "2": "swipe_right", "3": "wave", "4": "clap",
    "5": "throw", "6": "arm_cross", "7": "basketball_shoot", "8": "draw_x",
    "9": "draw_circle_cw", "10": "draw_circle_ccw", "11": "draw_triangle",
    "12": "bowling", "13": "boxing", "14": "baseball_swing", "15": "tennis_swing",
    "16": "arm_curl", "17": "tennis_serve", "18": "push", "19": "knock",
    "20": "catch", "21": "pickup_throw", "22": "jog",
}


def convert_czumhad(zip_path: Path, writer, args) -> Dict:
    """Parse CZU-MHAD .mat files."""
    try:
        import scipy.io as sio
    except ImportError:
        print("[CZU-MHAD] scipy not available; skipping.")
        return {"sequences": 0, "skipped": 0, "frames": 0}

    import re
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    fps = args.czumhad_fps

    with zipfile.ZipFile(zip_path) as zf:
        skel_entries = sorted(
            n for n in zf.namelist()
            if "skeleton_mat" in n and n.endswith(".mat") and "__MACOSX" not in n
        )
        sensor_entries = {
            Path(n).name: n
            for n in zf.namelist()
            if "sensor_mat" in n and n.endswith(".mat") and "__MACOSX" not in n
        }

        for skel_entry in skel_entries:
            fname = Path(skel_entry).name  # e.g. cx_a4_t7.mat
            # Parse subject/action/trial from filename
            m = re.match(r"(\w+)_a(\d+)_t(\d+)\.mat$", fname)
            subj   = m.group(1) if m else "unk"
            action = CZU_ACTIONS.get(m.group(2) if m else "0", "unknown")
            trial  = m.group(3) if m else "0"

            with zf.open(skel_entry) as f:
                mat = sio.loadmat(io.BytesIO(f.read()), squeeze_me=True)
            sk = mat.get("skeleton")
            if sk is None or sk.ndim != 2 or sk.shape[1] < 75:
                stats["skipped"] += 1
                continue

            t_skel = sk.shape[0]
            # Layout: (T, 25*4) where each joint is (x, y, z, timestamp_100ns).
            sk3d = sk.reshape(t_skel, 25, 4)[:, :, :3].astype(np.float32)
            # Kinect uses right-handed coordinate, Y-up world frame (metres).
            positions_kin = torch.from_numpy(sk3d)  # [T, 25, 3]

            joints, joint_mask = zeros_joints(t_skel)
            for src_i, kin_name in enumerate(KINECT_JOINTS):
                canon = KINECT_TO_CANONICAL.get(kin_name)
                if canon is None:
                    continue
                dst = JOINT_INDEX.get(canon)
                if dst is None:
                    continue
                joints[:, dst] = positions_kin[:, src_i]
                joint_mask[dst] = True

            if joint_mask.sum().item() < 5:
                stats["skipped"] += 1
                continue

            # Synthesize IMU from Kinect joint positions.
            imu, imu_mask = synthesize_imu(joints, joint_mask, fps)
            if not imu_mask.any():
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "czumhad",
                "all",
                f"{subj}_{action}_t{trial}",
                fps,
                imu,
                imu_mask,
                joints,
                joint_mask,
                "synthetic_imu",
                "Kinect V2 25-joint skeleton; IMU synthesized from positions",
                {"activity": action, "subject": subj},
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += t_skel

    return stats


# ---------------------------------------------------------------------------
# mm-fit -- from mm-fit.zip
# ---------------------------------------------------------------------------

# pose_3d dim layout: (3=xyz, T, 18) where [:, :, 0] = frame counter.
# Slots 1-17 correspond to OpenPose 18-joint skeleton (COCO-style), minus Nose:
MMFIT_JOINT_ORDER = [
    # slot index in pose_3d dim-2 : canonical joint name  (slot 0 = frame counter)
    None,               # 0: frame counter
    "Neck",             # 1
    "RightShoulder",    # 2
    "RightForeArm",     # 3  (RElbow)
    "RightHand",        # 4  (RWrist)
    "LeftShoulder",     # 5
    "LeftForeArm",      # 6  (LElbow)
    "LeftHand",         # 7  (LWrist)
    "RightUpperLeg",    # 8  (RHip)
    "RightLowerLeg",    # 9  (RKnee)
    "RightFoot",        # 10 (RAnkle)
    "LeftUpperLeg",     # 11 (LHip)
    "LeftLowerLeg",     # 12 (LKnee)
    "LeftFoot",         # 13 (LAnkle)
    None,               # 14 REye  → no canonical slot
    None,               # 15 LEye  → no canonical slot
    None,               # 16 REar  → no canonical slot
    None,               # 17 LEar  → no canonical slot
]

# IMU sensor keys in mm-fit and their canonical IMU slot.
MMFIT_SENSORS = {
    "sw_l": "LeftHand",     # smartwatch left wrist
    "sw_r": "RightHand",    # smartwatch right wrist
    "sp_r": "Pelvis",       # smartphone right hip pocket ≈ Pelvis
    "eb_l": "Head",         # earbuds left ≈ Head (no better canonical slot)
}


def convert_mmfit(zip_path: Path, writer, args) -> Dict:
    """Parse mm-fit workout sessions."""
    stats = {"sequences": 0, "skipped": 0, "frames": 0}
    fps_pose = args.mmfit_fps

    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.namelist()
        workouts = sorted(set(
            n.split("/")[1]
            for n in entries
            if n.startswith("mm-fit/w") and "/" in n[len("mm-fit/"):]
        ))

        for w in workouts:
            prefix = f"mm-fit/{w}/{w}"
            pose3d_key = f"{prefix}_pose_3d.npy"
            if pose3d_key not in entries:
                stats["skipped"] += 1
                continue

            with zf.open(pose3d_key) as f:
                pose3d = np.load(io.BytesIO(f.read()), allow_pickle=True)
            # pose3d: (3, T, 18)  where dim0=xyz, dim1=frames, dim2=joints
            if pose3d.ndim != 3 or pose3d.shape[0] != 3 or pose3d.shape[2] != 18:
                stats["skipped"] += 1
                continue
            T = pose3d.shape[1]
            frame_idx = pose3d[0, :, 0].astype(int)  # frame counter stored at joint slot 0

            # Build joint positions (mm → metres).
            joints, joint_mask = zeros_joints(T)
            for slot_i, canon_name in enumerate(MMFIT_JOINT_ORDER):
                if canon_name is None:
                    continue
                dst = JOINT_INDEX.get(canon_name)
                if dst is None:
                    continue
                xyz_mm = pose3d[:, :, slot_i].T.astype(np.float32)  # [T, 3]
                joints[:, dst] = torch.from_numpy(xyz_mm / 1000.0)
                joint_mask[dst] = True

            if joint_mask.sum().item() < 5:
                stats["skipped"] += 1
                continue

            # Synthesize orientation from joint positions (no real orientation in mm-fit).
            imu, imu_mask = synthesize_imu(joints, joint_mask, fps_pose)

            # Replace acc with real smartwatch/smartphone accelerometer data where available.
            for sensor_key, canonical_imu in MMFIT_SENSORS.items():
                acc_key = f"{prefix}_{sensor_key}_acc.npy"
                if acc_key not in entries:
                    continue
                with zf.open(acc_key) as f:
                    acc_arr = np.load(io.BytesIO(f.read()), allow_pickle=True)
                # acc_arr: (T_imu, 5) = (frame_idx, timestamp_ms, x, y, z) in m/s^2
                if acc_arr.ndim != 2 or acc_arr.shape[1] < 5:
                    continue
                imu_frame_idx = acc_arr[:, 0].astype(int)
                acc_xyz = acc_arr[:, 2:5].astype(np.float32)  # (T_imu, 3)

                # Align IMU frames to pose frames via frame counter.
                dst_imu = IMU_INDEX.get(canonical_imu)
                if dst_imu is None:
                    continue
                # For each pose frame, find closest IMU frame by frame counter.
                aligned_acc = _align_by_frame_idx(frame_idx, imu_frame_idx, acc_xyz)
                imu[:, dst_imu, 9:] = torch.from_numpy(aligned_acc)
                imu_mask[dst_imu] = True

            if not imu_mask.any():
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "mmfit",
                "all",
                w,
                fps_pose,
                imu,
                imu_mask,
                joints,
                joint_mask,
                "synthetic_imu",
                "Vicon 3D pose (mm→m); orientation synthesized; real acc from smartwatch/smartphone",
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += T

    return stats


def _align_by_frame_idx(
    target_idx: np.ndarray,
    source_idx: np.ndarray,
    source_data: np.ndarray,
) -> np.ndarray:
    """For each frame in target_idx, return source_data at the nearest source_idx."""
    out = np.zeros((len(target_idx), source_data.shape[1]), dtype=np.float32)
    # searchsorted for efficient O(T log T_imu) lookup.
    pos = np.searchsorted(source_idx, target_idx, side="left").clip(0, len(source_idx) - 1)
    out[:] = source_data[pos]
    return out


# ---------------------------------------------------------------------------
# MRI -- from mri.zip (nested dataset_release.zip)
# ---------------------------------------------------------------------------

# COCO 17-joint order (confirmed by MRI paper and data inspection).
MRI_COCO_JOINTS = [
    "Head",          # 0  Nose → map to Head (closest canonical)
    "Head",          # 1  LeftEye → merge into Head
    "Head",          # 2  RightEye → merge into Head
    None,            # 3  LeftEar → skip (redundant with Head)
    None,            # 4  RightEar → skip
    "LeftShoulder",  # 5
    "RightShoulder", # 6
    "LeftForeArm",   # 7  LeftElbow
    "RightForeArm",  # 8  RightElbow
    "LeftHand",      # 9  LeftWrist
    "RightHand",     # 10 RightWrist
    "LeftUpperLeg",  # 11 LeftHip
    "RightUpperLeg", # 12 RightHip
    "LeftLowerLeg",  # 13 LeftKnee
    "RightLowerLeg", # 14 RightKnee
    "LeftFoot",      # 15 LeftAnkle
    "RightFoot",     # 16 RightAnkle
]

# IMU placement in mRI dataset: wrists(2), ankles(2), chest(1), head(1).
# Assumed order IMU0-5 based on typical sensor attachment (chest first for calibration).
MRI_IMU_PLACEMENT = [
    "T8",            # IMU0: chest / sternum
    "RightHand",     # IMU1: right wrist
    "LeftHand",      # IMU2: left wrist
    "RightFoot",     # IMU3: right ankle
    "LeftFoot",      # IMU4: left ankle
    "Head",          # IMU5: head
]


def convert_mri(zip_path: Path, writer, args) -> Dict:
    """Parse mRI dataset (nested zip)."""
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    with zipfile.ZipFile(zip_path) as outer:
        inner_data = outer.read("dataset_release.zip")

    with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
        cpl_entries = [
            n for n in inner.namelist()
            if "aligned_data/pose_labels/" in n and n.endswith(".cpl")
        ]

        for cpl_entry in sorted(cpl_entries):
            # subject name from filename e.g. subject10_all_labels.cpl
            subj = Path(cpl_entry).stem.replace("_all_labels", "")
            with inner.open(cpl_entry) as f:
                labels = pickle.loads(f.read())

            refined_kps = labels.get("refined_gt_kps")  # (T, 3, 17)
            if refined_kps is None:
                stats["skipped"] += 1
                continue

            T = refined_kps.shape[0]
            gt_frames = labels.get("gt_avail_frames", [0, T - 1])
            imu_frames = labels.get("imu_avail_frames", [0, T - 1])

            # Overlap of valid GT and IMU frames.
            valid_start = max(gt_frames[0], imu_frames[0]) if len(gt_frames) >= 2 else 0
            valid_end   = min(gt_frames[1], imu_frames[1]) if len(imu_frames) >= 2 else T - 1
            if valid_end - valid_start < 10:
                stats["skipped"] += 1
                continue

            # Build joint positions from COCO-17 keypoints (assumed metres).
            kps = torch.from_numpy(
                refined_kps[valid_start:valid_end + 1].astype(np.float32)
            )  # [T', 3, 17]
            t_valid = kps.shape[0]
            joints, joint_mask = zeros_joints(t_valid)

            head_vals = []
            for coco_i, canon_name in enumerate(MRI_COCO_JOINTS):
                if canon_name is None:
                    continue
                if canon_name == "Head":
                    # Accumulate head/eye/nose for averaging.
                    head_vals.append(kps[:, :, coco_i])
                    continue
                dst = JOINT_INDEX.get(canon_name)
                if dst is None:
                    continue
                joints[:, dst] = kps[:, :, coco_i]
                joint_mask[dst] = True

            if head_vals:
                head_pos = torch.stack(head_vals, dim=0).mean(dim=0)  # [T', 3]
                dst = JOINT_INDEX.get("Head")
                if dst is not None:
                    joints[:, dst] = head_pos
                    joint_mask[dst] = True

            if joint_mask.sum().item() < 4:
                stats["skipped"] += 1
                continue

            # Load IMU CSVs for this subject.
            imu, imu_mask = zeros_imu(t_valid)
            imu_dir_prefix = f"dataset_release/aligned_data/imu/{subj}/"

            for imu_i, canonical_imu in enumerate(MRI_IMU_PLACEMENT):
                imu_csv_key = f"{imu_dir_prefix}IMU{imu_i}.csv"
                if imu_csv_key not in inner.namelist():
                    continue
                with inner.open(imu_csv_key) as f:
                    rows = list(csv.DictReader(io.TextIOWrapper(f)))
                if len(rows) < 5:
                    continue

                # Columns: address, axg, ayg, azg, AngleXdeg, AngleYdeg, AngleZdeg,
                #          hx, hy, hz, q0, q1, q2, q3
                # q0,q1,q2,q3 = (w, x, y, z) quaternion.
                # axg, ayg, azg = acceleration in g → × 9.81 → m/s^2.
                n_rows = len(rows)

                # Align IMU row count to valid GT frames by linear interpolation.
                # Both are temporally aligned in the aligned_data folder.
                imu_t = min(n_rows - valid_start,
                            min(n_rows - valid_start, t_valid))
                if imu_t < 5:
                    continue

                # Extract quaternions and acceleration.
                qs  = np.zeros((n_rows, 4), dtype=np.float32)
                acc = np.zeros((n_rows, 3), dtype=np.float32)
                for ri, row in enumerate(rows):
                    try:
                        qs[ri]  = [float(row["q0"]),  float(row["q1"]),
                                   float(row["q2"]),  float(row["q3"])]
                        acc[ri] = [float(row["axg"]) * 9.80665,
                                   float(row["ayg"]) * 9.80665,
                                   float(row["azg"]) * 9.80665]
                    except (ValueError, KeyError):
                        continue

                # Slice from valid_start if IMU and GT are frame-aligned.
                qs_valid  = qs[valid_start: valid_start + t_valid]
                acc_valid = acc[valid_start: valid_start + t_valid]
                actual_t  = min(len(qs_valid), t_valid)

                if actual_t < 5:
                    continue

                rot9 = quat_to_mat(
                    torch.from_numpy(qs_valid[:actual_t])
                ).reshape(actual_t, 9)

                dst_imu = IMU_INDEX.get(canonical_imu)
                if dst_imu is None:
                    continue
                imu[:actual_t, dst_imu, :9] = rot9
                imu[:actual_t, dst_imu, 9:] = torch.from_numpy(acc_valid[:actual_t])
                imu_mask[dst_imu] = True

            if not imu_mask.any():
                # Fall back: synthesize from GT positions.
                imu, imu_mask = synthesize_imu(joints, joint_mask, args.mri_fps)

            if not imu_mask.any():
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "mri",
                "all",
                f"{subj}",
                args.mri_fps,
                imu,
                imu_mask,
                joints,
                joint_mask,
                "primary" if imu_mask.any() else "synthetic_imu",
                "mRI NeurIPS22: triangulated COCO-17 GT (m) + 6 real IMU sensors (quat+acc_g)",
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += t_valid

    return stats


# ---------------------------------------------------------------------------
# TotalCapture -- data/raw_data/TotalCapture/  (TEST SET)
# ---------------------------------------------------------------------------

# TotalCapture raw skeleton joint names → canonical (from preprocess_raw_to_shards.py)
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

# TotalCapture 6 IMU sensors (MobilePoser order, same as DIP paper convention)
TC_IMU_NAMES = ["LeftForeArm", "RightForeArm", "LeftUpperLeg", "RightUpperLeg", "Head", "Pelvis"]


def _parse_tc_txt(path: Path, values_per_joint: int) -> Tuple[Optional[torch.Tensor], List[str]]:
    """Parse TotalCapture global skeleton txt file.

    Format: tab-separated columns, each column is a joint whose value is
    space-separated floats (e.g. "x y z" for positions, "w x y z" for quats).
    """
    try:
        with open(path) as f:
            header = [h for h in f.readline().strip().split("\t") if h]
            rows = []
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) < len(header):
                    continue
                row = []
                ok = True
                for col in cols[:len(header)]:
                    parts = col.strip().split()
                    if len(parts) < values_per_joint:
                        ok = False
                        break
                    row.extend(float(p) for p in parts[:values_per_joint])
                if ok:
                    rows.append(row)
        if not rows:
            return None, header
        data = torch.tensor(rows, dtype=torch.float32)
        return data.view(len(rows), len(header), values_per_joint), header
    except Exception:
        return None, []


def convert_totalcapture(raw_data_root: Path, writer, args) -> Dict:
    """Process TotalCapture: real 6-IMU pkl + global skeleton txt files."""
    tc_root = raw_data_root / "TotalCapture"
    raw_root = tc_root / "raw"
    imu_root = tc_root / "IMU"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    if not tc_root.exists():
        print(f"  [TotalCapture] {tc_root} not found; skipping.")
        return stats

    for subject_dir in sorted(raw_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name.lower()  # S1 → s1

        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name  # acting1, walking1, etc.

            pos_data, pos_names = _parse_tc_txt(seq_dir / "gt_skel_gbl_pos.txt", 3)
            ori_data, ori_names = _parse_tc_txt(seq_dir / "gt_skel_gbl_ori.txt", 4)

            if pos_data is None or ori_data is None:
                stats["skipped"] += 1
                continue

            t = min(pos_data.shape[0], ori_data.shape[0])
            pos_data = pos_data[:t] / 100.0  # cm → metres
            ori_data = ori_data[:t]          # [T, J, 4] wxyz

            rot_mats = quat_to_mat(ori_data).reshape(t, len(ori_names), 3, 3)

            joints, joint_mask = zeros_joints(t)
            for src_i, raw_name in enumerate(pos_names):
                cn = TC_RAW_JOINT_MAP.get(raw_name)
                if cn and cn in JOINT_INDEX:
                    joints[:, JOINT_INDEX[cn]] = pos_data[:, src_i]
                    joint_mask[JOINT_INDEX[cn]] = True

            canon_ori_names = [TC_RAW_JOINT_MAP.get(n, n) for n in ori_names]
            orient6, orient_mask = root_normalized_r6d(rot_mats, canon_ori_names)

            # Real IMU data from pkl
            imu_pkl = imu_root / f"{subject}_{seq_name}.pkl"
            imu, imu_mask = zeros_imu(t)
            if imu_pkl.exists():
                with open(imu_pkl, "rb") as f:
                    imu_data = pickle.load(f, encoding="latin1")
                acc = torch.tensor(imu_data["acc"], dtype=torch.float32)  # [T, 6, 3]
                ori = torch.tensor(imu_data["ori"], dtype=torch.float32)  # [T, 6, 3, 3]
                t_imu = min(t, acc.shape[0], ori.shape[0])
                for src_i, name in enumerate(TC_IMU_NAMES[:acc.shape[1]]):
                    dst = IMU_INDEX.get(name)
                    if dst is None:
                        continue
                    imu[:t_imu, dst, :9] = ori[:t_imu, src_i].reshape(t_imu, 9)
                    imu[:t_imu, dst, 9:] = acc[:t_imu, src_i]
                    imu_mask[dst] = True
            else:
                imu, imu_mask = synthesize_imu(joints, joint_mask, 60.0)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "totalcapture", "test",
                f"{subject_dir.name}_{seq_name}",
                60.0, imu, imu_mask, joints, joint_mask,
                "primary",
                "TotalCapture: real 6-IMU + global skeleton at 60Hz (test set)",
                {"joint_orient_r6d": orient6, "orient_mask": orient_mask},
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += t

    return stats


# ---------------------------------------------------------------------------
# IMUPoser -- data/raw_data/IMUPoser/  (TEST SET)
# ---------------------------------------------------------------------------

def convert_imuposer(raw_data_root: Path, writer, args) -> Dict:
    """Process IMUPoser: SMPL pose + synthesized IMU (test set)."""
    ip_root = raw_data_root / "IMUPoser"
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    if not ip_root.exists():
        print(f"  [IMUPoser] {ip_root} not found; skipping.")
        return stats

    for subject_dir in sorted(ip_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_file in sorted(subject_dir.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as e:
                stats["skipped"] += 1
                continue

            pose = data["pose"]
            if not isinstance(pose, torch.Tensor):
                pose = torch.from_numpy(pose)
            pose = pose.float()
            t = pose.shape[0]

            local_aa  = pose.view(t, 24, 3)
            local_rot = _axis_angle_to_matrix(local_aa)
            global_rot = _smpl_local_to_global(local_rot)
            orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])
            joints, joint_mask = _smpl_fk_positions(global_rot)

            imu, imu_mask = synthesize_imu(joints, joint_mask, 60.0)

            if t < args.min_len:
                stats["skipped"] += 1
                continue

            seq = build_sequence(
                "imuposer", "test",
                f"{subject_dir.name}/{pkl_file.stem}",
                60.0, imu, imu_mask, joints, joint_mask,
                "synthetic_imu",
                "IMUPoser: SMPL GT pose + synthesized IMU at 60Hz (test set)",
                {"joint_orient_r6d": orient6, "orient_mask": orient_mask},
            )
            if seq is None:
                stats["skipped"] += 1
                continue

            writer["samples"].append(seq)
            stats["sequences"] += 1
            stats["frames"] += t

    return stats


# ---------------------------------------------------------------------------
# AMASS -- data/raw_data/AMASS/  (TRAIN SET, synthesized IMU)
# ---------------------------------------------------------------------------

def convert_amass_subset(subset_dir: Path, writer, args) -> Dict:
    """Process one AMASS subset: SMPL-H pose → FK positions → synthesized IMU."""
    stats = {"sequences": 0, "skipped": 0, "frames": 0}

    for npz_path in sorted(subset_dir.rglob("*.npz")):
        try:
            data = dict(np.load(str(npz_path), allow_pickle=True))
        except Exception:
            stats["skipped"] += 1
            continue

        poses = data.get("poses")
        if poses is None:
            stats["skipped"] += 1
            continue

        poses = torch.from_numpy(poses.astype(np.float32))
        fps = float(data.get("mocap_framerate", 60.0))
        t = poses.shape[0]

        if t < args.min_len:
            stats["skipped"] += 1
            continue

        # SMPL-H has 52 joints (156 params); use first 24 SMPL body joints (72 params).
        local_aa  = poses[:, :72].view(t, 24, 3)
        local_rot = _axis_angle_to_matrix(local_aa)
        global_rot = _smpl_local_to_global(local_rot)
        orient6, orient_mask = root_normalized_r6d(global_rot, SMPL_JOINTS[:24])
        joints, joint_mask = _smpl_fk_positions(global_rot)

        # Resample to 60 Hz
        if abs(fps - 60.0) > 1.0:
            out_t = max(2, int(round(t * 60.0 / fps)))
            if out_t < args.min_len:
                stats["skipped"] += 1
                continue
            def _resample(x, t_in, t_out):
                shape = x.shape
                flat = x.float().reshape(t_in, -1).T.unsqueeze(0)
                return F.interpolate(flat, size=t_out, mode="linear",
                                     align_corners=False).squeeze(0).T.reshape(t_out, *shape[1:])
            joints  = _resample(joints,   t, out_t)
            orient6 = _resample(orient6,  t, out_t)
            t = out_t

        imu, imu_mask = synthesize_imu(joints, joint_mask, 60.0)

        if not imu_mask.any():
            stats["skipped"] += 1
            continue

        rel_path = npz_path.relative_to(subset_dir.parent)
        seq = build_sequence(
            f"amass_{subset_dir.name}", "train",
            str(rel_path),
            60.0, imu, imu_mask, joints, joint_mask,
            "synthetic_imu",
            f"AMASS/{subset_dir.name}: SMPL-H FK positions + synthesized IMU at 60Hz",
            {"joint_orient_r6d": orient6, "orient_mask": orient_mask},
        )
        if seq is None:
            stats["skipped"] += 1
            continue

        writer["samples"].append(seq)
        stats["sequences"] += 1
        stats["frames"] += t

    return stats


# ---------------------------------------------------------------------------
# Writer / output
# ---------------------------------------------------------------------------

def save_dataset(samples: List[Dict], dataset_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "samples": samples,
        "dataset_name": dataset_name,
        "canonical_imus": CANONICAL_IMUS,
        "canonical_joints": CANONICAL_JOINTS,
    }
    torch.save(obj, out_path)
    total_frames = sum(int(s["joint_delta"].shape[0]) for s in samples)
    print(f"[save] {out_path}  ({len(samples)} sequences, {total_frames} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data",
                    help="Directory containing the raw dataset zip files")
    ap.add_argument("--output", default="data_processed",
                    help="Output directory for per-dataset .pt files")
    ap.add_argument("--sources", default="all",
                    help="Comma-separated list of sources to process: hps,czumhad,mmfit,mri or 'all'")
    ap.add_argument("--seq-len", type=int, default=300,
                    help="Window length in frames (0 = keep full sequences unsplit)")
    ap.add_argument("--stride",  type=int, default=150,
                    help="Stride for sliding window (only used when seq-len > 0)")
    ap.add_argument("--min-len", type=int, default=60,
                    help="Minimum window length; shorter windows are discarded")
    # Per-dataset fps overrides (use when auto-detection is unavailable)
    ap.add_argument("--czumhad-fps", type=float, default=30.0)
    ap.add_argument("--mmfit-fps",   type=float, default=30.0)
    ap.add_argument("--mri-fps",     type=float, default=30.0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.output)
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    if "all" in sources:
        sources = {"dip", "hps", "czumhad", "mmfit", "mri",
                   "totalcapture", "imuposer", "amass"}

    # -----------------------------------------------------------------------
    # DIP_IMU
    # -----------------------------------------------------------------------
    if "dip" in sources:
        raw_data_root = data_root / "raw_data"
        if not raw_data_root.exists():
            print(f"[DIP] {raw_data_root} not found; skipping.")
        else:
            print("[DIP] Processing DIP_IMU raw data ...")
            writer: Dict = {"samples": []}
            stats = convert_dip(raw_data_root, writer, args)
            print(f"[DIP] {stats}")
            if args.seq_len > 0:
                writer["samples"] = _window_samples(writer["samples"], args)
            if writer["samples"]:
                save_dataset(writer["samples"], "dip", out_root / "dip.pt")

    # -----------------------------------------------------------------------
    # HPS
    # -----------------------------------------------------------------------
    if "hps" in sources:
        hps_zip = data_root / "hps_mvnx.zip"
        if not hps_zip.exists():
            print(f"[HPS] {hps_zip} not found; skipping.")
        else:
            print("[HPS] Processing hps_mvnx.zip ...")
            writer: Dict = {"samples": []}
            stats = convert_hps_mvnx(hps_zip, writer, args)
            print(f"[HPS] {stats}")
            if args.seq_len > 0:
                writer["samples"] = _window_samples(writer["samples"], args)
            if writer["samples"]:
                save_dataset(writer["samples"], "hps", out_root / "hps.pt")

    # -----------------------------------------------------------------------
    # CZU-MHAD
    # -----------------------------------------------------------------------
    if "czumhad" in sources:
        czu_zip = data_root / "CZU-MHAD.zip"
        if not czu_zip.exists():
            print(f"[CZU-MHAD] {czu_zip} not found; skipping.")
        else:
            print("[CZU-MHAD] Processing CZU-MHAD.zip ...")
            writer = {"samples": []}
            stats = convert_czumhad(czu_zip, writer, args)
            print(f"[CZU-MHAD] {stats}")
            if args.seq_len > 0:
                writer["samples"] = _window_samples(writer["samples"], args)
            if writer["samples"]:
                save_dataset(writer["samples"], "czumhad", out_root / "czumhad.pt")

    # -----------------------------------------------------------------------
    # mm-fit
    # -----------------------------------------------------------------------
    if "mmfit" in sources:
        mmfit_zip = data_root / "mm-fit.zip"
        if not mmfit_zip.exists():
            print(f"[mm-fit] {mmfit_zip} not found; skipping.")
        else:
            print("[mm-fit] Processing mm-fit.zip ...")
            writer = {"samples": []}
            stats = convert_mmfit(mmfit_zip, writer, args)
            print(f"[mm-fit] {stats}")
            if args.seq_len > 0:
                writer["samples"] = _window_samples(writer["samples"], args)
            if writer["samples"]:
                save_dataset(writer["samples"], "mmfit", out_root / "mmfit.pt")

    # -----------------------------------------------------------------------
    # MRI
    # -----------------------------------------------------------------------
    if "mri" in sources:
        mri_zip = data_root / "mri.zip"
        if not mri_zip.exists():
            print(f"[MRI] {mri_zip} not found; skipping.")
        else:
            print("[MRI] Processing mri.zip ...")
            writer = {"samples": []}
            stats = convert_mri(mri_zip, writer, args)
            print(f"[MRI] {stats}")
            if args.seq_len > 0:
                writer["samples"] = _window_samples(writer["samples"], args)
            if writer["samples"]:
                save_dataset(writer["samples"], "mri", out_root / "mri.pt")

    # -----------------------------------------------------------------------
    # TotalCapture (test set)
    # -----------------------------------------------------------------------
    if "totalcapture" in sources:
        raw_data_root = data_root / "raw_data"
        print("[TotalCapture] Processing raw_data/TotalCapture ...")
        writer = {"samples": []}
        stats = convert_totalcapture(raw_data_root, writer, args)
        print(f"[TotalCapture] {stats}")
        if args.seq_len > 0:
            writer["samples"] = _window_samples(writer["samples"], args)
        if writer["samples"]:
            save_dataset(writer["samples"], "totalcapture", out_root / "totalcapture.pt")

    # -----------------------------------------------------------------------
    # IMUPoser (test set)
    # -----------------------------------------------------------------------
    if "imuposer" in sources:
        raw_data_root = data_root / "raw_data"
        print("[IMUPoser] Processing raw_data/IMUPoser ...")
        writer = {"samples": []}
        stats = convert_imuposer(raw_data_root, writer, args)
        print(f"[IMUPoser] {stats}")
        if args.seq_len > 0:
            writer["samples"] = _window_samples(writer["samples"], args)
        if writer["samples"]:
            save_dataset(writer["samples"], "imuposer", out_root / "imuposer.pt")

    # -----------------------------------------------------------------------
    # AMASS (train set, synthesized IMU, one file per subset)
    # -----------------------------------------------------------------------
    if "amass" in sources:
        amass_root = data_root / "raw_data" / "AMASS"
        if not amass_root.exists():
            print(f"[AMASS] {amass_root} not found; skipping.")
        else:
            subset_dirs = sorted(d for d in amass_root.iterdir() if d.is_dir())
            print(f"[AMASS] Found {len(subset_dirs)} subsets in {amass_root}")
            total_amass = {"sequences": 0, "skipped": 0, "frames": 0}
            for subset_dir in subset_dirs:
                print(f"[AMASS/{subset_dir.name}] processing ...")
                writer = {"samples": []}
                stats = convert_amass_subset(subset_dir, writer, args)
                for k in total_amass:
                    total_amass[k] += stats[k]
                print(f"[AMASS/{subset_dir.name}] {stats}")
                if args.seq_len > 0:
                    writer["samples"] = _window_samples(writer["samples"], args)
                if writer["samples"]:
                    out_name = f"amass_{subset_dir.name.lower()}.pt"
                    save_dataset(writer["samples"],
                                 f"amass_{subset_dir.name}",
                                 out_root / out_name)
            print(f"[AMASS] total: {total_amass}")

    print("[done]")


def _window_samples(samples: List[Dict], args) -> List[Dict]:
    """Apply sliding-window segmentation to raw sequences."""
    out = []
    for seq in samples:
        t = int(seq["joint_delta"].shape[0])
        for start, end in windows(t, args.seq_len, args.stride):
            if end - start < args.min_len:
                continue
            win = {}
            for key, val in seq.items():
                if isinstance(val, torch.Tensor) and val.shape[:1] == (t,):
                    win[key] = val[start:end].contiguous()
                else:
                    win[key] = val
            win["start"] = start
            win["end"]   = end
            out.append(win)
    return out


if __name__ == "__main__":
    main()
