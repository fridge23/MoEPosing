"""Synthesize IMU signals (orientation 9d + free acceleration) from joint global
positions, AMASS/DIP-style, for motion-only sources (ceti, vidimu, ultra).

Convention (must match the real-IMU sources dynaip/mobileposer in the unified set):
  - acceleration = world-frame FREE acceleration (gravity removed), units m/s^2,
    computed as the second time-difference of the segment's global position:
        a[t] = (p[t-1] - 2 p[t] + p[t+1]) * fps^2
  - orientation = a per-segment orthonormal frame built from joint positions
    (long axis = bone direction to a neighbor joint; the remaining axes from a
    reference joint), flattened to 9d. This is the standard way to recover segment
    orientation from mocap and keeps every synthetic source in the SAME world frame
    as the joint positions (no unregistered IMU frames).

Only an IMU whose segment joint AND a neighbor joint are present (so orientation is
defined) is synthesized; its imu_mask is then set True.
"""
from typing import Dict, List, Tuple

import torch

from schema import CANONICAL_IMUS, CANONICAL_JOINTS

JOINT_INDEX = {n: i for i, n in enumerate(CANONICAL_JOINTS)}
IMU_INDEX = {n: i for i, n in enumerate(CANONICAL_IMUS)}

# For each canonical IMU: the segment joint, ordered candidate "neighbor" joints
# that define the long (bone) axis, and ordered candidate "reference" joints used
# to disambiguate the remaining axes. First present candidate wins.
SEGMENT_FRAME: Dict[str, Tuple[str, List[str], List[str]]] = {
    "Pelvis":        ("Pelvis", ["L5", "Spine1", "T8", "Neck"], ["LeftUpperLeg", "RightUpperLeg"]),
    "T8":            ("T8", ["Neck", "Head", "T12", "L5"], ["LeftShoulder", "RightShoulder"]),
    "Head":          ("Head", ["Neck", "T8"], ["LeftShoulder", "RightShoulder"]),
    "LeftUpperArm":  ("LeftUpperArm", ["LeftForeArm", "LeftHand"], ["LeftShoulder", "T8"]),
    "RightUpperArm": ("RightUpperArm", ["RightForeArm", "RightHand"], ["RightShoulder", "T8"]),
    "LeftForeArm":   ("LeftForeArm", ["LeftHand", "LeftHandIndex"], ["LeftUpperArm", "LeftShoulder"]),
    "RightForeArm":  ("RightForeArm", ["RightHand", "RightHandIndex"], ["RightUpperArm", "RightShoulder"]),
    "LeftHand":      ("LeftHand", ["LeftHandIndex", "LeftForeArm"], ["LeftForeArm", "LeftUpperArm"]),
    "RightHand":     ("RightHand", ["RightHandIndex", "RightForeArm"], ["RightForeArm", "RightUpperArm"]),
    "LeftUpperLeg":  ("LeftUpperLeg", ["LeftLowerLeg", "LeftFoot"], ["Pelvis", "RightUpperLeg"]),
    "RightUpperLeg": ("RightUpperLeg", ["RightLowerLeg", "RightFoot"], ["Pelvis", "LeftUpperLeg"]),
    "LeftLowerLeg":  ("LeftLowerLeg", ["LeftFoot", "LeftToe"], ["LeftUpperLeg", "Pelvis"]),
    "RightLowerLeg": ("RightLowerLeg", ["RightFoot", "RightToe"], ["RightUpperLeg", "Pelvis"]),
    "LeftFoot":      ("LeftFoot", ["LeftToe", "LeftLowerLeg"], ["LeftLowerLeg", "LeftUpperLeg"]),
    "RightFoot":     ("RightFoot", ["RightToe", "RightLowerLeg"], ["RightLowerLeg", "RightUpperLeg"]),
}


def _first_present(cands: List[str], jmask) -> int:
    for name in cands:
        idx = JOINT_INDEX.get(name)
        if idx is not None and bool(jmask[idx]):
            return idx
    return -1


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def synth_free_acc(pos: torch.Tensor, fps: float, smooth_n: int = 4) -> torch.Tensor:
    """World-frame free acceleration from positions, smoothed to suppress the noise
    amplification of raw double-differencing (same wide-stencil idea as the repo's
    utils.data._syn_acc). pos [T,3] -> [T,3].

        a[i] = (p[i] + p[i+2n] - 2 p[i+n]) * fps^2 / n^2
    """
    t = pos.shape[0]
    acc = torch.zeros_like(pos)
    n = max(1, smooth_n)
    if t >= 2 * n + 1:
        acc[n:-n] = (pos[: t - 2 * n] + pos[2 * n:] - 2 * pos[n:-n]) * (float(fps) ** 2) / (n * n)
        acc[:n] = acc[n:n + 1]
        acc[-n:] = acc[-n - 1:-n]
    elif t >= 3:
        acc[1:-1] = (pos[:-2] + pos[2:] - 2 * pos[1:-1]) * (float(fps) ** 2)
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc


def _segment_orientation(seg: torch.Tensor, nb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Build [T,9] orientation frames. seg/nb/ref are [T,3] joint positions.

    long axis  y = dir(nb - seg);  z = y x (ref - seg);  x = y x z.
    Falls back to a global reference when (ref-seg) is colinear with y.
    """
    t = seg.shape[0]
    y = _normalize(nb - seg)
    r = ref - seg
    z = torch.cross(y, r, dim=-1)
    # fallback where ref is degenerate (near-colinear with the bone)
    bad = z.norm(dim=-1) < 1e-6
    if bad.any():
        up = torch.tensor([0.0, 1.0, 0.0], device=seg.device).expand_as(y)
        alt = torch.where((y[..., 1].abs() > 0.9).unsqueeze(-1),
                          torch.tensor([1.0, 0.0, 0.0], device=seg.device).expand_as(y), up)
        z = torch.where(bad.unsqueeze(-1), torch.cross(y, alt, dim=-1), z)
    z = _normalize(z)
    x = _normalize(torch.cross(y, z, dim=-1))
    # columns [x, y, z] -> rotation matrix, row-major flatten to 9d
    R = torch.stack([x, y, z], dim=-1)  # [T,3,3]
    return R.reshape(t, 9)


def synthesize_imu(joints: torch.Tensor, joint_mask: torch.Tensor, fps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """joints [T,J,3], joint_mask [J] -> (imu [T,15,12], imu_mask [15]).

    Fills orientation(9) + free-acceleration(3) for every canonical IMU whose
    segment joint and at least one neighbor joint are present.
    """
    t = joints.shape[0]
    imu = torch.zeros(t, len(CANONICAL_IMUS), 12, dtype=torch.float32)
    mask = torch.zeros(len(CANONICAL_IMUS), dtype=torch.bool)
    for imu_name, (seg_name, nb_cands, ref_cands) in SEGMENT_FRAME.items():
        seg_i = JOINT_INDEX.get(seg_name)
        if seg_i is None or not bool(joint_mask[seg_i]):
            continue
        nb_i = _first_present(nb_cands, joint_mask)
        if nb_i < 0:
            continue
        ref_i = _first_present(ref_cands, joint_mask)
        if ref_i < 0:
            ref_i = nb_i  # degenerate; handled by fallback in _segment_orientation
        dst = IMU_INDEX[imu_name]
        imu[:, dst, :9] = _segment_orientation(joints[:, seg_i], joints[:, nb_i], joints[:, ref_i])
        imu[:, dst, 9:] = synth_free_acc(joints[:, seg_i], fps)
        mask[dst] = True
    return imu, mask
