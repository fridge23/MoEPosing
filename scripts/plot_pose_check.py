#!/usr/bin/env python3
"""Plot a random human-pose frame from each dataset to eyeball coordinate frame,
scale and orientation. Renders both the unified training shards and the raw
source files, side by side (front X-Y and side Z-Y, vertical axis = Y).

Run with the env that has torch+numpy+matplotlib:
    /home/pengfei/Downloads/.codex-data-env/bin/python scripts/plot_pose_check.py --mode both
"""
import argparse
import csv
import glob
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynaip" / "expert_pretrain"))
from schema import CANONICAL_JOINTS, SMPL_JOINTS, XSENS_JOINTS  # noqa: E402

DATA_ROOT = Path("/home/pengfei/Downloads/data")
UNIFIED_ROOT = Path("/home/pengfei/Downloads/poser_mle_unified_60hz_by_dataset")

# --- skeleton edges defined by joint NAME (index-agnostic) ---------------------
# Superset over canonical names; a bone is drawn only when both endpoints are
# present (mask=True) and finite, so xsens-spine and smpl-spine each draw cleanly.
CANONICAL_EDGES = [
    # legs
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToe"),
    # spine (xsens chain)
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "Neck"), ("Neck", "Head"),
    # spine (smpl chain)
    ("Pelvis", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"),
    ("Spine3", "Neck"),
    # shoulders/arms, connect to both T8 and Neck so either skeleton links up
    ("T8", "LeftShoulder"), ("Neck", "LeftShoulder"),
    ("LeftShoulder", "LeftUpperArm"), ("LeftUpperArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"), ("LeftHand", "LeftHandIndex"),
    ("T8", "RightShoulder"), ("Neck", "RightShoulder"),
    ("RightShoulder", "RightUpperArm"), ("RightUpperArm", "RightForeArm"),
    ("RightForeArm", "RightHand"), ("RightHand", "RightHandIndex"),
]

SMPL_EDGES = [
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToe"),
    ("Pelvis", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"),
    ("Spine3", "Neck"), ("Neck", "Head"),
    ("Spine3", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("LeftHand", "LeftHandIndex"),
    ("Spine3", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("RightHand", "RightHandIndex"),
]

XSENS_EDGES = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("T8", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToe"),
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
]

# raw vidimu BodyTrack keypoint names -> edges (their own naming)
VIDIMU_EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "torso"), ("torso", "neck"), ("neck", "nose"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

# raw CeTI body-part names -> edges (their own naming, <part>_position_*)
CETI_EDGES = [
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"),
    ("Pelvis", "Chest"), ("Chest", "Head"),
    ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
]


def _draw(ax, pts, names, edges, hx, vy, title_axis):
    """pts: dict name->(x,y,z) for present joints. Plot axis hx (horizontal) vs vy (vertical)."""
    comp = {"x": 0, "y": 1, "z": 2}
    h, v = comp[hx], comp[vy]
    xs = [p[h] for p in pts.values()]
    ys = [p[v] for p in pts.values()]
    ax.scatter(xs, ys, s=18, c="tab:blue", zorder=3)
    for a, b in edges:
        if a in pts and b in pts:
            ax.plot([pts[a][h], pts[b][h]], [pts[a][v], pts[b][v]],
                    c="tab:gray", lw=1.5, zorder=2)
    for label in ("Head", "nose"):
        if label in pts:
            ax.annotate("HEAD", (pts[label][h], pts[label][v]),
                        color="tab:red", fontsize=9, weight="bold")
    if "Pelvis" in pts:
        ax.scatter([pts["Pelvis"][h]], [pts["Pelvis"][v]], s=40,
                   c="tab:green", marker="s", zorder=4, label="Pelvis(root)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(hx)
    ax.set_ylabel(vy)
    ax.set_title(title_axis)
    ax.grid(True, alpha=0.3)


def render(pts, edges, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    _draw(axes[0], pts, None, edges, "x", "y", "front (X-Y)")
    _draw(axes[1], pts, None, edges, "z", "y", "side (Z-Y)")
    ys = [p[1] for p in pts.values()]
    yr = max(ys) - min(ys) if ys else 0
    fig.suptitle(f"{title}\nY range = {yr:.3f}  (vertical axis is Y; head-up means Y grows upward)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[saved] {out_path}  (Y range {yr:.3f})")


# --- unified shards ------------------------------------------------------------
def _find_source_sample(source, rng):
    """Return a random sample for `source` from either the by_dataset layout
    (UNIFIED_ROOT/source/split/shard) or a flat layout (root/shard with a
    'source' field). Controlled by the module-level UNIFIED_ROOT."""
    nested = sorted(glob.glob(str(UNIFIED_ROOT / source / "*" / "shard_*.pt")))
    if nested:
        s = torch.load(rng.choice(nested), map_location="cpu")["samples"]
        return s[rng.randrange(len(s))]
    flat = sorted(glob.glob(str(UNIFIED_ROOT / "shard_*.pt")))
    rng.shuffle(flat)
    for path in flat:
        hits = [x for x in torch.load(path, map_location="cpu")["samples"]
                if str(x.get("source")) == source]
        if hits:
            return hits[rng.randrange(len(hits))]
    return None


def plot_unified(source, rng, out_dir, n_frames):
    s = _find_source_sample(source, rng)
    if s is None:
        print(f"[skip] unified {source}: no samples")
        return
    jpr = s["joint_position_root_relative"].numpy()  # [T,28,3]
    mask = s["joint_mask"].numpy().astype(bool)
    T = jpr.shape[0]
    frames = rng.sample(range(T), min(n_frames, T))
    for fi in frames:
        frame = jpr[fi]
        pts = {CANONICAL_JOINTS[j]: frame[j] for j in range(len(CANONICAL_JOINTS))
               if mask[j] and np.isfinite(frame[j]).all()}
        imucount = int(s["imu_mask"].sum()) if "imu_mask" in s else 0
        title = (f"UNIFIED {source} | {s['quality']} | jmask={int(mask.sum())} imu={imucount} | "
                 f"{Path(str(s['motion'])).name} | frame {fi}/{T}")
        out = out_dir / f"unified_{source}_f{fi}.png"
        render(pts, CANONICAL_EDGES, title, out)


# --- raw sources ---------------------------------------------------------------
def _pts_from_array(arr, names):
    return {names[i]: arr[i] for i in range(min(len(names), arr.shape[0]))
            if np.isfinite(arr[i]).all()}


def plot_raw(source, rng, out_dir, n_frames):
    if source == "vidimu":
        files = sorted(glob.glob(str(DATA_ROOT / "vidimu/dataset/videoandimusync/**/*.csv"), recursive=True))
        if not files:
            print("[skip] raw vidimu: no csv"); return
        f = rng.choice(files)
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        kps = sorted({c.rsplit("_", 1)[0] for c in rows[0] if c.rsplit("_", 1)[-1] in ("x", "y", "z")})
        for fi in rng.sample(range(len(rows)), min(n_frames, len(rows))):
            r = rows[fi]
            pts = {}
            for kp in kps:
                try:
                    v = np.array([float(r[f"{kp}_x"]), float(r[f"{kp}_y"]), float(r[f"{kp}_z"])])
                except (KeyError, ValueError):
                    continue
                if np.isfinite(v).all():
                    pts[kp] = v
            render(pts, VIDIMU_EDGES, f"RAW vidimu (BodyTrack camera frame, mm) | {Path(f).name} | frame {fi}",
                   out_dir / f"raw_vidimu_f{fi}.png")

    elif source == "ceti_age_kinematics":
        files = sorted(glob.glob(str(DATA_ROOT / "CeTI-Age-Kinematics/ceti-age-kinematics/**/*_motion.tsv"), recursive=True))
        if not files:
            print("[skip] raw ceti: none"); return
        f = rng.choice(files)
        with open(f) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        parts = sorted({c.rsplit("_position_", 1)[0] for c in rows[0] if "_position_" in c})
        for fi in rng.sample(range(len(rows)), min(n_frames, len(rows))):
            r = rows[fi]
            pts = {}
            for p in parts:
                try:
                    v = np.array([float(r[f"{p}_position_x"]), float(r[f"{p}_position_y"]), float(r[f"{p}_position_z"])])
                except (KeyError, ValueError):
                    continue
                pts[p] = v
            render(pts, CETI_EDGES, f"RAW CeTI (Rokoko, cm) | {Path(f).name} | frame {fi}",
                   out_dir / f"raw_ceti_f{fi}.png")

    elif source == "ultra_mocap":
        files = sorted(glob.glob(str(DATA_ROOT / "ULTRA-MoCap/**/Raw Marker/*.trc"), recursive=True))
        if not files:
            print("[skip] raw ultra: none"); return
        f = rng.choice(files)
        with open(f) as fh:
            head = [next(fh).rstrip("\n") for _ in range(6)]
            rest = fh.readlines()
        markers = [x for x in head[3].split("\t")[2:] if x]
        rows = [ln.strip().split("\t") for ln in rest if ln.strip()]
        rows = [r for r in rows if len(r) >= 2 + 3 * len(markers)]
        for fi in rng.sample(range(len(rows)), min(n_frames, len(rows))):
            vals = np.array([float(x) for x in rows[fi][2:2 + 3 * len(markers)]]).reshape(-1, 3)
            pts = {markers[i]: vals[i] for i in range(len(markers)) if np.isfinite(vals[i]).all()}
            render(pts, [], f"RAW ULTRA markers (Vicon, mm) | {Path(f).name} | frame {fi}",
                   out_dir / f"raw_ultra_f{fi}.png")

    elif source == "mobileposer":
        files = sorted(glob.glob(str(DATA_ROOT / "mobileposer/*.pt")))
        if not files:
            print("[skip] raw mobileposer: none"); return
        f = rng.choice(files)
        obj = torch.load(f, map_location="cpu")
        seqs = obj["joint"]
        seq = seqs[rng.randrange(len(seqs))].numpy()  # [T,24,3]
        for fi in rng.sample(range(len(seq)), min(n_frames, len(seq))):
            pts = _pts_from_array(seq[fi], SMPL_JOINTS)
            render(pts, SMPL_EDGES, f"RAW mobileposer (SMPL joints, m) | {Path(f).name} | frame {fi}",
                   out_dir / f"raw_mobileposer_f{fi}.png")

    elif source.startswith("dynaip_"):
        ds = source.split("_", 1)[1]
        files = sorted(glob.glob(str(Path("/home/pengfei/Downloads/dynaip/datasets/extract") / ds / "*.pt")))
        if not files:
            print(f"[skip] raw {source}: none"); return
        f = rng.choice(files)
        d = torch.load(f, map_location="cpu")
        pos = d["joint"]["position"].numpy()  # [T,23,3]
        names = d["joint"].get("name", XSENS_JOINTS)
        for fi in rng.sample(range(len(pos)), min(n_frames, len(pos))):
            pts = _pts_from_array(pos[fi], names)
            render(pts, XSENS_EDGES, f"RAW {source} (Xsens joints, m) | {Path(f).name} | frame {fi}",
                   out_dir / f"raw_{source}_f{fi}.png")
    else:
        print(f"[skip] raw {source}: no raw handler")


ALL_SOURCES = ["ceti_age_kinematics", "dynaip_andy", "dynaip_cip", "dynaip_emokine",
               "dynaip_unipd", "dynaip_virginia", "mobileposer", "ultra_mocap", "vidimu"]


def main():
    global UNIFIED_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unified", "raw", "both"], default="both")
    ap.add_argument("--sources", default=",".join(ALL_SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--out", default="/home/pengfei/Downloads/pose_check")
    ap.add_argument("--unified-root", default=str(UNIFIED_ROOT),
                    help="dir with unified shards (by_dataset nested or flat shard_*.pt)")
    args = ap.parse_args()
    UNIFIED_ROOT = Path(args.unified_root)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    for src in sources:
        if args.mode in ("unified", "both"):
            try:
                plot_unified(src, rng, out_dir, args.frames)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] unified {src}: {type(exc).__name__}: {exc}")
        if args.mode in ("raw", "both"):
            try:
                plot_raw(src, rng, out_dir, args.frames)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] raw {src}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
