#!/usr/bin/env python3
"""Evaluate a whole-body recovery checkpoint with orientation and movement metrics.

The recovery target used here is 9D per joint:
  joint_orient_r6d (6D rotation) + joint_delta (3D adjacent-frame movement).

Because joint_delta is displacement per frame in metres, this script reports:
  - delta error in cm/frame, and
  - MPJVE-from-delta in mm/s = delta_error_m_per_frame * fps * 1000.

It does not report MPJPE unless the checkpoint was trained with absolute joint
position targets; the current recovery checkpoints use motion delta.
"""
import argparse
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dynaip.articulate as art
from masked_dataset import MaskedMotionDataset, apply_visible_imu_sampling, build_fixed_visible_sets, collate
from multiexpert_model import MultiExpert, orientation_6d_loss, motion_delta_loss
from rotation_utils import rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS, SMPL_JOINTS
from target_spec import slice_for, split_keys, target_dim
from train_multiexpert import geodesic_sums_per_joint, xyz_dist_sums_per_joint
from train_wholebody import (
    fixed_known_from_selection,
    load_joint_expert_bank,
    sample_known,
)
from wholebody_model import WholeBodyPoser


def _dict_get(args, key, default=None):
    return args[key] if key in args and args[key] is not None else default


def _slice_batch(batch, n: int):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch["imu"].shape[0],):
            out[key] = value[:n]
        elif isinstance(value, list) and len(value) == batch["imu"].shape[0]:
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _dataset_fps(data_root: str) -> float:
    try:
        meta = torch.load(f"{data_root.rstrip('/')}/metadata.pt", map_location="cpu")
    except FileNotFoundError:
        return 0.0
    return float(meta.get("window", {}).get("target_fps", 0.0))


def _load_combined_experts(path: str, target_dim_value: int, device):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    expert_dim = int(a.get("target_dim", target_dim(a.get("target", "joint_orient_r6d"))))
    if expert_dim != target_dim_value:
        raise ValueError(f"Experts target_dim={expert_dim} does not match recovery target_dim={target_dim_value}")
    experts = MultiExpert(
        len(CANONICAL_IMUS), len(CANONICAL_JOINTS),
        hidden_dim=int(a["hidden"]), nhead=int(a["nhead"]),
        num_layers=int(a["layers"]), target_dim=expert_dim,
    )
    experts.load_state_dict(ck["model"])
    experts.to(device).eval()
    for p in experts.parameters():
        p.requires_grad_(False)
    return experts


MOBILEPOSER_IGNORED_SMPL = [0, 7, 8, 10, 11, 20, 21, 22, 23]
MOBILEPOSER_MASK_SMPL = [2, 5, 16, 20]
MOBILEPOSER_METRIC_NAMES = [
    "SIP Error (deg)",
    "Angular Error (deg)",
    "Masked Angular Error (deg)",
    "Positional Error (cm)",
    "Masked Positional Error (cm)",
    "Mesh Error (cm)",
    "Jitter Error (100m/s^3)",
    "Distance Error (cm)",
]


def _ensure_chumpy_pickle_compat():
    """Old SMPL pickles may reference chumpy classes even when arrays are plain numpy."""
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")

    class Ch:
        def __array__(self, dtype=None):
            if not hasattr(self, "x"):
                raise TypeError("chumpy compatibility object has no numeric .x payload")
            return torch.as_tensor(self.x).cpu().numpy().astype(dtype, copy=False) if dtype else self.x

    chumpy.Ch = Ch
    chumpy_ch.Ch = Ch
    chumpy.ch = chumpy_ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy_ch


class MobilePoserMetricAccumulator:
    """MobilePoser-compatible metric formulas for SMPL-24 pose outputs.

    The official MobilePoser evaluator consumes SMPL local rotations plus global
    translation. Our recovery target stores root-normalized canonical rotations,
    so this adapter maps the shared SMPL joints back to local rotations before
    applying the same FK-based metric definitions and scaling.
    """
    def __init__(self, smpl_file: str, device: torch.device, fps: float):
        self.device = device
        self.fps = float(fps) if fps > 0 else 60.0
        _ensure_chumpy_pickle_compat()
        self.body_model = art.ParametricModel(smpl_file, device=device)
        self.smpl_from_canonical = torch.tensor(
            [CANONICAL_JOINTS.index(name) for name in SMPL_JOINTS],
            device=device, dtype=torch.long,
        )
        needed = [
            CANONICAL_JOINTS.index(SMPL_JOINTS[i])
            for i in range(len(SMPL_JOINTS))
            if i not in MOBILEPOSER_IGNORED_SMPL
        ]
        self.needed_canonical = torch.tensor(needed, device=device, dtype=torch.long)
        self.ignored = torch.tensor(MOBILEPOSER_IGNORED_SMPL, device=device, dtype=torch.long)
        self.mask = torch.tensor(MOBILEPOSER_MASK_SMPL, device=device, dtype=torch.long)
        self.eye = torch.eye(3, device=device)
        self.rows = []
        self.skipped = 0

    def _canonical_to_local_smpl(self, r6d: torch.Tensor) -> torch.Tensor:
        smpl_r6d = r6d.index_select(1, self.smpl_from_canonical)
        root_relative_global = rotation_6d_to_matrix(smpl_r6d).view(-1, len(SMPL_JOINTS), 3, 3)
        return self.body_model.inverse_kinematics_R(root_relative_global)

    @staticmethod
    def _mean_std(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_tensor([float("nan"), float("nan")])
        if x.ndim == 1:
            return torch.stack([x.mean(), x.std(unbiased=True)])
        return torch.stack([x.mean(), x.std(dim=0, unbiased=True).mean()])

    @staticmethod
    def _angle_deg(rot_p: torch.Tensor, rot_t: torch.Tensor) -> torch.Tensor:
        rel = rot_p.transpose(-1, -2) @ rot_t
        cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1.0) / 2.0
        return torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / torch.pi)

    def _eval_local_pose(self, pose_p: torch.Tensor, pose_t: torch.Tensor) -> torch.Tensor:
        pose_p = pose_p.clone().view(-1, len(SMPL_JOINTS), 3, 3)
        pose_t = pose_t.clone().view(-1, len(SMPL_JOINTS), 3, 3)
        pose_p[:, self.ignored] = self.eye
        pose_t[:, self.ignored] = self.eye

        glob_p, joint_p, vertex_p = self.body_model.forward_kinematics(pose_p, calc_mesh=True)
        glob_t, joint_t, vertex_t = self.body_model.forward_kinematics(pose_t, calc_mesh=True)

        offset = (joint_t[:, :1] - joint_p[:, :1])
        joint_err = (joint_p + offset - joint_t).norm(dim=2)
        vertex_err = (vertex_p + offset - vertex_t).norm(dim=2)
        local_ang = self._angle_deg(pose_p, pose_t)
        global_ang = self._angle_deg(glob_p, glob_t)
        masked_joint_err = joint_err[:, self.mask]
        masked_global_ang = global_ang[:, self.mask]

        if joint_p.shape[0] > 3:
            jitter = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3])
                      * (self.fps ** 3)).norm(dim=2)
        else:
            jitter = joint_p.new_empty((0, len(SMPL_JOINTS)))

        nan = pose_p.new_tensor([float("nan"), float("nan")])
        return torch.stack([
            self._mean_std(masked_global_ang),
            self._mean_std(global_ang),
            self._mean_std(masked_global_ang),
            self._mean_std(joint_err * 100.0),
            self._mean_std(masked_joint_err * 100.0),
            self._mean_std(vertex_err * 100.0),
            self._mean_std(jitter / 100.0),
            nan,  # unavailable: poser_mle_orient shards do not carry global tran.
        ])

    def add_batch(self, pred6: torch.Tensor, tgt6: torch.Tensor,
                  avail: torch.Tensor, lengths: torch.Tensor):
        for i in range(pred6.shape[0]):
            if not bool(avail[i].index_select(0, self.needed_canonical).all()):
                self.skipped += 1
                continue
            n = int(lengths[i].item())
            if n <= 0:
                self.skipped += 1
                continue
            pred_local = self._canonical_to_local_smpl(pred6[i, :n].float())
            tgt_local = self._canonical_to_local_smpl(tgt6[i, :n].float())
            self.rows.append(self._eval_local_pose(pred_local, tgt_local).detach().cpu())

    def summary(self):
        if not self.rows:
            return None
        stacked = torch.stack(self.rows)
        return torch.nanmean(stacked, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/wholebody_phase2_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=0, help="0=use checkpoint batch size")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="0=read target_fps from metadata.pt; used for MPJVE conversion")
    ap.add_argument("--mobileposer-metrics", action=argparse.BooleanOptionalAction, default=True,
                    help="also report MobilePoser-style FK pose/mesh/jitter metrics")
    ap.add_argument("--mobileposer-smpl",
                    default="smpl_models/basicmodel_m.pkl",
                    help="SMPL model file used by the cloned official MobilePoser evaluator")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="debug cap; 0 evaluates the full split")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    target_keys = split_keys(_dict_get(a, "target", "joint_orient_r6d,joint_delta"))
    orient_slice = slice_for(target_keys, "joint_orient_r6d")
    delta_slice = slice_for(target_keys, "joint_delta")
    if orient_slice is None or delta_slice is None:
        raise ValueError(f"Checkpoint target must include joint_orient_r6d,joint_delta, got {target_keys}")
    target_dim_value = int(_dict_get(a, "target_dim", target_dim(target_keys)))

    data = _dict_get(a, "data", "data")
    manifest = _dict_get(a, "manifest", f"{data.rstrip('/')}/splits.json")
    mask_key = _dict_get(a, "mask_key", "auto")
    eval_imus = int(_dict_get(a, "eval_imus", 3))
    seed = int(_dict_get(a, "seed", 0))
    batch_size = args.batch_size or int(_dict_get(a, "batch_size", 64))
    fps = args.fps if args.fps > 0 else _dataset_fps(data)

    # Build val too because the trainer's fixed visible sets are chosen from val+test.
    val_ds = MaskedMotionDataset(
        data, augment=False, target_key=",".join(target_keys), mask_key=mask_key,
        manifest=manifest, split="val", require_min_imus=True,
    )
    eval_ds = MaskedMotionDataset(
        data, augment=False, target_key=",".join(target_keys), mask_key=mask_key,
        manifest=manifest, split=args.split, require_min_imus=True,
    )
    visible_sets = build_fixed_visible_sets(
        [val_ds, eval_ds], seed=seed, min_k=eval_imus, max_k=eval_imus, split="val",
    )
    dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate, pin_memory=(dev.type == "cuda"))

    model = WholeBodyPoser(
        _dict_get(a, "prior", "pretrained/student_kl_18to21_best_64.pth"),
        lora_r=int(_dict_get(a, "lora_r", 8)),
        train_prior=_dict_get(a, "train_prior", "ln_lora"),
        residual=bool(_dict_get(a, "residual", True)),
        target_dim=target_dim_value,
    ).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    combined_experts = None
    joint_experts = None
    if _dict_get(a, "experts", ""):
        combined_experts = _load_combined_experts(a["experts"], target_dim_value, dev)
    elif _dict_get(a, "joint_expert_dir", ""):
        joint_experts = load_joint_expert_bank(
            a["joint_expert_dir"], _dict_get(a, "selected_expert_idxs", []), target_dim_value, dev,
        )
    mobileposer_eval = None
    if args.mobileposer_metrics:
        if not Path(args.mobileposer_smpl).exists():
            raise FileNotFoundError(
                f"MobilePoser SMPL file not found: {args.mobileposer_smpl}. "
                "Clone SPICExLAB/MobilePoser or pass --mobileposer-smpl."
            )
        mobileposer_eval = MobilePoserMetricAccumulator(args.mobileposer_smpl, dev, fps)

    J = len(CANONICAL_JOINTS)
    sums = {
        "unk_o_num": torch.zeros(J), "unk_o_den": torch.zeros(J),
        "all_o_num": torch.zeros(J), "all_o_den": torch.zeros(J),
        "unk_x_num": torch.zeros(J), "unk_x_den": torch.zeros(J),
        "all_x_num": torch.zeros(J), "all_x_den": torch.zeros(J),
    }
    raw_o = raw_x = batches = 0.0
    gen = torch.Generator(device=dev).manual_seed(0)
    seen_windows = 0

    with torch.no_grad():
        for batch in dl:
            if args.max_windows and seen_windows >= args.max_windows:
                break
            if args.max_windows:
                keep = min(batch["imu"].shape[0], args.max_windows - seen_windows)
                if keep < batch["imu"].shape[0]:
                    batch = _slice_batch(batch, keep)
            seen_windows += int(batch["imu"].shape[0])
            batch = apply_visible_imu_sampling(
                batch, epoch=0, batch_idx=0, seed=seed,
                min_k=eval_imus, max_k=eval_imus, split="val",
                fixed_per_dataset=True, visible_sets=visible_sets,
            )
            imu = batch["imu"].to(dev, non_blocking=True)
            tgt = batch["target"].to(dev, non_blocking=True)
            imu_mask = batch["imu_mask"].to(dev, non_blocking=True)
            avail = batch["mask"].to(dev, non_blocking=True)
            lengths = batch["lengths"].to(dev, non_blocking=True)

            known = sample_known(avail, int(_dict_get(a, "eval_known", 4)),
                                 int(_dict_get(a, "eval_known", 4)), gen)
            if joint_experts is not None:
                jin, valid = joint_experts(imu, imu_mask, lengths)
                known = valid & avail
            elif combined_experts is not None:
                jin = combined_experts(imu, imu_mask, lengths)
                selected = _dict_get(a, "selected_expert_idxs", [])
                if selected:
                    known = fixed_known_from_selection(avail, selected)
            else:
                jin = tgt

            pred = model(jin, known, imu, imu_mask, lengths)
            unk = avail & ~known

            oa, ob = orient_slice
            xa, xb = delta_slice
            if mobileposer_eval is not None:
                mobileposer_eval.add_batch(
                    pred[..., oa:ob].float(), tgt[..., oa:ob].float(), avail, lengths,
                )
            raw_o += float(orientation_6d_loss(pred[..., oa:ob], tgt[..., oa:ob], avail, lengths, per_joint=False))
            raw_x += float(motion_delta_loss(pred[..., xa:xb], tgt[..., xa:xb], avail, lengths, per_joint=False))
            batches += 1

            n, d = geodesic_sums_per_joint(pred[..., oa:ob].float(), tgt[..., oa:ob].float(), unk, lengths)
            sums["unk_o_num"] += n.cpu(); sums["unk_o_den"] += d.cpu()
            n, d = geodesic_sums_per_joint(pred[..., oa:ob].float(), tgt[..., oa:ob].float(), avail, lengths)
            sums["all_o_num"] += n.cpu(); sums["all_o_den"] += d.cpu()
            n, d = xyz_dist_sums_per_joint(pred[..., xa:xb].float(), tgt[..., xa:xb].float(), unk, lengths)
            sums["unk_x_num"] += n.cpu(); sums["unk_x_den"] += d.cpu()
            n, d = xyz_dist_sums_per_joint(pred[..., xa:xb].float(), tgt[..., xa:xb].float(), avail, lengths)
            sums["all_x_num"] += n.cpu(); sums["all_x_den"] += d.cpu()

    def ratio(num_key, den_key):
        out = sums[num_key] / sums[den_key].clamp_min(1.0)
        out[sums[den_key] == 0] = float("nan")
        return out

    unk_deg = ratio("unk_o_num", "unk_o_den")
    all_deg = ratio("all_o_num", "all_o_den")
    unk_cm = ratio("unk_x_num", "unk_x_den")
    all_cm = ratio("all_x_num", "all_x_den")
    unk_mpjve = unk_cm * 10.0 * fps if fps > 0 else torch.full_like(unk_cm, float("nan"))
    all_mpjve = all_cm * 10.0 * fps if fps > 0 else torch.full_like(all_cm, float("nan"))

    def macro(x):
        valid = ~torch.isnan(x)
        return float(x[valid].mean()) if valid.any() else float("nan")

    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} phase="
          f"{'II(experts)' if combined_experts is not None or joint_experts is not None else 'I(GT)'}")
    print(f"[data] split={args.split} windows={len(eval_ds)} fixed_visible={visible_sets} fps={fps:.1f}")
    print(f"[summary:unknown] MPJRE={macro(unk_deg):.2f}deg "
          f"motion_delta={macro(unk_cm):.3f}cm/frame MPJVE={macro(unk_mpjve):.1f}mm/s")
    print(f"[summary:all]     MPJRE={macro(all_deg):.2f}deg "
          f"motion_delta={macro(all_cm):.3f}cm/frame MPJVE={macro(all_mpjve):.1f}mm/s "
          f"raw_orient={raw_o/max(batches,1):.5f} raw_delta={raw_x/max(batches,1):.5f}")

    if mobileposer_eval is not None:
        mp = mobileposer_eval.summary()
        print("\n[mobileposer-style] official metric names/scales from SPICExLAB/MobilePoser")
        if mp is None:
            print("  no windows had the required SMPL-24 supervised joints")
        else:
            print(f"  windows={len(mobileposer_eval.rows)} skipped={mobileposer_eval.skipped} "
                  f"smpl={args.mobileposer_smpl}")
            for name, row in zip(MOBILEPOSER_METRIC_NAMES, mp):
                if torch.isnan(row[0]):
                    print(f"  {name}: n/a")
                else:
                    print(f"  {name}: {float(row[0]):.2f} (+/- {float(row[1]):.2f})")
            print("  note: Distance Error is n/a because this recovery dataset stores root-relative "
                  "targets, not MobilePoser global tran.")

    print(f"\n{'joint':>18s}  {'unk_deg':>8s} {'unk_cm/f':>9s} {'unk_mpjve':>10s}  "
          f"{'all_deg':>8s} {'all_cm/f':>9s} {'all_mpjve':>10s}")
    for j, name in enumerate(CANONICAL_JOINTS):
        def fmt(value, suffix="", width=8, prec=2):
            if torch.isnan(value):
                return f"{'n/a':>{width}s}"
            return f"{float(value):>{width}.{prec}f}{suffix}"
        print(f"{name:>18s}  "
              f"{fmt(unk_deg[j], width=8, prec=2)} {fmt(unk_cm[j], width=9, prec=3)} "
              f"{fmt(unk_mpjve[j], width=10, prec=1)}  "
              f"{fmt(all_deg[j], width=8, prec=2)} {fmt(all_cm[j], width=9, prec=3)} "
              f"{fmt(all_mpjve[j], width=10, prec=1)}")

    print("\n[note] MPJVE is derived from joint_delta. If you need MPJPE, train/evaluate a target "
          "that includes joint_position_root_relative or integrate deltas with a defined root trajectory.")


if __name__ == "__main__":
    main()
