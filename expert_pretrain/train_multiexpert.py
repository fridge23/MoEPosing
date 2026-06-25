#!/usr/bin/env python3
"""Train 28 independent per-joint transformer experts on GB10 (GPU).

Each expert sees the same on-the-fly 2-5 sparse-IMU augmentation and regresses a
per-joint 9D target vector: root-normalized orientation (6D rotation) plus
adjacent-frame xyz movement delta:
--target joint_orient_r6d,joint_delta.
Uses bf16 autocast (GB10 cc 12.1) and weighted per-joint losses. Convergence is
per joint: each expert is snapshotted at its own best-val epoch and (with
--freeze-converged) drops out of compute once it plateaus; the run stops when all
joints have converged.
"""
import argparse
import json
import math
import os
import time

import random

import torch
from torch.utils.data import DataLoader, Subset

from masked_dataset import (
    MaskedMotionDataset,
    apply_visible_imu_sampling,
    build_fixed_visible_sets,
    collate,
)
from multiexpert_model import MultiExpert, expert_target_loss, orientation_6d_loss, motion_delta_loss
from rotation_utils import normalize_rotation_6d, rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys, target_dim


def r6d_to_mat(r6d: torch.Tensor) -> torch.Tensor:
    """r6d [...,6] (two columns) -> rotation matrix [...,3,3] via Gram-Schmidt."""
    return rotation_6d_to_matrix(r6d)


def geodesic_deg(pred6, tgt6, mask, lengths):
    """Mean geodesic angle (degrees) between predicted and target rotations,
    over valid joints and unpadded frames."""
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / 3.14159265)
    b, t, j = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (ang * v).sum() / v.sum().clamp_min(1.0)


def geodesic_sums_per_joint(pred6, tgt6, mask, lengths):
    """Per-joint geodesic-angle numerator/denominator sums (degrees), so val can
    be accumulated correctly across batches and reported PER JOINT. Returns
    (num[J], den[J]); per-joint angle = num/den, pooled angle = num.sum()/den.sum()."""
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / 3.14159265)
    b, t, j = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (ang * v).sum(dim=(0, 1)), v.sum(dim=(0, 1))     # [J], [J]


def xyz_dist_sums_per_joint(pred_xyz, tgt_xyz, mask, lengths):
    """Per-joint Euclidean motion-delta error sums, in CENTIMETRES (joint_delta
    targets are root-relative per-frame movement in metres). Returns (num[J],
    den[J]); per-joint cm = num/den, pooled cm = num.sum()/den.sum()."""
    d = (pred_xyz - tgt_xyz).norm(dim=-1) * 100.0          # [B,T,J] metres -> cm
    b, t, j = d.shape
    steps = torch.arange(t, device=d.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (d * v).sum(dim=(0, 1)), v.sum(dim=(0, 1))      # [J], [J]


def component_loss_sums_per_joint(pred, target, mask, lengths, *, squared: bool, normalize_r6d: bool = False):
    """Per-joint raw loss numerator/denominator matching the training loss.

    Orientation uses MSE on normalized 6D rotations; xyz movement uses L1. The
    returned denominator counts target dimensions, so num/den is directly
    comparable to orientation_6d_loss / motion_delta_loss per joint.
    """
    pred = pred.float()
    target = target.float()
    if normalize_r6d:
        pred = normalize_rotation_6d(pred)
        target = normalize_rotation_6d(target)
    err = pred - target
    err = err.pow(2) if squared else err.abs()
    b, t, j, _ = err.shape
    steps = torch.arange(t, device=err.device).view(1, t, 1, 1)
    valid = mask.to(err.dtype).view(b, 1, j, 1)
    valid = valid * (steps < lengths.to(err.device).view(b, 1, 1, 1)).to(err.dtype)
    valid = valid.expand_as(err)
    return (err * valid).sum(dim=(0, 1, 3)), valid.sum(dim=(0, 1, 3))


def identity_baseline_deg(tgt6, mask, lengths):
    """Geodesic angle of targets from the identity rotation (the predict-identity baseline)."""
    ident = torch.zeros_like(tgt6); ident[..., 0] = 1.0; ident[..., 4] = 1.0
    return geodesic_deg(ident, tgt6, mask, lengths)


def add_imu_noise(imu: torch.Tensor, imu_mask: torch.Tensor, acc_std: float, ori_deg: float) -> torch.Tensor:
    """Sim-to-real IMU noise, vectorized on-device (applied in the training step, NOT
    in DataLoader workers). imu [B,T,S,12] = orientation_9d + acc_3. Per-sequence
    orientation bias (calibration error) + per-frame accel noise on present sensors."""
    b, t, s, _ = imu.shape
    out = imu
    if ori_deg > 0:
        axis = torch.randn(b, s, 3, device=imu.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ang = torch.randn(b, s, 1, 1, device=imu.device) * (ori_deg * math.pi / 180.0)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(b, s, 3, 3)
        R = torch.eye(3, device=imu.device).view(1, 1, 3, 3) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)
        ori = out[..., :9].view(b, t, s, 3, 3)
        ori = torch.einsum("bsij,btsjk->btsik", R, ori).reshape(b, t, s, 9)
        out = torch.cat([ori, out[..., 9:]], dim=-1)
    if acc_std > 0:
        m = imu_mask.to(imu.dtype).view(b, 1, s, 1)
        out = torch.cat([out[..., :9], out[..., 9:] + torch.randn(b, t, s, 3, device=imu.device) * acc_std * m], dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pengfei/Downloads/poser_mle_orient")
    ap.add_argument("--epochs", type=int, default=100, help="max epochs (a generous cap; "
                    "training stops earlier once val converges, see --patience)")
    ap.add_argument("--patience", type=int, default=15, help="0=off; else stop when the macro "
                    "val MPJRE has not improved for this many epochs (convergence-based)")
    ap.add_argument("--freeze-converged", action=argparse.BooleanOptionalAction, default=True,
                    help="drop converged experts out of training compute (forward+backward), so "
                         "the GPU only works on the still-training joints. Identical final weights.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--target", default="joint_orient_r6d,joint_delta",
                    help="comma-separated target tensors to concatenate per joint/frame")
    ap.add_argument("--mask-key", default="auto",
                    help="'auto' uses orient_mask for joint_orient_r6d and joint_mask for xyz targets; "
                         "also accepts one mask key or comma-separated mask keys")
    ap.add_argument("--target-dim", type=int, default=0,
                    help="0=derive from --target; otherwise override for old checkpoints/experiments")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--manifest", default="/home/pengfei/Downloads/poser_mle_orient/splits.json",
                    help="frozen sequence-level split manifest; leak-free. Set '' to fall back to "
                         "random window split + --holdout-kw keyword test set.")
    ap.add_argument("--loss", default="per_joint", choices=["per_joint", "pooled"],
                    help="per_joint: each joint normalized & weighted equally (recommended); "
                         "pooled: one mean over all valid elements (the old behaviour).")
    ap.add_argument("--lambda-orientation", type=float, default=1.0)
    ap.add_argument("--lambda-motion-delta", type=float, default=1.0,
                    help="fixed xyz weight (only used when --loss-balance fixed)")
    ap.add_argument("--loss-balance", default="adaptive", choices=["fixed", "adaptive"],
                    help="adaptive: dynamically weight xyz so weighted_xyz ~= xyz_weight * weighted_orient "
                         "throughout training (xyz loss collapses fast, so a fixed weight drifts)")
    ap.add_argument("--xyz-weight", type=float, default=2.0,
                    help="adaptive target ratio: weighted xyz error / weighted orientation error")
    ap.add_argument("--ema-momentum", type=float, default=0.98)
    ap.add_argument("--sample-visible-per-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="deterministically sample a dataset-valid 2-5 visible IMU set every train batch")
    ap.add_argument("--eval-fixed-visible-set", action=argparse.BooleanOptionalAction, default=True,
                    help="use one fixed dataset-specific visible IMU set for validation/test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False,
                    help="preload all split samples into RAM once (workers inherit via fork) so "
                         "epochs are compute-bound, not disk-bound (~9GB for the 53k train split)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs; >0 = smoke test")
    ap.add_argument("--log-every", type=int, default=100,
                    help="print running train loss every N batches; 0 disables mid-epoch logging")
    ap.add_argument("--per-joint-log", default="",
                    help="optional JSONL path for full per-joint val orient_deg/xyz_cm/raw/total each epoch")
    ap.add_argument("--early-stop-metric", default="orient_deg", choices=["orient_deg", "total_loss"],
                    help="per-joint score used for best snapshots and patience")
    ap.add_argument("--min-delta", type=float, default=None,
                    help="minimum per-joint improvement; default 1e-3 deg or 1e-6 total_loss")
    ap.add_argument("--holdout-kw", default="dip,totalcapture,imuposer",
                    help="comma-separated motion-name keywords held OUT of train/val and used as a "
                         "separate test set (DIP & TotalCapture are the standard sparse-IMU benchmarks).")
    ap.add_argument("--acc-noise", type=float, default=0.0,
                    help="sim-to-real: per-frame accelerometer Gaussian noise std (m/s^2) on train inputs")
    ap.add_argument("--ori-noise-deg", type=float, default=0.0,
                    help="sim-to-real: per-sequence orientation bias std (deg) on train inputs")
    ap.add_argument("--save", default="/home/pengfei/Downloads/dynaip/weights/multiexpert.pt")
    ap.add_argument("--resume", default="",
                    help="resume model weights from a checkpoint; if <resume>_best.pt exists, "
                         "restore per-joint best snapshots from it")
    ap.add_argument("--resume-reset-epoch", action=argparse.BooleanOptionalAction, default=False,
                    help="load resume weights but start the new training schedule at epoch 0")
    ap.add_argument("--resume-reset-best", action=argparse.BooleanOptionalAction, default=False,
                    help="load resume weights but start per-joint best/early-stop tracking fresh")
    args = ap.parse_args()
    if args.target_dim <= 0:
        args.target_dim = target_dim(args.target)
    args.target_keys = split_keys(args.target)
    rotation_key, orient_slice = next(
        ((key, slice_for(args.target_keys, key))
         for key in ("joint_orient_r6d", "joint_rot_delta_r6d")
         if slice_for(args.target_keys, key) is not None),
        (None, None),
    )
    motion_key, motion_delta_slice = next(
        ((key, slice_for(args.target_keys, key))
         for key in ("joint_delta", "joint_delta_local")
         if slice_for(args.target_keys, key) is not None),
        (None, None),
    )
    if orient_slice is None or motion_delta_slice is None:
        raise ValueError(
            "--target must include either joint_orient_r6d or joint_rot_delta_r6d, "
            "and either joint_delta or joint_delta_local"
        )
    args.rotation_target_key = rotation_key
    args.motion_target_key = motion_key
    holdout = [k.strip() for k in args.holdout_kw.split(",") if k.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # Allocate CUDA parameters before optional CPU preload. On GB10/unified-memory
    # systems, preloading tens of GB of sample tensors first can make model.to(cuda)
    # fail even though the model itself is small.
    model = MultiExpert(len(CANONICAL_IMUS), len(CANONICAL_JOINTS),
                        hidden_dim=args.hidden, nhead=args.nhead, num_layers=args.layers,
                        target_dim=args.target_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        resume_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = 0 if args.resume_reset_epoch else resume_epoch
        reset_msg = "; reset epoch schedule to 0" if args.resume_reset_epoch else ""
        print(f"[resume] loaded {args.resume} at checkpoint_epoch={resume_epoch}{reset_msg}")

    # Train sparse-masks every batch in run_batch(); val/test keep all present sensors.
    manifest = args.manifest or None
    if manifest:
        # leak-free: splits are by SEQUENCE (no motion straddles train/val).
        train_ds = MaskedMotionDataset(args.data, min_k=args.min_k, max_k=args.max_k, augment=False,
                                       target_key=args.target, mask_key=args.mask_key,
                                       manifest=manifest, split="train", require_min_imus=True,
                                       preload=args.preload)
        val_ds = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                     mask_key=args.mask_key, manifest=manifest, split="val",
                                     require_min_imus=True, preload=args.preload)
        test_full = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                        mask_key=args.mask_key, manifest=manifest, split="test",
                                        require_min_imus=True, preload=args.preload)
    else:
        # fallback: random WINDOW split (optimistic val) + keyword test set.
        train_full = MaskedMotionDataset(args.data, min_k=args.min_k, max_k=args.max_k, augment=False,
                                         target_key=args.target, mask_key=args.mask_key, exclude_kw=holdout,
                                         require_min_imus=True)
        val_full = MaskedMotionDataset(args.data, augment=False,
                                       target_key=args.target, mask_key=args.mask_key, exclude_kw=holdout,
                                       require_min_imus=True)
        n = len(train_full)
        n_val = max(1, int(n * args.val_frac))
        idx = list(range(n)); random.Random(0).shuffle(idx)
        val_ds = Subset(val_full, idx[:n_val])
        train_ds = Subset(train_full, idx[n_val:])
        test_full = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                        mask_key=args.mask_key, include_only_kw=holdout,
                                        require_min_imus=True)
    test_dl = None
    if len(test_full):
        test_dl = DataLoader(test_full, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate, pin_memory=True)
    eval_visible_sets = {}
    if args.eval_fixed_visible_set:
        eval_visible_sets = build_fixed_visible_sets(
            [val_ds, test_full], seed=args.seed, min_k=args.min_k, max_k=args.max_k, split="val",
        )
        print(f"[visible-eval] fixed visible IMUs by source: {eval_visible_sets}")
    print(f"[data] split={'manifest(seq-level)' if manifest else 'random-window'} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_full)} "
          f"imus={len(CANONICAL_IMUS)} joints={len(CANONICAL_JOINTS)} "
          f"target={args.target_keys} rotation={rotation_key} motion={motion_key} "
          f"target_dim={args.target_dim} loss={args.loss} balance={args.loss_balance} "
          f"lambda_orientation={args.lambda_orientation} lambda_motion_delta={args.lambda_motion_delta} "
          f"early_stop_metric={args.early_stop_metric}")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate,
                    pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, collate_fn=collate, pin_memory=True)

    print(f"[model] {len(CANONICAL_JOINTS)} independent experts, {n_params/1e6:.2f}M params total")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(dl)))
    if start_epoch:
        total_steps = args.epochs * max(1, len(dl))
        resume_steps = min(start_epoch * max(1, len(dl)), total_steps)
        resume_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * resume_steps / max(total_steps, 1)))
        for group in opt.param_groups:
            group["lr"] = resume_lr
        sched.last_epoch = resume_steps
        print(f"[resume] scheduler set to step={resume_steps} lr={resume_lr:.2e}")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and use_amp)

    # adaptive loss balancing state: EMA of the raw component losses + the current
    # weights (wo fixed, wx dynamic so weighted_xyz ~= xyz_weight * weighted_orient).
    ema = {"o": None, "x": None}
    lossw = {"wo": args.lambda_orientation, "wx": args.lambda_motion_delta}

    def component_weights(o_raw, x_raw, train: bool):
        if args.loss_balance != "adaptive":
            return args.lambda_orientation, args.lambda_motion_delta
        ov, xv = max(float(o_raw.detach()), 1e-8), max(float(x_raw.detach()), 1e-8)
        if ema["o"] is None:
            ema["o"], ema["x"] = ov, xv
        wo = args.lambda_orientation
        wx = args.xyz_weight * ema["o"] / max(ema["x"], 1e-8)   # dynamic so ratio ~= xyz_weight
        if train:
            mu = args.ema_momentum
            ema["o"] = mu * ema["o"] + (1 - mu) * ov
            ema["x"] = mu * ema["x"] + (1 - mu) * xv
        return wo, wx

    def run_batch(batch, train: bool, active=None, epoch: int = 0, batch_idx: int = 0):
        if train and args.sample_visible_per_batch:
            batch = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=batch_idx, seed=args.seed,
                min_k=args.min_k, max_k=args.max_k, split="train",
            )
        elif (not train) and args.eval_fixed_visible_set:
            batch = apply_visible_imu_sampling(
                batch, epoch=0, batch_idx=0, seed=args.seed,
                min_k=args.min_k, max_k=args.max_k, split="val",
                fixed_per_dataset=True, visible_sets=eval_visible_sets,
            )
        imu = batch["imu"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        imu_mask = batch["imu_mask"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        if train and (args.acc_noise > 0 or args.ori_noise_deg > 0):
            imu = add_imu_noise(imu, imu_mask, args.acc_noise, args.ori_noise_deg)
        per_joint = args.loss == "per_joint"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if train and active is not None:
                # only the still-training experts run; align target/mask to them
                pred = model(imu, imu_mask, lengths, active=active)
                t, m = target[:, :, active, :], mask[:, active]
            else:
                pred = model(imu, imu_mask, lengths)
                t, m = target, mask
            # RAW (unweighted) component errors, then the weighted total. Logging both
            # lets us see/tune the orientation-vs-xyz balance (xyz weighted ~2x orient).
            oa, ob = orient_slice
            o_raw = orientation_6d_loss(pred[..., oa:ob], t[..., oa:ob], m, lengths, per_joint=per_joint)
            if motion_delta_slice is not None:
                xa, xb = motion_delta_slice
                x_raw = motion_delta_loss(pred[..., xa:xb], t[..., xa:xb], m, lengths, per_joint=per_joint)
            else:
                x_raw = pred.new_zeros(())
            wo, wx = component_weights(o_raw, x_raw, train)
            lossw["wo"], lossw["wx"] = wo, wx
            loss = wo * o_raw + wx * x_raw
        if train:
            return loss, float(o_raw.detach()), float(x_raw.detach())
        # per-joint geodesic (deg), xyz distance (cm), and raw training-loss sums.
        # The raw sums let validation/early-stop use the exact same component losses
        # as training, but reported per expert.
        a, b = orient_slice
        num, den = geodesic_sums_per_joint(pred[..., a:b].float(), target[..., a:b].float(), mask, lengths)
        oraw_num, oraw_den = component_loss_sums_per_joint(
            pred[..., a:b], target[..., a:b], mask, lengths, squared=True, normalize_r6d=True,
        )
        if motion_delta_slice is not None:
            xa, xb = motion_delta_slice
            xnum, xden = xyz_dist_sums_per_joint(pred[..., xa:xb].float(), target[..., xa:xb].float(), mask, lengths)
            xraw_num, xraw_den = component_loss_sums_per_joint(
                pred[..., xa:xb], target[..., xa:xb], mask, lengths, squared=False, normalize_r6d=False,
            )
        else:
            z = torch.zeros(pred.shape[2], device=pred.device)
            xnum, xden, xraw_num, xraw_den = z, z, z, z
        return loss, num, den, xnum, xden, oraw_num, oraw_den, xraw_num, xraw_den, float(o_raw.detach()), float(x_raw.detach())

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    J = len(CANONICAL_JOINTS)

    def eval_per_joint(loader):
        """Return per-joint human metrics plus raw training-loss metrics.

        Output: (deg[J], cm[J], orient_raw[J], xyz_raw[J], pooled_loss,
        orient_raw_macro, xyz_raw_macro). deg/cm are for reading; raw values are
        what the optimizer and total-loss early stopping use.
        """
        model.eval()
        num = torch.zeros(J, device=device); den = torch.zeros(J, device=device)
        xnum = torch.zeros(J, device=device); xden = torch.zeros(J, device=device)
        oraw_num = torch.zeros(J, device=device); oraw_den = torch.zeros(J, device=device)
        xraw_num = torch.zeros(J, device=device); xraw_den = torch.zeros(J, device=device)
        ltot = 0.0; otot = 0.0; xtot = 0.0; ln = 0
        with torch.no_grad():
            for batch in loader:
                loss, n_, d_, xn_, xd_, on_, od_, xxn_, xxd_, o_raw, x_raw = run_batch(batch, False)
                ltot += float(loss); otot += o_raw; xtot += x_raw; ln += 1
                num += n_; den += d_
                xnum += xn_; xden += xd_
                oraw_num += on_; oraw_den += od_
                xraw_num += xxn_; xraw_den += xxd_
        per_joint = (num / den.clamp_min(1.0)).cpu()       # [J] deg; nan-safe
        per_joint[den.cpu() == 0] = float("nan")
        per_joint_cm = (xnum / xden.clamp_min(1.0)).cpu()  # [J] cm; nan-safe
        per_joint_cm[xden.cpu() == 0] = float("nan")
        per_joint_oraw = (oraw_num / oraw_den.clamp_min(1.0)).cpu()
        per_joint_oraw[oraw_den.cpu() == 0] = float("nan")
        per_joint_xraw = (xraw_num / xraw_den.clamp_min(1.0)).cpu()
        per_joint_xraw[xraw_den.cpu() == 0] = float("nan")
        return (per_joint, per_joint_cm, per_joint_oraw, per_joint_xraw,
                ltot / max(ln, 1), otot / max(ln, 1), xtot / max(ln, 1))

    step = 0
    # per-joint early stopping: each expert is snapshotted at ITS OWN best-val
    # epoch (experts share no params, so this == training each one separately).
    best_joint_ang = [float("inf")] * J
    best_joint_score = [float("inf")] * J
    best_joint_state = [None] * J
    since_improve_j = [0] * J          # per-joint epochs since that joint's val improved
    converged = [False] * J            # per-joint convergence flags
    score_min_delta = args.min_delta
    if score_min_delta is None:
        score_min_delta = 1e-6 if args.early_stop_metric == "total_loss" else 1e-3
    print(f"[early-stop] metric={args.early_stop_metric} patience={args.patience} min_delta={score_min_delta:g}")
    if args.resume and args.resume_reset_best:
        print("[resume] reset per-joint best/convergence tracking for this new loss function")
    elif args.resume:
        best_path = args.resume.replace(".pt", "_best.pt")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu")
            saved_best = best_ckpt.get("per_joint_best_ang")
            saved_score = best_ckpt.get("per_joint_best_score")
            if saved_best is not None:
                best_joint_ang = [float(x) for x in saved_best]
            if saved_score is not None and best_ckpt.get("early_stop_metric", "orient_deg") == args.early_stop_metric:
                best_joint_score = [float(x) for x in saved_score]
            elif args.early_stop_metric == "orient_deg" and saved_best is not None:
                best_joint_score = list(best_joint_ang)
            best_model_cpu = MultiExpert(len(CANONICAL_IMUS), J, hidden_dim=args.hidden, nhead=args.nhead,
                                         num_layers=args.layers, target_dim=args.target_dim)
            best_model_cpu.load_state_dict(best_ckpt["model"])
            best_joint_state = [
                {k: v.detach().cpu().clone() for k, v in best_model_cpu.experts[j].state_dict().items()}
                for j in range(J)
            ]
            valid_best = [a for a in best_joint_ang if a < float("inf")]
            valid_score = [a for a in best_joint_score if a < float("inf")]
            print(f"[resume] restored per-joint best from {best_path}; "
                  f"macro_ang={sum(valid_best)/max(len(valid_best),1):.2f}deg "
                  f"macro_score={sum(valid_score)/max(len(valid_score),1):.6f}")
        else:
            print(f"[resume] no per-joint best checkpoint found at {best_path}; starting best state fresh")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # converged experts drop out of training compute (their best is already saved)
        active = [j for j in range(J) if not converged[j]] if args.freeze_converged else None
        if active is not None:
            if not active:                       # all converged -> nothing left to train
                print(f"[early-stop] all {J} joints converged (no active experts) at epoch {epoch+1}")
                break
            for j in range(J):
                model.experts[j].requires_grad_(not converged[j])
        t0 = time.time(); tot = 0.0; otot = 0.0; xtot = 0.0; n = 0
        for batch_idx, batch in enumerate(dl):
            loss, o_raw, x_raw = run_batch(batch, True, active=active, epoch=epoch, batch_idx=batch_idx)
            otot += o_raw; xtot += x_raw
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            tot += float(loss.detach()); n += 1; step += 1
            if args.log_every and (batch_idx + 1) % args.log_every == 0:
                lo, lx = lossw["wo"], lossw["wx"]
                avg_loss = tot / max(n, 1)
                avg_o = otot / max(n, 1)
                avg_x = xtot / max(n, 1)
                active_count = len(active) if active is not None else J
                print(f"[train-progress] epoch={epoch+1}/{args.epochs} "
                      f"batch={batch_idx+1}/{len(dl)} active={active_count}/{J} "
                      f"loss={avg_loss:.6f} orient_raw={avg_o:.5f} xyz_raw={avg_x:.5f} "
                      f"ORIENT_w={lo:.2f} XYZ_w={lx:.2f} "
                      f"scaled_xyz:orient={lx*avg_x/max(lo*avg_o,1e-9):.2f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"[smoke] {step} steps ok, last loss={float(loss.detach()):.6f}, "
                      f"{(time.time()-t0)/n*1000:.0f} ms/step")
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "imus": CANONICAL_IMUS, "joints": CANONICAL_JOINTS}, args.save)
                print(f"[save] {args.save}")
                return
        # validation (augment=False -> all present sensors), per joint
        vper, vper_cm, voraw_j, vxraw_j, vloss, vo, vx = eval_per_joint(vdl)
        lo, lx = lossw["wo"], lossw["wx"]    # current (possibly adaptive) loss weights
        vtotal_j = lo * voraw_j + lx * vxraw_j

        valid_j = ~torch.isnan(vper)
        macro = float(vper[valid_j].mean()) if valid_j.any() else float("nan")     # ORIENT err, deg
        pooled = float(vper[valid_j].median()) if valid_j.any() else float("nan")
        cm_valid = ~torch.isnan(vper_cm)
        macro_cm = float(vper_cm[cm_valid].mean()) if cm_valid.any() else float("nan")  # XYZ err, cm
        total_valid = ~torch.isnan(vtotal_j)
        macro_total = float(vtotal_j[total_valid].mean()) if total_valid.any() else float("nan")
        score_vec = vtotal_j if args.early_stop_metric == "total_loss" else vper
        valid_score = ~torch.isnan(score_vec)
        macro_score = float(score_vec[valid_score].mean()) if valid_score.any() else float("nan")
        score_unit = "loss" if args.early_stop_metric == "total_loss" else "deg"

        worst = sorted([(float(vper[j]), CANONICAL_JOINTS[j]) for j in range(J) if valid_j[j]],
                       reverse=True)[:3]
        worst_str = ", ".join(f"{nm}={a:.1f}deg" for a, nm in worst)
        tro, trx = otot / max(n, 1), xtot / max(n, 1)
        # Per epoch: the two interpretable errors (orientation in DEG, xyz motion in CM),
        # then for each the raw loss, its weight, and the scaled loss (raw*weight = what
        # the optimizer balances). val_total is the fixed/adaptive weighted sum used
        # when --early-stop-metric=total_loss.
        print(f"epoch={epoch+1}/{args.epochs} "
              f"val_orient={macro:.2f}deg val_xyz={macro_cm:.3f}cm median={pooled:.2f}deg "
              f"val_total={macro_total:.6f} val_loss={vloss:.6f} score={macro_score:.6f}{score_unit} | "
              f"ORIENT raw[t/v]={tro:.5f}/{vo:.5f} w={lo:.2f} scaled[t/v]={lo*tro:.5f}/{lo*vo:.5f} | "
              f"XYZ raw[t/v]={trx:.5f}/{vx:.5f} w={lx:.2f} scaled[t/v]={lx*trx:.5f}/{lx*vx:.5f} | "
              f"scaled xyz:orient[t/v]={lx*trx/max(lo*tro,1e-9):.2f}/{lx*vx/max(lo*vo,1e-9):.2f} "
              f"worst[{worst_str}] lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch + 1,
                    "imus": CANONICAL_IMUS, "joints": CANONICAL_JOINTS}, args.save)
        # PER-JOINT convergence: snapshot each expert at ITS OWN best-val epoch and
        # track each joint's own "epochs since improvement". A joint is converged when
        # its own patience runs out. The score can be human orientation deg or the
        # weighted total raw loss, depending on --early-stop-metric.
        improved = 0
        for j in range(J):
            if not valid_score[j]:
                converged[j] = True          # no GT for this joint -> nothing to train
                continue
            score = float(score_vec[j])
            if score < best_joint_score[j] - score_min_delta:
                best_joint_score[j] = score
                best_joint_ang[j] = float(vper[j]) if valid_j[j] else float("inf")
                best_joint_state[j] = {k: v.detach().cpu().clone()
                                       for k, v in model.experts[j].state_dict().items()}
                since_improve_j[j] = 0
                converged[j] = False
                improved += 1
            else:
                since_improve_j[j] += 1
            if args.patience and since_improve_j[j] >= args.patience:
                converged[j] = True
        # assemble & save the per-joint-best model (each expert at its own optimum)
        best_model = MultiExpert(len(CANONICAL_IMUS), J, hidden_dim=args.hidden, nhead=args.nhead,
                                 num_layers=args.layers, target_dim=args.target_dim)
        cur = model.state_dict()
        best_model.load_state_dict(cur)   # default to current for any joint not yet snapshotted
        for j in range(J):
            if best_joint_state[j] is not None:
                best_model.experts[j].load_state_dict(best_joint_state[j])
        torch.save({"model": best_model.state_dict(), "args": vars(args), "epoch": epoch + 1,
                    "imus": CANONICAL_IMUS, "joints": CANONICAL_JOINTS,
                    "per_joint_best_ang": best_joint_ang,
                    "per_joint_best_score": best_joint_score,
                    "early_stop_metric": args.early_stop_metric,
                    "orient_weight": float(lo), "xyz_weight": float(lx)},
                   args.save.replace(".pt", "_best.pt"))
        n_conv = sum(converged)
        still = [CANONICAL_JOINTS[j] for j in range(J) if not converged[j]]
        valid_best = [a for a in best_joint_ang if a < float("inf")]
        valid_best_score = [a for a in best_joint_score if a < float("inf")]
        if args.per_joint_log:
            row = {
                "epoch": epoch + 1,
                "early_stop_metric": args.early_stop_metric,
                "score_min_delta": float(score_min_delta),
                "val_orient_macro_deg": macro,
                "val_xyz_macro_cm": macro_cm,
                "val_total_loss_macro": macro_total,
                "score_macro": macro_score,
                "orient_weight": float(lo),
                "xyz_weight": float(lx),
                "train_total_loss": float(lo * tro + lx * trx),
                "train_orient_raw": float(tro),
                "train_xyz_raw": float(trx),
                "weighted_orient": float(lo * tro),
                "weighted_xyz": float(lx * trx),
                "val_orient_raw": float(vo),
                "val_xyz_raw": float(vx),
                "lr": float(sched.get_last_lr()[0]),
                "converged_count": int(n_conv),
                "joints": {
                    CANONICAL_JOINTS[j]: {
                        "orient_deg": None if torch.isnan(vper[j]) else float(vper[j]),
                        "xyz_cm": None if torch.isnan(vper_cm[j]) else float(vper_cm[j]),
                        "orient_raw": None if torch.isnan(voraw_j[j]) else float(voraw_j[j]),
                        "xyz_raw": None if torch.isnan(vxraw_j[j]) else float(vxraw_j[j]),
                        "total_loss": None if torch.isnan(vtotal_j[j]) else float(vtotal_j[j]),
                        "score": None if torch.isnan(score_vec[j]) else float(score_vec[j]),
                        "best_score": None if best_joint_score[j] == float("inf") else float(best_joint_score[j]),
                        "best_orient_deg": None if best_joint_ang[j] == float("inf") else float(best_joint_ang[j]),
                        "epochs_since_improve": int(since_improve_j[j]),
                        "converged": bool(converged[j]),
                    }
                    for j in range(J)
                },
            }
            os.makedirs(os.path.dirname(args.per_joint_log), exist_ok=True)
            with open(args.per_joint_log, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  [best] {improved} improved; converged {n_conv}/{J}; "
              f"per-joint-best macro={sum(valid_best)/max(len(valid_best),1):.2f}deg; "
              f"best_score_macro={sum(valid_best_score)/max(len(valid_best_score),1):.6f}; "
              f"still training: {still[:8]}{'...' if len(still) > 8 else ''}")
        if args.patience and all(converged):
            print(f"[early-stop] ALL {J} joints individually converged at epoch {epoch+1}")
            break
    valid_best = [a for a in best_joint_ang if a < float("inf")]
    valid_best_score = [a for a in best_joint_score if a < float("inf")]
    print(f"[done] saved {args.save}; per-joint-best "
          f"macro={sum(valid_best)/max(len(valid_best),1):.2f}deg "
          f"score_macro={sum(valid_best_score)/max(len(valid_best_score),1):.6f}")
    # ---- single, final held-out test on the assembled per-joint-best model ----
    if test_dl is not None:
        best_model = MultiExpert(len(CANONICAL_IMUS), J, hidden_dim=args.hidden, nhead=args.nhead,
                                 num_layers=args.layers, target_dim=args.target_dim).to(device)
        best_model.load_state_dict(torch.load(args.save.replace(".pt", "_best.pt"),
                                              map_location=device)["model"])
        saved_model = model
        model = best_model            # run_batch closes over `model`
        tper, tper_cm, toraw_j, txraw_j, _, t_o, t_x = eval_per_joint(test_dl)
        model = saved_model
        tv = ~torch.isnan(tper)
        cmv = ~torch.isnan(tper_cm)
        lo, lx = lossw["wo"], lossw["wx"]
        ttotal_j = lo * toraw_j + lx * txraw_j
        ttotal_valid = ~torch.isnan(ttotal_j)
        print(f"[test] FINAL held-out on per-joint-best ckpt: "
              f"orient={float(tper[tv].mean()):.2f}deg (median={float(tper[tv].median()):.2f}deg) "
              f"xyz={float(tper_cm[cmv].mean()):.3f}cm total_loss={float(ttotal_j[ttotal_valid].mean()):.6f} | "
              f"ORIENT raw={t_o:.5f} w={lo:.2f} scaled={lo*t_o:.5f} | "
              f"XYZ raw={t_x:.5f} w={lx:.2f} scaled={lx*t_x:.5f}")
        for j in range(J):
            if tv[j]:
                cm_j = f"{float(tper_cm[j]):.3f}cm" if cmv[j] else "n/a"
                total_j = "n/a" if torch.isnan(ttotal_j[j]) else f"{float(ttotal_j[j]):.6f}"
                print(f"    {CANONICAL_JOINTS[j]:>18s}  test_orient={float(tper[j]):.2f}deg "
                      f"test_xyz={cm_j} test_total={total_j} "
                      f"(val_best={best_joint_ang[j]:.2f}deg score={best_joint_score[j]:.6f})")


if __name__ == "__main__":
    main()
