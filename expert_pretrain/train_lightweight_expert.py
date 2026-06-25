#!/usr/bin/env python3
"""Stage 2: train lightweight per-joint experts on frozen SharedEncoder output.

Each expert is a 2-layer temporal Transformer + head that reads its own joint's
slice from the shared encoder's [B,T,28,d] representation and predicts that
joint's 9D target (6D orientation + 3D motion delta).

Usage:
  # single expert
  python train_lightweight_expert.py --target-joint left_wrist --device cuda

  # all 28 experts sequentially
  python train_lightweight_expert.py --train-all-experts --device cuda

  # parallel on multiple GPUs (run each in its own shell)
  CUDA_VISIBLE_DEVICES=0 python train_lightweight_expert.py --target-joint left_wrist
  CUDA_VISIBLE_DEVICES=1 python train_lightweight_expert.py --target-joint right_wrist
"""
import argparse
import json
import math
import os
import re
import time

import torch
from torch.utils.data import DataLoader, Subset

from masked_dataset import (
    MaskedMotionDataset,
    apply_visible_imu_sampling,
    build_fixed_visible_sets,
    collate,
)
from multiexpert_model import orientation_6d_loss, motion_delta_loss
from rotation_utils import rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from shared_encoder_model import (
    SharedEncoder,
    LightweightJointExpert,
    EXPERT_RAW_OUTPUT_DIM,
)
from target_spec import slice_for, split_keys, target_dim

import random


JOINT_ALIASES = {
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_knee": "LeftLowerLeg", "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_hip": "LeftUpperLeg", "right_hip": "RightUpperLeg",
    "left_shoulder": "LeftShoulder", "right_shoulder": "RightShoulder",
    "pelvis": "Pelvis", "head": "Head",
}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_joint(name_or_id) -> int:
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        idx = int(name_or_id)
        if 0 <= idx < len(CANONICAL_JOINTS):
            return idx
        raise ValueError(f"Joint index {idx} out of range")
    key = name_or_id.lower().strip().replace("-", "_")
    if key in JOINT_ALIASES:
        name = JOINT_ALIASES[key]
    else:
        lookup = {_norm_name(n): n for n in CANONICAL_JOINTS}
        normed = _norm_name(key)
        name = lookup.get(normed)
    if name is None:
        raise ValueError(f"Unknown joint '{name_or_id}'. Known: {CANONICAL_JOINTS}")
    return CANONICAL_JOINTS.index(name)


def r6d_to_mat(r6d):
    return rotation_6d_to_matrix(r6d)


def geodesic_deg_single(pred6, tgt6, lengths):
    """Geodesic angle (degrees) for a single joint: pred6/tgt6 [B,T,6]."""
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
    b, t = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t)
    v = (steps < lengths.view(b, 1)).to(ang.dtype)
    return (ang * v).sum() / v.sum().clamp_min(1.0)


def xyz_cm_single(pred_xyz, tgt_xyz, lengths):
    """Mean xyz distance (cm) for a single joint."""
    d = (pred_xyz - tgt_xyz).norm(dim=-1) * 100.0
    b, t = d.shape
    steps = torch.arange(t, device=d.device).view(1, t)
    v = (steps < lengths.view(b, 1)).to(d.dtype)
    return (d * v).sum() / v.sum().clamp_min(1.0)


def add_imu_noise(imu, imu_mask, acc_std, ori_deg):
    b, t, s, _ = imu.shape
    out = imu
    if ori_deg > 0:
        axis = torch.randn(b, s, 3, device=imu.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ang = torch.randn(b, s, 1, 1, device=imu.device) * (ori_deg * math.pi / 180.0)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(b, s, 3, 3)
        R = (torch.eye(3, device=imu.device).view(1, 1, 3, 3)
             + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K))
        ori = out[..., :9].view(b, t, s, 3, 3)
        ori = torch.einsum("bsij,btsjk->btsik", R, ori).reshape(b, t, s, 9)
        out = torch.cat([ori, out[..., 9:]], dim=-1)
    if acc_std > 0:
        m = imu_mask.to(imu.dtype).view(b, 1, s, 1)
        out = torch.cat([
            out[..., :9],
            out[..., 9:] + torch.randn(b, t, s, 3, device=imu.device) * acc_std * m,
        ], dim=-1)
    return out


def train_one_expert(joint_idx: int, args, encoder: SharedEncoder, device,
                     train_ds, val_ds, eval_visible_sets, orient_slice,
                     motion_delta_slice):
    """Train a single lightweight expert for the given joint."""
    joint_name = CANONICAL_JOINTS[joint_idx]
    print(f"\n{'='*60}")
    print(f"[expert] Training joint {joint_idx:02d}: {joint_name}")
    print(f"{'='*60}")

    expert = LightweightJointExpert(
        d=args.hidden, nhead=args.nhead, ff=args.ff,
        num_layers=args.expert_layers, target_dim=args.target_dim,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in expert.parameters())
    print(f"[expert] {joint_name}: {n_params/1e3:.1f}k params, "
          f"{args.expert_layers} layers, hidden={args.hidden}")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, collate_fn=collate, pin_memory=True)

    opt = torch.optim.AdamW(expert.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(dl))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps - args.warmup),
    )
    if args.warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=args.warmup,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warm, cos], milestones=[args.warmup],
        )
    else:
        sched = cos

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"

    # adaptive loss
    ema = {"o": None, "x": None}

    def component_weights(o_raw, x_raw, train):
        if args.loss_balance != "adaptive":
            return args.lambda_orientation, args.lambda_motion_delta
        ov = max(float(o_raw.detach()), 1e-8)
        xv = max(float(x_raw.detach()), 1e-8)
        if ema["o"] is None:
            ema["o"], ema["x"] = ov, xv
        wo = args.lambda_orientation
        wx = args.xyz_weight * ema["o"] / max(ema["x"], 1e-8)
        if train:
            mu = args.ema_momentum
            ema["o"] = mu * ema["o"] + (1 - mu) * ov
            ema["x"] = mu * ema["x"] + (1 - mu) * xv
        return wo, wx

    def run_batch(batch, train, epoch=0, batch_idx=0):
        if train:
            batch = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=batch_idx, seed=args.seed,
                min_k=args.min_k, max_k=args.max_k, split="train",
            )
        else:
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

        # skip batch if target joint is not available
        if not mask[:, joint_idx].any():
            return None

        if train and (args.acc_noise > 0 or args.ori_noise_deg > 0):
            imu = add_imu_noise(imu, imu_mask, args.acc_noise, args.ori_noise_deg)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                shared = encoder(imu, imu_mask, lengths)  # [B, T, 28, d]
            pred = expert(shared[:, :, joint_idx, :], lengths)  # [B, T, 9]
            target_j = target[:, :, joint_idx, :]  # [B, T, 9]
            mask_j = mask[:, joint_idx]  # [B]

            oa, ob = orient_slice
            valid = mask_j.to(pred.dtype).view(-1, 1, 1)
            o_raw = ((pred[..., oa:ob] - target_j[..., oa:ob]).pow(2) * valid).sum()
            o_raw = o_raw / valid.sum().clamp_min(1.0) / (ob - oa)
            xa, xb = motion_delta_slice
            x_raw = ((pred[..., xa:xb] - target_j[..., xa:xb]).abs() * valid).sum()
            x_raw = x_raw / valid.sum().clamp_min(1.0) / (xb - xa)

            wo, wx = component_weights(o_raw, x_raw, train)
            loss = wo * o_raw + wx * x_raw

        if train:
            return loss, float(o_raw.detach()), float(x_raw.detach())

        # validation: geodesic + xyz for this joint
        valid_mask = mask_j.bool()
        if valid_mask.any():
            a, b = orient_slice
            deg = geodesic_deg_single(
                pred[valid_mask, :, a:b].float(),
                target_j[valid_mask, :, a:b].float(),
                lengths[valid_mask],
            )
            xa, xb = motion_delta_slice
            cm = xyz_cm_single(
                pred[valid_mask, :, xa:xb].float(),
                target_j[valid_mask, :, xa:xb].float(),
                lengths[valid_mask],
            )
        else:
            deg = cm = torch.tensor(float("nan"), device=device)
        return loss, float(deg), float(cm), float(o_raw.detach()), float(x_raw.detach())

    save_dir = os.path.join(args.save_dir, joint_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")

    best_val_deg = float("inf")
    since_improve = 0
    step = 0

    for epoch in range(args.epochs):
        expert.train()
        t0 = time.time()
        tot = otot = xtot = 0.0
        n = 0

        for batch_idx, batch in enumerate(dl):
            result = run_batch(batch, True, epoch=epoch, batch_idx=batch_idx)
            if result is None:
                continue
            loss, o_raw, x_raw = result
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            otot += o_raw
            xtot += x_raw
            n += 1
            step += 1

            if args.max_steps and step >= args.max_steps:
                print(f"[smoke] {joint_name}: {step} steps ok, "
                      f"loss={float(loss.detach()):.6f}")
                torch.save({
                    "model": expert.state_dict(),
                    "joint_idx": joint_idx,
                    "joint_name": joint_name,
                    "args": vars(args),
                }, best_path)
                return best_val_deg

        if n == 0:
            print(f"[warn] {joint_name}: no valid batches at epoch {epoch+1}")
            continue

        # validation
        expert.eval()
        vdeg_sum = vcm_sum = 0.0
        vn = 0
        with torch.no_grad():
            for batch in vdl:
                result = run_batch(batch, False)
                if result is None:
                    continue
                _, deg, cm, _, _ = result
                if not math.isnan(deg):
                    vdeg_sum += deg
                    vcm_sum += cm
                    vn += 1

        val_deg = vdeg_sum / max(vn, 1)
        val_cm = vcm_sum / max(vn, 1)
        print(f"[{joint_name}] epoch={epoch+1}/{args.epochs} "
              f"train_loss={tot/n:.6f} val_orient={val_deg:.2f}deg "
              f"val_xyz={val_cm:.3f}cm lr={sched.get_last_lr()[0]:.2e} "
              f"{time.time()-t0:.1f}s")

        if val_deg < best_val_deg - 1e-3:
            best_val_deg = val_deg
            since_improve = 0
            torch.save({
                "model": expert.state_dict(),
                "joint_idx": joint_idx,
                "joint_name": joint_name,
                "epoch": epoch + 1,
                "val_orient_deg": val_deg,
                "val_xyz_cm": val_cm,
                "args": vars(args),
                "eval_visible_sets": eval_visible_sets,
                "imus": CANONICAL_IMUS,
                "joints": CANONICAL_JOINTS,
            }, best_path)
            print(f"  [best] {joint_name} val_orient={val_deg:.2f}deg -> {best_path}")
        else:
            since_improve += 1

        if args.patience and since_improve >= args.patience:
            print(f"[early-stop] {joint_name} plateaued for {args.patience} "
                  f"epochs at epoch {epoch+1}")
            break

    print(f"[done] {joint_name}: best val_orient={best_val_deg:.2f}deg "
          f"saved to {best_path}")
    return best_val_deg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data60hz")
    ap.add_argument("--encoder-checkpoint",
                    default="weights/shared_encoder/best_encoder.pt",
                    help="Stage 1 shared encoder checkpoint")
    ap.add_argument("--freeze-encoder", action="store_true", default=True,
                    help="freeze the shared encoder (default: frozen)")
    ap.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    ap.add_argument("--target-joint", default="",
                    help="joint name or index to train (e.g. 'left_wrist' or '24')")
    ap.add_argument("--train-all-experts", action="store_true",
                    help="train all 28 joint experts sequentially")
    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    # model
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--expert-layers", type=int, default=2)
    # target
    ap.add_argument("--target", default="joint_orient_r6d,joint_delta")
    ap.add_argument("--mask-key", default="auto")
    ap.add_argument("--target-dim", type=int, default=0)
    # loss
    ap.add_argument("--lambda-orientation", type=float, default=1.0)
    ap.add_argument("--lambda-motion-delta", type=float, default=1.0)
    ap.add_argument("--loss-balance", default="adaptive", choices=["fixed", "adaptive"])
    ap.add_argument("--xyz-weight", type=float, default=2.0)
    ap.add_argument("--ema-momentum", type=float, default=0.98)
    # data
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--holdout-kw", default="dip,totalcapture,imuposer")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--preload", action="store_true")
    # noise
    ap.add_argument("--acc-noise", type=float, default=0.0)
    ap.add_argument("--ori-noise-deg", type=float, default=0.0)
    # runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--max-steps", type=int, default=0)
    # save
    ap.add_argument("--save-dir",
                    default="weights/joint_experts_lightweight")
    args = ap.parse_args()

    if not args.target_joint and not args.train_all_experts:
        ap.error("Specify --target-joint or --train-all-experts")

    if args.target_dim <= 0:
        args.target_dim = target_dim(args.target)
    args.target_keys = split_keys(args.target)
    orient_slice = slice_for(args.target_keys, "joint_orient_r6d")
    motion_delta_slice = slice_for(args.target_keys, "joint_delta")
    if orient_slice is None or motion_delta_slice is None:
        raise ValueError("--target must include joint_orient_r6d and joint_delta")
    holdout = [k.strip() for k in args.holdout_kw.split(",") if k.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    print(f"[env] device={device} cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # --- load shared encoder ---
    ckpt = torch.load(args.encoder_checkpoint, map_location="cpu")
    enc_args = ckpt.get("args", {})
    encoder = SharedEncoder(
        d=int(enc_args.get("hidden", args.hidden)),
        nhead=int(enc_args.get("nhead", args.nhead)),
        ff=int(enc_args.get("ff", args.ff)),
        num_layers=int(enc_args.get("encoder_layers", 4)),
        num_spatial_layers=int(enc_args.get("spatial_layers", 2)),
        dropout=0.1,
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device).eval()
    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
    # override hidden/nhead/ff from encoder config
    args.hidden = encoder.d
    args.nhead = encoder.temporal_layers[0].self_attn.num_heads
    args.ff = encoder.temporal_layers[0].linear1.out_features
    print(f"[encoder] loaded from {args.encoder_checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val={ckpt.get('best_val_orient_deg', ckpt.get('val_orient_deg', '?'))}deg) "
          f"{'frozen' if args.freeze_encoder else 'fine-tuning'}")

    # --- data ---
    manifest = args.manifest or None
    if manifest:
        train_ds = MaskedMotionDataset(
            args.data, min_k=args.min_k, max_k=args.max_k, augment=False,
            target_key=args.target, mask_key=args.mask_key,
            manifest=manifest, split="train", require_min_imus=True,
            preload=args.preload,
        )
        val_ds = MaskedMotionDataset(
            args.data, augment=False, target_key=args.target,
            mask_key=args.mask_key, manifest=manifest, split="val",
            require_min_imus=True, preload=args.preload,
        )
    else:
        train_full = MaskedMotionDataset(
            args.data, min_k=args.min_k, max_k=args.max_k, augment=False,
            target_key=args.target, mask_key=args.mask_key,
            exclude_kw=holdout, require_min_imus=True,
        )
        eval_full = MaskedMotionDataset(
            args.data, augment=False, target_key=args.target,
            mask_key=args.mask_key, exclude_kw=holdout,
            require_min_imus=True,
        )
        n = len(train_full)
        n_val = max(1, int(n * args.val_frac))
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        val_ds = Subset(eval_full, idx[:n_val])
        train_ds = Subset(train_full, idx[n_val:])

    eval_visible_sets = build_fixed_visible_sets(
        [val_ds], seed=args.seed, min_k=args.min_k, max_k=args.max_k,
        split="val",
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"target={args.target_keys} target_dim={args.target_dim}")
    print(f"[visible-eval] {eval_visible_sets}")

    # --- train ---
    if args.train_all_experts:
        results = {}
        for j in range(len(CANONICAL_JOINTS)):
            val_deg = train_one_expert(
                j, args, encoder, device, train_ds, val_ds,
                eval_visible_sets, orient_slice, motion_delta_slice,
            )
            results[CANONICAL_JOINTS[j]] = val_deg

        print(f"\n{'='*60}")
        print(f"[summary] All 28 experts trained:")
        for name, deg in results.items():
            print(f"  {name:>18s}: {deg:.2f}deg")
        valid = [d for d in results.values() if d < float("inf")]
        if valid:
            print(f"  {'MACRO':>18s}: {sum(valid)/len(valid):.2f}deg")
    else:
        joint_idx = resolve_joint(args.target_joint)
        train_one_expert(
            joint_idx, args, encoder, device, train_ds, val_ds,
            eval_visible_sets, orient_slice, motion_delta_slice,
        )


if __name__ == "__main__":
    main()
