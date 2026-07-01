#!/usr/bin/env python3
"""Baseline: train per-joint experts DIRECTLY from raw IMU data (no shared encoder).

Each expert is a 4-layer 64-dim Transformer:
  - Layer 0-1: loaded from pretrained prior (student_kl), LoRA fine-tuned
  - Layer 2-3: randomly initialized, fully trained
  - Input: 15 IMU tokens projected to d=64
  - Output: 1 joint's 9D prediction (6D orient + 3D motion delta)

Loss: orient MSE + motion delta L1, fixed weights (same as Stage 1 & 2).
Early stopping with patience=15.

Usage:
  python train_direct_expert.py --train-all-experts --device cuda
  python train_direct_expert.py --target-joint left_knee --device cuda
"""
import argparse
import math
import os
import re
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from masked_dataset import (
    MaskedMotionDataset,
    apply_visible_imu_sampling,
    build_fixed_visible_sets,
    collate,
)
from rotation_utils import rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from shared_encoder_model import SinusoidalPE, _encoder_layer, load_student_layers
from wholebody_model import LoRALinear
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
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
    b, t = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t)
    v = (steps < lengths.view(b, 1)).to(ang.dtype)
    return (ang * v).sum() / v.sum().clamp_min(1.0)


def xyz_cm_single(pred_xyz, tgt_xyz, lengths):
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


class DirectJointExpert(nn.Module):
    """Per-joint expert: raw IMU -> single joint prediction (no shared encoder).

    4-layer Transformer:
      - Layer 0-1: from pretrained prior, frozen + LoRA on linear1/linear2,
                   LayerNorms trainable
      - Layer 2-3: randomly initialized, fully trainable
    Input: 15 IMU tokens -> project to d -> temporal Transformer -> head -> 9D
    """

    def __init__(self, prior_path: str, d: int = 64, nhead: int = 4,
                 ff: int = 128, lora_r: int = 8,
                 target_dim: int = 9, dropout: float = 0.1,
                 n_imus: int = len(CANONICAL_IMUS), imu_dim: int = 12,
                 max_len: int = 1024):
        super().__init__()
        self.d = d
        self.target_dim = target_dim
        self.n_imus = n_imus

        self.imu_proj = nn.Linear(imu_dim, d)
        self.imu_embed = nn.Embedding(n_imus, d)
        self.pe = SinusoidalPE(d, max_len)

        # Layer 0-1: pretrained prior + LoRA
        prior_layers = load_student_layers(prior_path, d=d, nhead=nhead,
                                           ff=ff, num_layers=2)
        for layer in prior_layers:
            for p in layer.parameters():
                p.requires_grad_(False)
            layer.linear1 = LoRALinear(layer.linear1, r=lora_r)
            layer.linear2 = LoRALinear(layer.linear2, r=lora_r)
            for p in layer.norm1.parameters():
                p.requires_grad_(True)
            for p in layer.norm2.parameters():
                p.requires_grad_(True)

        # Layer 2-3: fully trainable
        fresh_layers = nn.ModuleList([
            _encoder_layer(d, nhead, ff, dropout) for _ in range(2)
        ])

        self.layers = nn.ModuleList(list(prior_layers) + list(fresh_layers))

        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d, target_dim),
        )

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                lengths=None) -> torch.Tensor:
        """imu [B,T,S,12], imu_mask [B,S] -> [B,T,target_dim]"""
        b, t, s, _ = imu.shape

        # project IMU tokens: [B,T,15,d]
        tok = self.imu_proj(imu) + self.imu_embed(
            torch.arange(s, device=imu.device),
        ).view(1, 1, s, self.d)

        # mean-pool visible IMU tokens per frame -> [B,T,d]
        vis = imu_mask.bool().float().view(b, 1, s, 1)
        x = (tok * vis).sum(dim=2) / vis.sum(dim=2).clamp_min(1.0)

        x = self.pe(x)

        # temporal padding mask
        pad = None
        if lengths is not None:
            pad = (torch.arange(t, device=x.device).view(1, t)
                   >= lengths.view(-1, 1))

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)

        return self.head(x)


def train_one_expert(joint_idx: int, args, device,
                     train_ds, val_ds, eval_visible_sets,
                     orient_slice, motion_delta_slice):
    joint_name = CANONICAL_JOINTS[joint_idx]
    print(f"\n{'='*60}")
    print(f"[direct-expert] Training joint {joint_idx:02d}: {joint_name}")
    print(f"{'='*60}")

    expert = DirectJointExpert(
        prior_path=args.prior, d=args.hidden, nhead=args.nhead, ff=args.ff,
        lora_r=args.lora_r, target_dim=args.target_dim, dropout=0.1,
    ).to(device)
    n_total = sum(p.numel() for p in expert.parameters())
    n_train = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    print(f"[direct-expert] {joint_name}: total={n_total/1e3:.1f}k "
          f"trainable={n_train/1e3:.1f}k (LoRA r={args.lora_r})")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, collate_fn=collate,
                     pin_memory=True)

    opt = torch.optim.AdamW(
        [p for p in expert.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
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

    def run_batch(batch, train, epoch=0, batch_idx=0):
        if train:
            batch = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=batch_idx, seed=args.seed,
                min_k=args.min_k, max_k=args.max_k, split="train",
            )
        imu = batch["imu"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        imu_mask = batch["imu_mask"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        if not mask[:, joint_idx].any():
            return None

        if train and (args.acc_noise > 0 or args.ori_noise_deg > 0):
            imu = add_imu_noise(imu, imu_mask, args.acc_noise, args.ori_noise_deg)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = expert(imu, imu_mask, lengths)
            target_j = target[:, :, joint_idx, :]
            mask_j = mask[:, joint_idx]

            oa, ob = orient_slice
            valid = mask_j.to(pred.dtype).view(-1, 1, 1)
            o_raw = ((pred[..., oa:ob] - target_j[..., oa:ob]).pow(2) * valid).sum()
            o_raw = o_raw / valid.sum().clamp_min(1.0) / (ob - oa)
            xa, xb = motion_delta_slice
            x_raw = ((pred[..., xa:xb] - target_j[..., xa:xb]).abs() * valid).sum()
            x_raw = x_raw / valid.sum().clamp_min(1.0) / (xb - xa)

            loss = args.lambda_orientation * o_raw + args.lambda_motion_delta * x_raw

        if train:
            return loss, float(o_raw.detach()), float(x_raw.detach())

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

        pbar = tqdm(dl, desc=f"[{joint_name}] Epoch {epoch+1}/{args.epochs}",
                    unit="batch", leave=False)
        for batch_idx, batch in enumerate(pbar):
            result = run_batch(batch, True, epoch=epoch, batch_idx=batch_idx)
            if result is None:
                continue
            loss, o_raw, x_raw = result
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in expert.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            otot += o_raw
            xtot += x_raw
            n += 1
            step += 1
            if n > 0:
                pbar.set_postfix(loss=f"{tot/n:.4f}", refresh=False)

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

        expert.eval()
        vdeg_sum = vcm_sum = 0.0
        vn = 0
        with torch.no_grad():
            for batch in tqdm(vdl, desc=f"[{joint_name}] Val", unit="batch", leave=False):
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
    ap.add_argument("--prior", default="pretrained/student_kl_18to21_best_64.pth",
                    help="pretrained student_kl checkpoint for layers 0-1")
    ap.add_argument("--target-joint", default="",
                    help="joint name or index to train")
    ap.add_argument("--train-all-experts", action="store_true",
                    help="train all 28 joint experts sequentially")
    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    # model
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=8)
    # target
    ap.add_argument("--target", default="joint_orient_r6d,joint_delta")
    ap.add_argument("--mask-key", default="auto")
    ap.add_argument("--target-dim", type=int, default=0)
    # loss (fixed weights)
    ap.add_argument("--lambda-orientation", type=float, default=1.0)
    ap.add_argument("--lambda-motion-delta", type=float, default=1.0)
    # data
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--holdout-kw", default="dip,totalcapture,imuposer")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--preload", action="store_true")
    # noise
    ap.add_argument("--acc-noise", type=float, default=0.0)
    ap.add_argument("--ori-noise-deg", type=float, default=0.0)
    # runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--max-steps", type=int, default=0)
    # save
    ap.add_argument("--save-dir", default="weights/direct_experts")
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
            preload=args.preload,
        )
        n = len(train_full)
        n_val = max(1, int(n * args.val_frac))
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        val_ds = Subset(train_full, idx[:n_val])
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
        for ji in range(len(CANONICAL_JOINTS)):
            deg = train_one_expert(
                ji, args, device, train_ds, val_ds,
                eval_visible_sets, orient_slice, motion_delta_slice,
            )
            results[CANONICAL_JOINTS[ji]] = deg

        print(f"\n{'='*60}")
        print("[direct-expert] All 28 experts done")
        print(f"{'='*60}")
        for jname, deg in results.items():
            print(f"  {jname:20s}  val_orient={deg:.2f}deg")
        avg = sum(v for v in results.values() if not math.isinf(v))
        cnt = sum(1 for v in results.values() if not math.isinf(v))
        if cnt:
            print(f"  {'AVERAGE':20s}  val_orient={avg/cnt:.2f}deg")
    else:
        ji = resolve_joint(args.target_joint)
        train_one_expert(
            ji, args, device, train_ds, val_ds,
            eval_visible_sets, orient_slice, motion_delta_slice,
        )


if __name__ == "__main__":
    main()
