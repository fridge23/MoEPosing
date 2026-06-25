#!/usr/bin/env python3
"""Stage 1: pretrain the shared sparse-IMU encoder + full-body decoder.

Input:  2-5 visible IMUs (dataset-aware random masking per batch).
Target: full-body 28-joint orientation (6D) + motion delta (3D) = 9D per joint.

The first 2 encoder layers are initialized from the Sapiens-distilled prior
(student_kl_18to21_best_64.pth); layers 2-3 are fresh.  All layers are
fine-tuned by default (use --freeze-pretrained to freeze layers 0-1).

After training, the ENCODER checkpoint is the main artifact for Stage 2
(lightweight joint experts).
"""
import argparse
import json
import math
import os
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
from rotation_utils import normalize_rotation_6d, rotation_6d_to_matrix
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from shared_encoder_model import (
    SharedEncoder,
    PretrainDecoder,
    SharedEncoderPretrainModel,
)
from target_spec import slice_for, split_keys, target_dim

import random


def r6d_to_mat(r6d: torch.Tensor) -> torch.Tensor:
    return rotation_6d_to_matrix(r6d)


def geodesic_deg(pred6, tgt6, mask, lengths):
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
    b, t, j = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (ang * v).sum() / v.sum().clamp_min(1.0)


def geodesic_sums_per_joint(pred6, tgt6, mask, lengths):
    Rp, Rt = r6d_to_mat(pred6), r6d_to_mat(tgt6)
    rel = Rp.transpose(-1, -2) @ Rt
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)
    b, t, j = ang.shape
    steps = torch.arange(t, device=ang.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (ang * v).sum(dim=(0, 1)), v.sum(dim=(0, 1))


def xyz_dist_sums_per_joint(pred_xyz, tgt_xyz, mask, lengths):
    d = (pred_xyz - tgt_xyz).norm(dim=-1) * 100.0
    b, t, j = d.shape
    steps = torch.arange(t, device=d.device).view(1, t, 1)
    v = mask.view(b, 1, j) * (steps < lengths.view(b, 1, 1))
    return (d * v).sum(dim=(0, 1)), v.sum(dim=(0, 1))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data60hz")
    ap.add_argument("--prior", default="pretrained/student_kl_18to21_best_64.pth",
                    help="pretrained 2-layer encoder checkpoint (Sapiens-distilled student_kl)")
    ap.add_argument("--pretrained-layers", type=int, default=2,
                    help="number of encoder layers to initialize from --prior")
    ap.add_argument("--freeze-pretrained", action="store_true",
                    help="freeze the pretrained encoder layers during training")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20,
                    help="early stopping patience (0=off)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500, help="linear LR warmup steps")
    # encoder config
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--spatial-layers", type=int, default=2,
                    help="per-frame spatial self-attention layers before temporal")
    ap.add_argument("--mask-position", default="before", choices=["before", "after"],
                    help="'before'=mask tokens enter encoder; 'after'=encoder sees only visible, masks added after")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff", type=int, default=128)
    # decoder config
    ap.add_argument("--decoder-layers", type=int, default=2)
    # target
    ap.add_argument("--target", default="joint_orient_r6d,joint_delta")
    ap.add_argument("--mask-key", default="auto")
    ap.add_argument("--target-dim", type=int, default=0)
    # loss weights
    ap.add_argument("--lambda-orientation", type=float, default=1.0)
    ap.add_argument("--lambda-motion-delta", type=float, default=1.0)
    ap.add_argument("--loss-balance", default="fixed", choices=["fixed", "adaptive"])
    ap.add_argument("--xyz-weight", type=float, default=2.0)
    ap.add_argument("--ema-momentum", type=float, default=0.98)
    # data
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--manifest", default="",
                    help="frozen seq-level split manifest; '' = random window split")
    ap.add_argument("--holdout-kw", default="dip,totalcapture,imuposer")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--preload", action="store_true")
    # noise
    ap.add_argument("--acc-noise", type=float, default=0.0)
    ap.add_argument("--ori-noise-deg", type=float, default=0.0)
    # training
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--per-joint-log", default="")
    # save
    ap.add_argument("--save-dir", default="weights/shared_encoder")
    ap.add_argument("--resume", default="", help="path to latest.pt to resume training from")
    args = ap.parse_args()

    if args.target_dim <= 0:
        args.target_dim = target_dim(args.target)
    args.target_keys = split_keys(args.target)
    orient_slice = slice_for(args.target_keys, "joint_orient_r6d")
    motion_delta_slice = slice_for(args.target_keys, "joint_delta")
    if orient_slice is None:
        raise ValueError("--target must include joint_orient_r6d")
    if motion_delta_slice is None:
        raise ValueError("--target must include joint_delta")
    holdout = [k.strip() for k in args.holdout_kw.split(",") if k.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    print(f"[env] device={device} cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # --- model ---
    encoder = SharedEncoder(
        d=args.hidden, nhead=args.nhead, ff=args.ff,
        num_layers=args.encoder_layers,
        num_spatial_layers=args.spatial_layers, dropout=0.1,
        mask_position=args.mask_position,
    )
    n_loaded = encoder.load_pretrained(
        args.prior, num_layers_to_load=args.pretrained_layers,
        freeze=args.freeze_pretrained,
    )
    print(f"[encoder] loaded {n_loaded} pretrained layers from {args.prior}"
          f"{' (frozen)' if args.freeze_pretrained else ' (fine-tuning)'}")

    decoder = PretrainDecoder(
        d=args.hidden, nhead=args.nhead, ff=args.ff,
        num_layers=args.decoder_layers, target_dim=args.target_dim,
    )
    model = SharedEncoderPretrainModel(encoder, decoder).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total={n_params/1e3:.1f}k trainable={n_train/1e3:.1f}k "
          f"encoder_layers={args.encoder_layers} decoder_layers={args.decoder_layers} "
          f"hidden={args.hidden} target_dim={args.target_dim}")

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
        test_ds = MaskedMotionDataset(
            args.data, augment=False, target_key=args.target,
            mask_key=args.mask_key, manifest=manifest, split="test",
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
        test_ds = MaskedMotionDataset(
            args.data, augment=False, target_key=args.target,
            mask_key=args.mask_key, include_only_kw=holdout,
            require_min_imus=True,
        )
        n = len(train_full)
        n_val = max(1, int(n * args.val_frac))
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        val_ds = Subset(eval_full, idx[:n_val])
        train_ds = Subset(train_full, idx[n_val:])

    # report dataset sources
    for ds_obj in [train_ds, val_ds]:
        base = ds_obj
        while hasattr(base, "dataset"):
            base = base.dataset
        if hasattr(base, "describe_sources"):
            for line in base.describe_sources():
                print(f"  [source] {line}")
            break

    eval_visible_sets = build_fixed_visible_sets(
        [val_ds, test_ds], seed=args.seed, min_k=args.min_k, max_k=args.max_k,
        split="val",
    )
    print(f"[visible-eval] fixed visible IMUs by source: {eval_visible_sets}")

    test_dl = None
    if len(test_ds):
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate, pin_memory=True)

    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"target={args.target_keys} target_dim={args.target_dim}")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, collate_fn=collate, pin_memory=True)

    # --- optimizer ---
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and use_amp)

    # adaptive loss balancing
    ema = {"o": None, "x": None}
    lossw = {"wo": args.lambda_orientation, "wx": args.lambda_motion_delta}

    def component_weights(o_raw, x_raw, train: bool):
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

    J = len(CANONICAL_JOINTS)

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

        if train and (args.acc_noise > 0 or args.ori_noise_deg > 0):
            imu = add_imu_noise(imu, imu_mask, args.acc_noise, args.ori_noise_deg)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(imu, imu_mask, lengths)
            oa, ob = orient_slice
            o_raw = orientation_6d_loss(
                pred[..., oa:ob], target[..., oa:ob], mask, lengths,
                per_joint=True,
            )
            xa, xb = motion_delta_slice
            x_raw = motion_delta_loss(
                pred[..., xa:xb], target[..., xa:xb], mask, lengths,
                per_joint=True,
            )
            wo, wx = component_weights(o_raw, x_raw, train)
            lossw["wo"], lossw["wx"] = wo, wx
            loss = wo * o_raw + wx * x_raw

        if train:
            return loss, float(o_raw.detach()), float(x_raw.detach())

        # validation: per-joint geodesic + xyz error
        a, b = orient_slice
        num, den = geodesic_sums_per_joint(
            pred[..., a:b].float(), target[..., a:b].float(), mask, lengths,
        )
        xa, xb = motion_delta_slice
        xnum, xden = xyz_dist_sums_per_joint(
            pred[..., xa:xb].float(), target[..., xa:xb].float(), mask, lengths,
        )
        return (loss, num, den, xnum, xden,
                float(o_raw.detach()), float(x_raw.detach()))

    os.makedirs(args.save_dir, exist_ok=True)
    encoder_path = os.path.join(args.save_dir, "best_encoder.pt")
    decoder_path = os.path.join(args.save_dir, "best_decoder.pt")
    latest_path = os.path.join(args.save_dir, "latest.pt")

    def eval_epoch(loader):
        model.eval()
        num = torch.zeros(J, device=device)
        den = torch.zeros(J, device=device)
        xnum = torch.zeros(J, device=device)
        xden = torch.zeros(J, device=device)
        ltot = otot = xtot = 0.0
        ln = 0
        with torch.no_grad():
            for batch in loader:
                loss, n_, d_, xn_, xd_, o_raw, x_raw = run_batch(batch, False)
                ltot += float(loss)
                otot += o_raw
                xtot += x_raw
                ln += 1
                num += n_
                den += d_
                xnum += xn_
                xden += xd_
        per_joint_deg = (num / den.clamp_min(1.0)).cpu()
        per_joint_deg[den.cpu() == 0] = float("nan")
        per_joint_cm = (xnum / xden.clamp_min(1.0)).cpu()
        per_joint_cm[xden.cpu() == 0] = float("nan")
        return (per_joint_deg, per_joint_cm, ltot / max(ln, 1),
                otot / max(ln, 1), xtot / max(ln, 1))

    best_val = float("inf")
    since_improve = 0
    step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        if best_val == float("inf"):
            best_val = float("inf")
        step = start_epoch * max(1, len(dl))
        for _ in range(step):
            sched.step()
        print(f"[resume] from {args.resume} epoch={start_epoch} best_val_loss={best_val:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        tot = otot = xtot = 0.0
        n = 0
        for batch_idx, batch in enumerate(dl):
            loss, o_raw, x_raw = run_batch(batch, True, epoch=epoch,
                                           batch_idx=batch_idx)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            tot += float(loss.detach())
            otot += o_raw
            xtot += x_raw
            n += 1
            step += 1

            if args.log_every and (batch_idx + 1) % args.log_every == 0:
                lo, lx = lossw["wo"], lossw["wx"]
                print(f"[train] epoch={epoch+1}/{args.epochs} "
                      f"batch={batch_idx+1}/{len(dl)} "
                      f"loss={tot/n:.6f} orient={otot/n:.5f} xyz={xtot/n:.5f} "
                      f"w_o={lo:.2f} w_x={lx:.2f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

            if args.max_steps and step >= args.max_steps:
                print(f"[smoke] {step} steps ok, loss={float(loss.detach()):.6f} "
                      f"{(time.time()-t0)/n*1000:.0f}ms/step")
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "args": vars(args),
                    "imus": CANONICAL_IMUS,
                    "joints": CANONICAL_JOINTS,
                }, latest_path)
                print(f"[save] {latest_path}")
                return

        # --- validation ---
        vdeg, vcm, vloss, vo, vx = eval_epoch(vdl)
        valid_j = ~torch.isnan(vdeg)
        macro_deg = float(vdeg[valid_j].mean()) if valid_j.any() else float("nan")
        cm_valid = ~torch.isnan(vcm)
        macro_cm = float(vcm[cm_valid].mean()) if cm_valid.any() else float("nan")
        lo, lx = lossw["wo"], lossw["wx"]
        tro, trx = otot / max(n, 1), xtot / max(n, 1)

        worst = sorted(
            [(float(vdeg[j]), CANONICAL_JOINTS[j]) for j in range(J) if valid_j[j]],
            reverse=True,
        )[:3]
        worst_str = ", ".join(f"{nm}={a:.1f}deg" for a, nm in worst)
        print(f"epoch={epoch+1}/{args.epochs} "
              f"val_orient={macro_deg:.2f}deg val_xyz={macro_cm:.3f}cm "
              f"val_loss={vloss:.6f} | "
              f"ORIENT raw[t/v]={tro:.5f}/{vo:.5f} w={lo:.2f} | "
              f"XYZ raw[t/v]={trx:.5f}/{vx:.5f} w={lx:.2f} | "
              f"worst[{worst_str}] lr={sched.get_last_lr()[0]:.2e} "
              f"{time.time()-t0:.1f}s")

        # save latest
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "args": vars(args),
            "epoch": epoch + 1,
            "val_orient_deg": macro_deg,
            "val_xyz_cm": macro_cm,
            "val_loss": vloss,
            "eval_visible_sets": eval_visible_sets,
            "imus": CANONICAL_IMUS,
            "joints": CANONICAL_JOINTS,
        }
        torch.save(ckpt, latest_path)

        # per-joint log
        if args.per_joint_log:
            row = {
                "epoch": epoch + 1,
                "val_orient_macro_deg": macro_deg,
                "val_xyz_macro_cm": macro_cm,
                "val_loss": vloss,
                "train_orient_raw": float(tro),
                "train_xyz_raw": float(trx),
                "orient_weight": float(lo),
                "xyz_weight": float(lx),
                "lr": float(sched.get_last_lr()[0]),
                "joints": {
                    CANONICAL_JOINTS[j]: {
                        "orient_deg": None if torch.isnan(vdeg[j]) else float(vdeg[j]),
                        "xyz_cm": None if torch.isnan(vcm[j]) else float(vcm[j]),
                    }
                    for j in range(J)
                },
            }
            os.makedirs(os.path.dirname(args.per_joint_log), exist_ok=True)
            with open(args.per_joint_log, "a") as f:
                f.write(json.dumps(row) + "\n")

        # best checkpoint (tracked by val_loss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            since_improve = 0
            torch.save({**ckpt, "best_val_loss": vloss}, encoder_path)
            torch.save({
                "decoder": decoder.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
            }, decoder_path)
            print(f"  [best] val_loss={vloss:.6f} orient={macro_deg:.2f}deg -> {encoder_path}")
        else:
            since_improve += 1

        if args.patience and since_improve >= args.patience:
            print(f"[early-stop] val_loss plateaued for {args.patience} epochs "
                  f"at epoch {epoch+1}")
            break

    print(f"[done] best val_loss={best_val:.6f} saved to {encoder_path}")

    # --- final test evaluation (per-dataset) ---
    if test_dl is not None:
        best_ckpt = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(best_ckpt["encoder"])
        if os.path.exists(decoder_path):
            dec_ckpt = torch.load(decoder_path, map_location=device)
            decoder.load_state_dict(dec_ckpt["decoder"])

        # overall test
        tdeg, tcm, _, to_, tx_ = eval_epoch(test_dl)
        tv = ~torch.isnan(tdeg)
        cmv = ~torch.isnan(tcm)
        print(f"[test] FINAL held-out on best encoder: "
              f"orient={float(tdeg[tv].mean()):.2f}deg "
              f"xyz={float(tcm[cmv].mean()):.3f}cm")
        for j in range(J):
            if tv[j]:
                cm_j = f"{float(tcm[j]):.3f}cm" if cmv[j] else "n/a"
                print(f"    {CANONICAL_JOINTS[j]:>18s}  "
                      f"orient={float(tdeg[j]):.2f}deg xyz={cm_j}")

        # per-dataset test evaluation
        from collections import defaultdict
        model.eval()
        ds_num = defaultdict(lambda: torch.zeros(J, device=device))
        ds_den = defaultdict(lambda: torch.zeros(J, device=device))
        ds_xnum = defaultdict(lambda: torch.zeros(J, device=device))
        ds_xden = defaultdict(lambda: torch.zeros(J, device=device))
        with torch.no_grad():
            for batch in test_dl:
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
                sources = batch.get("source", ["unknown"] * imu.shape[0])

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    pred = model(imu, imu_mask, lengths)

                unique_sources = sorted(set(sources))
                for src in unique_sources:
                    src_idx = [i for i, s in enumerate(sources) if s == src]
                    src_idx_t = torch.tensor(src_idx, device=device)
                    p = pred[src_idx_t].float()
                    t = target[src_idx_t].float()
                    m = mask[src_idx_t]
                    ln = lengths[src_idx_t]
                    a, b = orient_slice
                    n_, d_ = geodesic_sums_per_joint(p[..., a:b], t[..., a:b], m, ln)
                    xa, xb = motion_delta_slice
                    xn_, xd_ = xyz_dist_sums_per_joint(p[..., xa:xb], t[..., xa:xb], m, ln)
                    ds_num[src] += n_
                    ds_den[src] += d_
                    ds_xnum[src] += xn_
                    ds_xden[src] += xd_

        print(f"\n[test] Per-dataset results:")
        for src in sorted(ds_num.keys()):
            pj_deg = (ds_num[src] / ds_den[src].clamp_min(1.0)).cpu()
            pj_deg[ds_den[src].cpu() == 0] = float("nan")
            pj_cm = (ds_xnum[src] / ds_xden[src].clamp_min(1.0)).cpu()
            pj_cm[ds_xden[src].cpu() == 0] = float("nan")
            v = ~torch.isnan(pj_deg)
            cv = ~torch.isnan(pj_cm)
            macro_deg = float(pj_deg[v].mean()) if v.any() else float("nan")
            macro_cm = float(pj_cm[cv].mean()) if cv.any() else float("nan")
            print(f"  [{src}] orient={macro_deg:.2f}deg xyz={macro_cm:.3f}cm")
            for j in range(J):
                if v[j]:
                    cm_j = f"{float(pj_cm[j]):.3f}cm" if cv[j] else "n/a"
                    print(f"      {CANONICAL_JOINTS[j]:>18s}  "
                          f"orient={float(pj_deg[j]):.2f}deg xyz={cm_j}")


if __name__ == "__main__":
    main()
