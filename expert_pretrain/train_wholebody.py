#!/usr/bin/env python3
"""ViPoser Stage-2: train the whole-body recovery poser (temporal, IMU + joints).

Phase I  (default):       known joints carry GROUND-TRUTH targets; the pose-noise
                          option applies to the r6d slice only.
Phase II (--experts CKPT): known joints carry the FROZEN experts' predictions, so
                          the poser adapts to its real upstream inputs.

Each batch: deterministic dataset-aware random 2-5 IMU subset (train) +
per-sequence random known-joint subset; supervise full-body 6D orientation and
motion-delta targets for ALL available joints + temporal smoothness; report
geodesic angle on the UNKNOWN orientation slice.
DIP/TotalCapture/
IMUPoser are the held-out test set; the rest splits 95/5 train/val.
"""
import argparse
import os
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from masked_dataset import MaskedMotionDataset, apply_visible_imu_sampling, build_fixed_visible_sets, collate
from multiexpert_model import (
    ExpertOutputAssembler,
    JointExpert,
    expert_target_loss,
    motion_delta_loss,
    orientation_6d_loss,
)
from wholebody_model import WholeBodyPoser, trainable_parameters
from train_multiexpert import (add_imu_noise, geodesic_deg, identity_baseline_deg,
                               xyz_dist_sums_per_joint)
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys, target_dim


def sample_known(avail: torch.Tensor, kmin: int, kmax: int, gen=None) -> torch.Tensor:
    """Per-SEQUENCE random kmin..kmax subset of available joints marked KNOWN
    (the experts a deployment would activate stay fixed across a window)."""
    b, j = avail.shape
    scores = torch.rand(b, j, generator=gen, device=avail.device).masked_fill(~avail, -1.0)
    k = torch.randint(kmin, kmax + 1, (b,), generator=gen, device=avail.device)
    k = torch.minimum(k, avail.sum(1).clamp(min=1))
    rank = scores.argsort(dim=1, descending=True)
    sel = torch.arange(j, device=avail.device).view(1, j) < k.view(b, 1)
    known = torch.zeros_like(avail)
    known.scatter_(1, rank, sel)
    return known & avail


JOINT_ALIASES = {
    "leftwrist": "LeftHand",
    "rightwrist": "RightHand",
    "leftknee": "LeftLowerLeg",
    "rightknee": "RightLowerLeg",
    "waist": "Pelvis",
    "root": "Pelvis",
}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def joint_index(name_or_id) -> int:
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        idx = int(name_or_id)
        if 0 <= idx < len(CANONICAL_JOINTS):
            return idx
        raise ValueError(f"joint id out of range: {idx}")
    lookup = {_norm_name(name): i for i, name in enumerate(CANONICAL_JOINTS)}
    key = _norm_name(name_or_id)
    key = _norm_name(JOINT_ALIASES.get(key, name_or_id))
    if key not in lookup:
        raise ValueError(f"Unknown joint '{name_or_id}'. Known joints: {CANONICAL_JOINTS}")
    return lookup[key]


def parse_selected_experts(text: str):
    if not text:
        return []
    return [joint_index(part.strip()) for part in text.split(",") if part.strip()]


def fixed_known_from_selection(avail: torch.Tensor, selected) -> torch.Tensor:
    known = torch.zeros_like(avail)
    if selected:
        known[:, torch.as_tensor(selected, device=avail.device, dtype=torch.long)] = True
    return known & avail


def _key_padding_mask(lengths: torch.Tensor, t: int, device) -> torch.Tensor:
    steps = torch.arange(t, device=device).view(1, t)
    return steps >= lengths.to(device).view(-1, 1)


class JointExpertBank(torch.nn.Module):
    """Run selected independently-trained joint experts and assemble dense tokens."""
    def __init__(self, experts, target_dim: int):
        super().__init__()
        if not experts:
            raise ValueError("JointExpertBank needs at least one expert")
        self.selected = sorted(experts)
        self.experts = torch.nn.ModuleDict({str(idx): experts[idx] for idx in self.selected})
        self.assembler = ExpertOutputAssembler(len(CANONICAL_JOINTS), target_dim)

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor, lengths=None):
        pad = _key_padding_mask(lengths, imu.shape[1], imu.device) if lengths is not None else None
        outs = [self.experts[str(idx)](imu, imu_mask, pad) for idx in self.selected]
        dense, valid = self.assembler(torch.stack(outs, dim=2), selected=self.selected)
        return dense, valid


def _checkpoint_joint_idx(path: Path, ckpt) -> int:
    if "joint_idx" in ckpt:
        return int(ckpt["joint_idx"])
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer joint index from checkpoint: {path}")


def load_joint_expert_bank(expert_dir: str, selected, target_dim_value: int, device) -> JointExpertBank:
    root = Path(expert_dir)
    if not root.exists():
        raise FileNotFoundError(f"joint expert directory does not exist: {expert_dir}")
    checkpoints = sorted(root.glob("*_best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no '*_best.pt' joint expert checkpoints found in {expert_dir}")

    by_joint = {}
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        idx = _checkpoint_joint_idx(path, ckpt)
        by_joint[idx] = (path, ckpt)
    selected = list(selected) if selected else sorted(by_joint)
    missing = [idx for idx in selected if idx not in by_joint]
    if missing:
        names = [CANONICAL_JOINTS[idx] if 0 <= idx < len(CANONICAL_JOINTS) else str(idx) for idx in missing]
        raise FileNotFoundError(f"missing selected joint expert checkpoints: {names}")

    experts = {}
    for idx in selected:
        path, ckpt = by_joint[idx]
        saved_args = ckpt.get("args") or {}
        expert_dim = int(saved_args.get("target_dim", target_dim_value))
        if expert_dim != target_dim_value:
            raise ValueError(f"{path} target_dim={expert_dim} does not match recovery target_dim={target_dim_value}")
        model = JointExpert(
            len(CANONICAL_IMUS),
            hidden_dim=int(saved_args.get("hidden", 128)),
            nhead=int(saved_args.get("nhead", 4)),
            num_layers=int(saved_args.get("layers", 4)),
            dropout=float(saved_args.get("dropout", 0.1)),
            target_dim=target_dim_value,
        )
        model.load_state_dict(ckpt["model"])
        experts[idx] = model
    bank = JointExpertBank(experts, target_dim_value).to(device).eval()
    for p in bank.parameters():
        p.requires_grad_(False)
    loaded = ", ".join(f"{idx:02d}:{CANONICAL_JOINTS[idx]}" for idx in bank.selected)
    print(f"[joint-experts] loaded selected experts from {expert_dir}: {loaded}")
    return bank


def smoothness(pred: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean squared discrete acceleration over valid frames (paper Eq. 13)."""
    if pred.shape[1] < 3:
        return pred.new_zeros(())
    acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    steps = torch.arange(acc.shape[1], device=pred.device).view(1, -1)
    v = (steps < (lengths.view(-1, 1) - 2)).to(pred.dtype).view(*acc.shape[:2], 1, 1)
    return (acc.pow(2) * v).sum() / v.sum().clamp_min(1.0) / acc.shape[-1] / acc.shape[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--prior", default="pretrained/student_kl_18to21_best_64.pth")
    ap.add_argument("--experts", default="", help="Phase II: MultiExpert ckpt; known joints "
                                                  "use its predictions instead of GT")
    ap.add_argument("--joint-expert-dir", default="", help="Phase II: directory of per-joint "
                    "'*_best.pt' checkpoints from train_joint_expert.py")
    ap.add_argument("--selected-experts", default="", help="comma-separated joint ids/names to run "
                    "from --joint-expert-dir or --experts; empty with --joint-expert-dir loads all "
                    "available per-joint checkpoints")
    ap.add_argument("--init", default="", help="warm-start poser weights (e.g. the Phase-I ckpt)")
    ap.add_argument("--epochs", type=int, default=100, help="max epochs (generous cap; stops early on convergence)")
    ap.add_argument("--patience", type=int, default=0, help="0=off; else stop when val_unk has not "
                    "improved for this many epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--train-prior", default="ln_lora", choices=["lora", "ln_lora", "full"])
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                    help="head adds a correction to the (filled) input joint stream")
    ap.add_argument("--warmup", type=int, default=500, help="linear LR warmup steps")
    ap.add_argument("--log-every", type=int, default=0, help="print running loss every N steps")
    ap.add_argument("--known-min", type=int, default=2)
    ap.add_argument("--known-max", type=int, default=8)
    ap.add_argument("--min-imus", type=int, default=2)
    ap.add_argument("--max-imus", type=int, default=5)
    ap.add_argument("--eval-imus", type=int, default=3, help="val/test: keep a FIXED k sensors "
                    "(deployment-matched, in the 2-5 train regime) instead of all present")
    ap.add_argument("--eval-known", type=int, default=4, help="val/test: a FIXED number of known "
                    "joints (the 'few activated experts' scenario), deterministic across epochs")
    ap.add_argument("--pose-noise", type=float, default=0.02, help="Phase-I sigma on known r6d")
    ap.add_argument("--acc-noise", type=float, default=0.2)
    ap.add_argument("--ori-noise-deg", type=float, default=8.0)
    ap.add_argument("--target", default="joint_orient_r6d,joint_delta",
                    help="comma-separated target tensors to concatenate per joint/frame")
    ap.add_argument("--mask-key", default="auto",
                    help="'auto' uses orient_mask for joint_orient_r6d and joint_mask for xyz targets")
    ap.add_argument("--target-dim", type=int, default=0,
                    help="0=derive from --target")
    ap.add_argument("--smooth-w", type=float, default=0.08,
                    help="weight on temporal-smoothness vs target-vector L1. Lowered from 0.5 after the L1 "
                         "was corrected to a per-element mean (~6x smaller), preserving the balance.")
    ap.add_argument("--lambda-full-pose", type=float, default=1.0)
    ap.add_argument("--lambda-full-motion-delta", type=float, default=1.0,
                    help="fixed xyz weight (only used when --loss-balance fixed)")
    ap.add_argument("--loss-balance", default="adaptive", choices=["fixed", "adaptive"],
                    help="adaptive: dynamically weight xyz so weighted_xyz ~= xyz_weight * weighted_orient")
    ap.add_argument("--xyz-weight", type=float, default=2.0,
                    help="adaptive target ratio: weighted xyz error / weighted orientation error")
    ap.add_argument("--ema-momentum", type=float, default=0.98)
    ap.add_argument("--sample-visible-per-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="deterministically sample a dataset-valid 2-5 visible IMU set every train batch")
    ap.add_argument("--eval-fixed-visible-set", action=argparse.BooleanOptionalAction, default=True,
                    help="use one fixed dataset-specific visible IMU set for validation/test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-kw", default="dip,totalcapture,imuposer")
    ap.add_argument("--manifest", default="data/splits.json",
                    help="frozen sequence-level split manifest (leak-free); '' falls back to random split")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False,
                    help="preload all split samples into RAM so epochs are compute-bound, not disk-bound")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save", default="weights/wholebody.pt")
    args = ap.parse_args()
    if args.target_dim <= 0:
        args.target_dim = target_dim(args.target)
    args.target_keys = split_keys(args.target)
    args.selected_expert_idxs = parse_selected_experts(args.selected_experts)
    if args.experts and args.joint_expert_dir:
        raise ValueError("Use either --experts or --joint-expert-dir, not both")
    orient_slice = slice_for(args.target_keys, "joint_orient_r6d")
    if orient_slice is None:
        raise ValueError("--target must include joint_orient_r6d so validation can report MPJRE")
    motion_delta_slice = slice_for(args.target_keys, "joint_delta")
    if motion_delta_slice is None:
        raise ValueError("--target must include joint_delta for full-body motion-delta recovery")
    holdout = [k.strip() for k in args.holdout_kw.split(",") if k.strip()]

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={dev} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # same split convention as the experts: holdout -> test; rest 95/5 train/val.
    # Train sparse-masks every batch in step(); val/test use deterministic sparse eval masks.
    manifest = args.manifest or None
    if manifest:
        train_ds = MaskedMotionDataset(args.data, min_k=args.min_imus, max_k=args.max_imus,
                                       augment=False, target_key=args.target,
                                       mask_key=args.mask_key, manifest=manifest, split="train",
                                       require_min_imus=True, preload=args.preload)
        val_ds = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                     mask_key=args.mask_key, manifest=manifest, split="val",
                                     require_min_imus=True, preload=args.preload)
        test_ds = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                      mask_key=args.mask_key, manifest=manifest, split="test",
                                      require_min_imus=True, preload=args.preload)
    else:
        train_full = MaskedMotionDataset(args.data, min_k=args.min_imus, max_k=args.max_imus,
                                         augment=False, target_key=args.target,
                                         mask_key=args.mask_key, exclude_kw=holdout,
                                         require_min_imus=True)
        eval_full = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                        mask_key=args.mask_key, exclude_kw=holdout,
                                        require_min_imus=True)
        test_ds = MaskedMotionDataset(args.data, augment=False, target_key=args.target,
                                      mask_key=args.mask_key, include_only_kw=holdout,
                                      require_min_imus=True)
        n = len(train_full); idx = list(range(n)); random.Random(0).shuffle(idx)
        nv = max(1, int(args.val_frac * n))
        train_ds, val_ds = Subset(train_full, idx[nv:]), Subset(eval_full, idx[:nv])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    eval_visible_sets = {}
    if args.eval_fixed_visible_set:
        eval_visible_sets = build_fixed_visible_sets(
            [val_ds, test_ds], seed=args.seed, min_k=args.eval_imus, max_k=args.eval_imus, split="val",
        )
        print(f"[visible-eval] fixed visible IMUs by source: {eval_visible_sets}")
    print(f"[data] split={'manifest(seq-level)' if manifest else 'random-window'} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"imus={len(CANONICAL_IMUS)} joints={len(CANONICAL_JOINTS)} "
          f"target={args.target_keys} target_dim={args.target_dim}")

    model = WholeBodyPoser(args.prior, lora_r=args.lora_r, train_prior=args.train_prior,
                           residual=args.residual, target_dim=args.target_dim).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        print(f"[init] warm-started from {args.init}")
    tr = trainable_parameters(model)
    phase = "II(joint-experts)" if args.joint_expert_dir else ("II(experts)" if args.experts else "I(GT)")
    print(f"[model] trainable={sum(p.numel() for p in tr)/1e3:.1f}k "
          f"frozen={sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e3:.1f}k "
          f"phase={phase}")

    experts = None
    joint_experts = None
    if args.experts:
        from multiexpert_model import MultiExpert
        ck = torch.load(args.experts, map_location="cpu")
        a = ck["args"]
        expert_dim = a.get("target_dim", target_dim(a.get("target", "joint_orient_r6d")))
        if expert_dim != args.target_dim:
            raise ValueError(f"Experts output dim {expert_dim} does not match recovery target_dim {args.target_dim}")
        experts = MultiExpert(len(CANONICAL_IMUS), len(CANONICAL_JOINTS), hidden_dim=a["hidden"],
                              nhead=a["nhead"], num_layers=a["layers"], target_dim=a["target_dim"])
        experts.load_state_dict(ck["model"])
        experts.to(dev).eval()
        for p in experts.parameters():
            p.requires_grad_(False)
        score = ck.get("score")
        if score is None and ck.get("per_joint_best_ang"):
            vals = [float(x) for x in ck["per_joint_best_ang"] if float(x) < float("inf")]
            score = sum(vals) / max(len(vals), 1)
        print(f"[experts] frozen, from {args.experts} (epoch {ck.get('epoch')}, "
              f"score {float(score) if score is not None else float('nan'):.2f}deg)")
    elif args.joint_expert_dir:
        joint_experts = load_joint_expert_bank(
            args.joint_expert_dir, args.selected_expert_idxs, args.target_dim, dev,
        )

    opt = torch.optim.AdamW(tr, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_dl))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - args.warmup))
    if args.warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=args.warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])
    else:
        sched = cos
    amp = args.amp == "bf16" and dev.type == "cuda"

    # adaptive loss balancing (xyz weighted dynamically to ~= xyz_weight * orient)
    ema = {"o": None, "x": None}
    lossw = {"wo": args.lambda_full_pose, "wx": args.lambda_full_motion_delta}

    def component_weights(o_raw, x_raw, train: bool):
        if args.loss_balance != "adaptive":
            return args.lambda_full_pose, args.lambda_full_motion_delta
        ov, xv = max(float(o_raw.detach()), 1e-8), max(float(x_raw.detach()), 1e-8)
        if ema["o"] is None:
            ema["o"], ema["x"] = ov, xv
        wo = args.lambda_full_pose
        wx = args.xyz_weight * ema["o"] / max(ema["x"], 1e-8)
        if train:
            mu = args.ema_momentum
            ema["o"] = mu * ema["o"] + (1 - mu) * ov
            ema["x"] = mu * ema["x"] + (1 - mu) * xv
        return wo, wx

    def step(batch, train: bool, gen=None, epoch: int = 0, batch_idx: int = 0):
        if train and args.sample_visible_per_batch:
            batch = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=batch_idx, seed=args.seed,
                min_k=args.min_imus, max_k=args.max_imus, split="train",
            )
        elif (not train) and args.eval_fixed_visible_set:
            batch = apply_visible_imu_sampling(
                batch, epoch=0, batch_idx=0, seed=args.seed,
                min_k=args.eval_imus, max_k=args.eval_imus, split="val",
                fixed_per_dataset=True, visible_sets=eval_visible_sets,
            )
        imu = batch["imu"].to(dev, non_blocking=True)
        tgt = batch["target"].to(dev, non_blocking=True)          # [B,T,J,D]
        imu_mask = batch["imu_mask"].to(dev, non_blocking=True)
        avail = batch["mask"].to(dev, non_blocking=True)          # [B,J]
        lengths = batch["lengths"].to(dev, non_blocking=True)
        if train:
            if args.acc_noise > 0 or args.ori_noise_deg > 0:
                imu = add_imu_noise(imu, imu_mask, args.acc_noise, args.ori_noise_deg)
            # training: sparse IMUs were sampled above; known expert joints vary per sequence
            known = sample_known(avail, args.known_min, args.known_max, gen)
        else:
            # deployment-matched eval: a FIXED sparse sensor + known-joint config,
            # deterministic across epochs (so model selection matches how the model is
            # actually used), instead of the all-sensors regime the model never trains on.
            if not args.eval_fixed_visible_set:
                keep = sample_known(imu_mask, args.eval_imus, args.eval_imus, gen)   # k of present
                imu = imu * keep.view(imu.shape[0], 1, imu.shape[2], 1).to(imu.dtype)
                imu_mask = keep
            known = sample_known(avail, args.eval_known, args.eval_known, gen)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            if joint_experts is not None:
                with torch.no_grad():
                    jin, expert_valid = joint_experts(imu, imu_mask, lengths)
                known = expert_valid & avail
            elif experts is not None:
                with torch.no_grad():
                    jin = experts(imu, imu_mask, lengths)
                if args.selected_expert_idxs:
                    known = fixed_known_from_selection(avail, args.selected_expert_idxs)
            else:
                jin = tgt
                if train and args.pose_noise > 0:
                    jin = jin.clone()
                    a, b = orient_slice
                    jin[..., a:b] = jin[..., a:b] + torch.randn_like(jin[..., a:b]) * args.pose_noise
            pred = model(jin, known, imu, imu_mask, lengths)
            # RAW (unweighted) component errors + weighted total (+ smoothness reg).
            oa, ob = orient_slice
            o_raw = orientation_6d_loss(pred[..., oa:ob], tgt[..., oa:ob], avail, lengths, per_joint=False)
            if motion_delta_slice is not None:
                xa, xb = motion_delta_slice
                x_raw = motion_delta_loss(pred[..., xa:xb], tgt[..., xa:xb], avail, lengths, per_joint=False)
            else:
                x_raw = pred.new_zeros(())
            wo, wx = component_weights(o_raw, x_raw, train)
            lossw["wo"], lossw["wx"] = wo, wx
            loss = wo * o_raw + wx * x_raw + args.smooth_w * smoothness(pred, lengths)
        if train:
            return loss, float(o_raw.detach()), float(x_raw.detach())
        unk = avail & ~known
        a, b = orient_slice
        ang_unk = geodesic_deg(pred[..., a:b].float(), tgt[..., a:b].float(), unk, lengths)
        ang_all = geodesic_deg(pred[..., a:b].float(), tgt[..., a:b].float(), avail, lengths)
        base = identity_baseline_deg(tgt[..., a:b].float(), unk, lengths)
        # xyz motion error (cm) on the RECOVERED (unknown) joints, mirroring ang_unk
        if motion_delta_slice is not None:
            xa, xb = motion_delta_slice
            xn, xd = xyz_dist_sums_per_joint(pred[..., xa:xb].float(), tgt[..., xa:xb].float(), unk, lengths)
            cm_unk = float(xn.sum() / xd.sum().clamp_min(1.0))
        else:
            cm_unk = float("nan")
        return loss, ang_unk, ang_all, base, cm_unk, float(o_raw.detach()), float(x_raw.detach())

    def evalset(dl):
        gen = torch.Generator(device=dev).manual_seed(0)   # reproducible known sets
        a = u = c = bs = cm = orw = xrw = 0.0
        with torch.no_grad():
            for b in dl:
                _, ang_unk, ang_all, base, cm_unk, o_raw, x_raw = step(b, False, gen)
                u += float(ang_unk); a += float(ang_all); bs += float(base); cm += cm_unk
                orw += o_raw; xrw += x_raw; c += 1
        return (u / max(c, 1), a / max(c, 1), bs / max(c, 1), cm / max(c, 1),
                orw / max(c, 1), xrw / max(c, 1))

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    best = float("inf"); gstep = 0; since_improve = 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; otot = 0.0; xtot = 0.0; nb = 0
        for batch_idx, batch in enumerate(train_dl):
            loss, o_raw, x_raw = step(batch, True, epoch=ep, batch_idx=batch_idx)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
            tot += float(loss.detach()); otot += o_raw; xtot += x_raw; nb += 1; gstep += 1
            if args.log_every and gstep % args.log_every == 0:
                print(f"  step={gstep} loss={tot/nb:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.max_steps and gstep >= args.max_steps:
                print(f"[smoke] {gstep} steps ok loss={float(loss.detach()):.4f} "
                      f"{(time.time()-t0)/nb*1000:.0f}ms/step")
                torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
                return
        model.eval()
        vu, va, vb, vcm, vo, vx = evalset(val_dl)
        # NOTE: the test set (DIP/TotalCapture/IMUPoser) is NOT evaluated here.
        # Model selection uses VAL ONLY; test is touched exactly once at the end.
        lo, lx = lossw["wo"], lossw["wx"]    # current (possibly adaptive) weights
        tro, trx = otot / max(nb, 1), xtot / max(nb, 1)
        # val_unk = orientation error (deg) on recovered joints; val_xyz = motion error (cm)
        # on the same joints. Then raw loss, weight, scaled loss for each (t/v = train/val).
        print(f"epoch={ep+1}/{args.epochs} train_loss={tot/max(nb,1):.4f} "
              f"val_unk={vu:.2f}deg val_xyz={vcm:.3f}cm val_all={va:.2f}deg (identity~{vb:.1f}deg) | "
              f"ORIENT raw[t/v]={tro:.5f}/{vo:.5f} w={lo:.2f} scaled[t/v]={lo*tro:.5f}/{lo*vo:.5f} | "
              f"XYZ raw[t/v]={trx:.5f}/{vx:.5f} w={lx:.2f} scaled[t/v]={lx*trx:.5f}/{lx*vx:.5f} | "
              f"scaled xyz:orient[t/v]={lx*trx/max(lo*tro,1e-9):.2f}/{lx*vx/max(lo*vo,1e-9):.2f} "
              f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s")
        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": ep + 1,
                "joints": CANONICAL_JOINTS, "imus": CANONICAL_IMUS}
        torch.save(ckpt, args.save)
        if vu < best - 1e-3:
            best = vu; since_improve = 0
            torch.save({**ckpt, "val_unk": vu}, args.save.replace(".pt", "_best.pt"))
            print(f"  [best] val_unk={vu:.2f}deg -> {args.save.replace('.pt', '_best.pt')}")
        else:
            since_improve += 1
        if args.patience and since_improve >= args.patience:
            print(f"[early-stop] val_unk plateaued for {args.patience} epochs at epoch {ep+1}")
            break
    print(f"[done] best val_unk={best:.2f}deg")
    # ---- single, final held-out evaluation on the best-VAL checkpoint ----
    best_path = args.save.replace(".pt", "_best.pt")
    model.load_state_dict(torch.load(best_path, map_location=dev)["model"])
    model.eval()
    tu, ta, tb, tcm, to_, tx_ = evalset(test_dl)
    lo, lx = lossw["wo"], lossw["wx"]
    print(f"[test] FINAL held-out {holdout} on best-val ckpt: "
          f"test_unk={tu:.2f}deg test_xyz={tcm:.3f}cm test_all={ta:.2f}deg (identity~{tb:.1f}deg) | "
          f"ORIENT raw={to_:.5f} w={lo:.2f} scaled={lo*to_:.5f} | "
          f"XYZ raw={tx_:.5f} w={lx:.2f} scaled={lx*tx_:.5f}")


if __name__ == "__main__":
    main()
