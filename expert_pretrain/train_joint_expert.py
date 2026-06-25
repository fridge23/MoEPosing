#!/usr/bin/env python3
"""Train one joint expert at a time, with independent state/checkpoints."""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from masked_dataset import MaskedMotionDataset, apply_visible_imu_sampling, build_fixed_visible_sets, collate
from multiexpert_model import (
    EXPERT_RAW_OUTPUT_DIM,
    JointExpert,
    expert_target_loss,
    motion_delta_loss,
    orientation_6d_loss,
)
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys, target_dim
from train_multiexpert import add_imu_noise


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


def load_config(path: str) -> Dict:
    if not path:
        return {}
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML; use JSON config or install pyyaml") from exc
    return yaml.safe_load(text) or {}


def deep_get(cfg: Dict, path: Iterable[str], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def joint_index(name_or_id) -> int:
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        idx = int(name_or_id)
        if 0 <= idx < len(CANONICAL_JOINTS):
            return idx
        raise ValueError(f"target joint id out of range: {idx}")
    lookup = {_norm_name(n): i for i, n in enumerate(CANONICAL_JOINTS)}
    key = _norm_name(name_or_id)
    key = _norm_name(JOINT_ALIASES.get(key, name_or_id))
    if key not in lookup:
        raise ValueError(f"Unknown joint '{name_or_id}'. Known joints: {CANONICAL_JOINTS}")
    return lookup[key]


def imu_indices(values) -> List[int]:
    lookup = {_norm_name(n): i for i, n in enumerate(CANONICAL_IMUS)}
    out = []
    for value in values:
        if isinstance(value, int) or str(value).isdigit():
            out.append(int(value))
            continue
        key = _norm_name(value)
        if key not in lookup:
            raise ValueError(f"Unknown IMU '{value}'. Known IMUs: {CANONICAL_IMUS}")
        out.append(lookup[key])
    return out


def eval_visible_sets_from_config(cfg: Dict) -> Dict[str, List[int]]:
    out = {}
    for dataset_name, spec in (cfg.get("datasets") or {}).items():
        if not isinstance(spec, dict) or spec.get("eval_visible_set") is None:
            continue
        out[str(dataset_name)] = imu_indices(spec["eval_visible_set"])
    return out


def describe_dataset_sources(ds) -> List[str]:
    if hasattr(ds, "describe_sources"):
        return ds.describe_sources()
    return []


def per_joint_setting(settings: Dict, joint_name: str, default):
    if not isinstance(settings, dict):
        return default
    keys = {joint_name, _norm_name(joint_name)}
    for raw, canonical in JOINT_ALIASES.items():
        if canonical == joint_name:
            keys.add(raw)
    for key in keys:
        if key in settings:
            return settings[key]
    return settings.get("default", default)


@dataclass
class JointEarlyStopper:
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    best: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int):
        is_better = self.improved(value)
        if is_better:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = self.patience > 0 and self.epochs_without_improvement >= self.patience
        return is_better, should_stop


def simulate_early_stop_epochs(val_curves: Dict[str, List[float]], patience: Dict[str, int],
                               min_delta: float = 0.0):
    """Small deterministic helper used by smoke tests."""
    stoppers = {
        name: JointEarlyStopper(patience=patience.get(name, patience.get("default", 1)), min_delta=min_delta)
        for name in val_curves
    }
    active = set(val_curves)
    stop_epochs = {}
    update_counts = {name: 0 for name in val_curves}
    max_len = max(len(v) for v in val_curves.values())
    for epoch in range(max_len):
        for name in list(active):
            curve = val_curves[name]
            if epoch >= len(curve):
                active.remove(name)
                continue
            update_counts[name] += 1
            _, stop = stoppers[name].step(curve[epoch], epoch)
            if stop:
                stop_epochs[name] = epoch
                active.remove(name)
        if not active:
            break
    return stop_epochs, update_counts


def key_padding_mask(lengths: torch.Tensor, t: int, device) -> torch.Tensor:
    steps = torch.arange(t, device=device).view(1, t)
    return steps >= lengths.to(device).view(-1, 1)


def joint_losses(pred, target, mask, lengths, orient_slice, motion_slice, per_joint=True):
    pred4 = pred.unsqueeze(2)
    pieces = {
        "loss": expert_target_loss(
            pred4, target, mask, lengths,
            orientation_slice=orient_slice, motion_delta_slice=motion_slice,
            per_joint=per_joint,
        )
    }
    a, b = orient_slice
    pieces["orientation_loss"] = orientation_6d_loss(pred4[..., a:b], target[..., a:b], mask, lengths,
                                                     per_joint=per_joint)
    a, b = motion_slice
    pieces["motion_delta_loss"] = motion_delta_loss(pred4[..., a:b], target[..., a:b], mask, lengths,
                                                    per_joint=per_joint)
    return pieces


def checkpoint_path(save_dir: str, joint_idx: int, joint_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", joint_name)
    return str(Path(save_dir) / f"{joint_idx:02d}_{safe}_best.pt")


def save_joint_checkpoint(path: str, model: JointExpert, optimizer, scheduler, *, joint_idx: int,
                          joint_name: str, epoch: int, val_loss: float, args: Dict,
                          eval_visible_sets: Dict[str, List[int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "joint_idx": joint_idx,
        "joint_name": joint_name,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "args": args,
        "imus": CANONICAL_IMUS,
        "joints": CANONICAL_JOINTS,
        "eval_visible_sets": eval_visible_sets,
    }, path)


def train_one_joint(joint_idx: int, train_dl, val_dl, eval_visible_sets, args, device, slices):
    joint_name = CANONICAL_JOINTS[joint_idx]
    max_epochs = int(per_joint_setting(args["max_epochs_per_joint"], joint_name, args["epochs"]))
    patience = int(per_joint_setting(args["patience_per_joint"], joint_name, args["patience"]))
    stopper = JointEarlyStopper(patience=patience, min_delta=args["min_delta"], mode=args["monitor_mode"])

    model = JointExpert(len(CANONICAL_IMUS), hidden_dim=args["hidden"], nhead=args["nhead"],
                        num_layers=args["layers"], dropout=args["dropout"],
                        target_dim=args["target_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs * len(train_dl)))
    ckpt_path = checkpoint_path(args["save_dir"], joint_idx, joint_name)
    orient_slice, motion_slice = slices

    print(f"[joint] {joint_idx:02d} {joint_name} max_epochs={max_epochs} patience={patience} "
          f"fixed_val_visible={eval_visible_sets} ckpt={ckpt_path}")
    for epoch in range(max_epochs):
        model.train(); t0 = time.time()
        train_sum = orient_sum = delta_sum = 0.0; nb = 0; visible_sizes = {}
        for batch_idx, batch in enumerate(train_dl):
            batch = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=batch_idx, seed=args["seed"],
                min_k=args["min_k"], max_k=args["max_k"], split="train",
            )
            for keep in batch["visible_imu_indices"]:
                visible_sizes[len(keep)] = visible_sizes.get(len(keep), 0) + 1
            imu = batch["imu"].to(device)
            imu_mask = batch["imu_mask"].to(device)
            if args["acc_noise"] > 0 or args["ori_noise_deg"] > 0:
                imu = add_imu_noise(imu, imu_mask, args["acc_noise"], args["ori_noise_deg"])
            target = batch["target"][:, :, joint_idx:joint_idx + 1, :].to(device)
            mask = batch["mask"][:, joint_idx:joint_idx + 1].to(device)
            lengths = batch["lengths"].to(device)
            pred = model(imu, imu_mask, key_padding_mask(lengths, imu.shape[1], device))
            losses = joint_losses(pred, target, mask, lengths, orient_slice, motion_slice,
                                  per_joint=args["loss"] == "per_joint")
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            train_sum += float(losses["loss"].detach())
            orient_sum += float(losses["orientation_loss"].detach())
            delta_sum += float(losses["motion_delta_loss"].detach())
            nb += 1

        model.eval()
        val_sum = val_orient = val_delta = vc = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = apply_visible_imu_sampling(
                    batch, epoch=0, batch_idx=0, seed=args["seed"],
                    min_k=args["min_k"], max_k=args["max_k"], split="val",
                    fixed_per_dataset=True, visible_sets=eval_visible_sets,
                )
                imu = batch["imu"].to(device)
                imu_mask = batch["imu_mask"].to(device)
                target = batch["target"][:, :, joint_idx:joint_idx + 1, :].to(device)
                mask = batch["mask"][:, joint_idx:joint_idx + 1].to(device)
                lengths = batch["lengths"].to(device)
                pred = model(imu, imu_mask, key_padding_mask(lengths, imu.shape[1], device))
                losses = joint_losses(pred, target, mask, lengths, orient_slice, motion_slice,
                                      per_joint=args["loss"] == "per_joint")
                val_sum += float(losses["loss"]); val_orient += float(losses["orientation_loss"])
                val_delta += float(losses["motion_delta_loss"]); vc += 1
        train_loss = train_sum / max(nb, 1)
        val_loss = val_sum / max(vc, 1)
        improved, should_stop = stopper.step(val_loss, epoch)
        if improved:
            save_joint_checkpoint(ckpt_path, model, opt, sched, joint_idx=joint_idx, joint_name=joint_name,
                                  epoch=epoch + 1, val_loss=val_loss, args=args,
                                  eval_visible_sets=eval_visible_sets)
        print(f"joint={joint_name} epoch={epoch+1}/{max_epochs} train={train_loss:.6f} "
              f"val={val_loss:.6f} orient={val_orient/max(vc,1):.6f} "
              f"delta={val_delta/max(vc,1):.6f} visible_sizes={visible_sizes} "
              f"best_epoch={stopper.best_epoch+1 if stopper.best_epoch >= 0 else None} "
              f"early_stop={should_stop} {time.time()-t0:.1f}s")
        if should_stop:
            print(f"[early-stop] {joint_name} stopped at epoch {epoch+1}; best_epoch={stopper.best_epoch+1}")
            break
    return {"joint": joint_name, "best_epoch": stopper.best_epoch + 1, "best_val_loss": stopper.best,
            "checkpoint": ckpt_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--target-joint", default="")
    ap.add_argument("--train-all-experts", action="store_true")
    ap.add_argument("--data", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--num-workers", type=int, default=-1)
    cli = ap.parse_args()
    cfg = load_config(cli.config)

    args = {
        "data": cli.data or deep_get(cfg, ["data"], "data"),
        "manifest": cli.manifest if cli.manifest else deep_get(cfg, ["manifest"], "data/splits.json"),
        "save_dir": cli.save_dir or deep_get(cfg, ["expert_training", "save_dir"], "checkpoints/joint_experts"),
        "target": deep_get(cfg, ["target"], "joint_orient_r6d,joint_delta"),
        "mask_key": deep_get(cfg, ["mask_key"], "auto"),
        "epochs": int(deep_get(cfg, ["expert_training", "max_epochs_per_joint", "default"], 100)),
        "max_epochs_per_joint": deep_get(cfg, ["expert_training", "max_epochs_per_joint"], {"default": 100}),
        "patience": int(deep_get(cfg, ["expert_training", "early_stopping", "patience", "default"], 15)),
        "patience_per_joint": deep_get(cfg, ["expert_training", "early_stopping", "patience"], {"default": 15}),
        "min_delta": float(deep_get(cfg, ["expert_training", "early_stopping", "min_delta"], 0.0)),
        "monitor_mode": deep_get(cfg, ["expert_training", "early_stopping", "mode"], "min"),
        "batch_size": int(deep_get(cfg, ["batch_size"], 64)),
        "lr": float(deep_get(cfg, ["lr"], 3e-4)),
        "weight_decay": float(deep_get(cfg, ["weight_decay"], 1e-4)),
        "hidden": int(deep_get(cfg, ["hidden"], 128)),
        "layers": int(deep_get(cfg, ["layers"], 4)),
        "nhead": int(deep_get(cfg, ["nhead"], 4)),
        "dropout": float(deep_get(cfg, ["dropout"], 0.1)),
        "min_k": int(deep_get(cfg, ["visible_imu_sampling", "min_k"], 2)),
        "max_k": int(deep_get(cfg, ["visible_imu_sampling", "max_k"], 5)),
        "seed": int(deep_get(cfg, ["visible_imu_sampling", "seed"], 42)),
        "loss": deep_get(cfg, ["loss"], "per_joint"),
        "acc_noise": float(deep_get(cfg, ["acc_noise"], 0.0)),
        "ori_noise_deg": float(deep_get(cfg, ["ori_noise_deg"], 0.0)),
        "num_workers": cli.num_workers if cli.num_workers >= 0 else int(deep_get(cfg, ["num_workers"], 4)),
    }
    args["target_keys"] = split_keys(args["target"])
    args["target_dim"] = target_dim(args["target"])
    orient_slice = slice_for(args["target_keys"], "joint_orient_r6d")
    motion_slice = slice_for(args["target_keys"], "joint_delta")
    if args["target_dim"] != EXPERT_RAW_OUTPUT_DIM or orient_slice is None or motion_slice is None:
        raise ValueError("train_joint_expert.py expects target joint_orient_r6d,joint_delta (9D)")

    device = torch.device((cli.device or deep_get(cfg, ["device"], "cuda")) if torch.cuda.is_available() else "cpu")
    manifest = args["manifest"] or None
    if manifest:
        train_ds = MaskedMotionDataset(args["data"], min_k=args["min_k"], max_k=args["max_k"], augment=False,
                                       target_key=args["target"], mask_key=args["mask_key"],
                                       manifest=manifest, split="train", require_min_imus=True)
        val_ds = MaskedMotionDataset(args["data"], augment=False, target_key=args["target"],
                                     mask_key=args["mask_key"], manifest=manifest, split="val",
                                     require_min_imus=True)
    else:
        full = MaskedMotionDataset(args["data"], min_k=args["min_k"], max_k=args["max_k"], augment=False,
                                   target_key=args["target"], mask_key=args["mask_key"], require_min_imus=True)
        idx = list(range(len(full)))
        n_val = max(1, int(0.05 * len(full)))
        train_ds, val_ds = Subset(full, idx[n_val:]), Subset(full, idx[:n_val])
    train_dl = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, drop_last=True,
                          num_workers=args["num_workers"], collate_fn=collate,
                          persistent_workers=args["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False,
                        num_workers=max(0, min(2, args["num_workers"])), collate_fn=collate)
    eval_visible_sets = build_fixed_visible_sets(
        [val_ds], seed=args["seed"], min_k=args["min_k"], max_k=args["max_k"],
        split="val", provided=eval_visible_sets_from_config(cfg),
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} fixed_val_visible={eval_visible_sets}")
    for line in describe_dataset_sources(train_ds):
        print(f"[data:train-source] {line}")
    for line in describe_dataset_sources(val_ds):
        print(f"[data:val-source] {line}")

    if cli.train_all_experts or deep_get(cfg, ["expert_training", "train_mode"], "") == "all_experts":
        selected = list(range(len(CANONICAL_JOINTS)))
    else:
        target = cli.target_joint or deep_get(cfg, ["expert_training", "target_joint_id"], None)
        if target is None:
            raise ValueError("Provide --target-joint or --train-all-experts")
        selected = [joint_index(target)]

    results = []
    for joint_idx in selected:
        results.append(train_one_joint(joint_idx, train_dl, val_dl, eval_visible_sets, args, device,
                                       (orient_slice, motion_slice)))
    print("[done]", results)


if __name__ == "__main__":
    main()
