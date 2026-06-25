#!/usr/bin/env python3
"""Rigorous held-out evaluation of the per-joint experts (MPJRE).

Metric: MPJRE = mean per-joint geodesic angle (degrees) on the orientation slice.
For multi-target experts, non-orientation outputs such as xyz delta/position are
loaded and predicted too, but this script reports the standard orientation metric.

Reporting (professional):
  * per ACTIVE-IMU count k in {2,3,4,5}: deterministically mask each test window
    to a random k-subset of its present sensors (fixed seed) -> the rehab/exercise
    deployment regime,
  * per-sequence MPJRE aggregated as mean +/- std OVER WINDOWS (not pooled frames),
  * per-joint MPJRE table (which joints are easy/hard),
  * identity baseline for reference.
Evaluates on the frozen manifest test split (DIP/TotalCapture/IMUPoser) by default.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from masked_dataset import MaskedMotionDataset, collate, sample_visible_set
from multiexpert_model import MultiExpert
from train_multiexpert import geodesic_sums_per_joint, identity_baseline_deg
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pengfei/Downloads/poser_mle_orient")
    ap.add_argument("--manifest", default="/home/pengfei/Downloads/poser_mle_orient/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="/home/pengfei/Downloads/dynaip/weights/multiexpert_orient_v4_best.pt")
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ks = [int(x) for x in args.ks.split(",")]
    J = len(CANONICAL_JOINTS)

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    target_keys = split_keys(a.get("target", "joint_orient_r6d"))
    orient_slice = slice_for(target_keys, "joint_orient_r6d")
    if orient_slice is None:
        raise ValueError(f"Checkpoint target has no joint_orient_r6d slice: {target_keys}")
    model = MultiExpert(len(CANONICAL_IMUS), J, hidden_dim=a["hidden"], nhead=a["nhead"],
                        num_layers=a["layers"], target_dim=a["target_dim"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} loss={a.get('loss')} "
          f"split={args.split}")

    ds = MaskedMotionDataset(args.data, augment=False, target_key=",".join(target_keys),
                             mask_key=a.get("mask_key", "auto"), manifest=args.manifest, split=args.split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)
    print(f"[data] {len(ds)} windows in '{args.split}'")

    # identity baseline (per window macro angle)
    base_vals = []
    with torch.no_grad():
        for b in dl:
            ta, tb = orient_slice
            base_vals.append(float(identity_baseline_deg(
                b["target"].to(dev)[..., ta:tb], b["mask"].to(dev), b["lengths"].to(dev))))
    base = torch.tensor(base_vals)
    print(f"[identity baseline] MPJRE={base.mean():.2f} +/- {base.std():.2f} deg\n")

    # per-k evaluation
    pj_num_all = torch.zeros(J)   # for the all-k per-joint table (pooled over k)
    pj_den_all = torch.zeros(J)
    print(f"{'k':>3} {'MPJRE mean+/-std (deg)':>26} {'median':>8}  (over windows)")
    for k in ks:
        per_window = []
        with torch.no_grad():
            for b in dl:
                imu = b["imu"].to(dev); tgt = b["target"].to(dev)
                im = b["available_imu_mask"][0]; ln = b["lengths"].to(dev); mk = b["mask"].to(dev)
                source = str(b["source"][0])
                keep = sample_visible_set(source, 0, 0, im.nonzero().flatten().tolist(),
                                          seed=args.seed + k, sample_idx=0,
                                          min_k=k, max_k=k, split="eval")
                km = torch.zeros_like(im)
                km[keep] = True
                km = km.unsqueeze(0).to(dev)
                pred = model(imu, km, ln)
                ta, tb = orient_slice
                num, den = geodesic_sums_per_joint(pred[..., ta:tb].float(), tgt[..., ta:tb].float(), mk, ln)
                if den.sum() > 0:
                    per_window.append(float(num.sum() / den.sum()))
                    pj_num_all += num.cpu(); pj_den_all += den.cpu()
        pw = torch.tensor(per_window)
        print(f"{k:>3} {pw.mean():>14.2f} +/- {pw.std():<8.2f} {pw.median():>8.2f}")

    # per-joint table (pooled over all k)
    pj = pj_num_all / pj_den_all.clamp_min(1.0)
    pj[pj_den_all == 0] = float("nan")
    order = sorted(range(J), key=lambda j: (float("inf") if torch.isnan(pj[j]) else float(pj[j])))
    print("\n[per-joint MPJRE] (pooled over k, easiest -> hardest)")
    for j in order:
        if not torch.isnan(pj[j]):
            print(f"  {CANONICAL_JOINTS[j]:>18s}  {float(pj[j]):6.2f} deg")
    valid = ~torch.isnan(pj)
    print(f"\n[summary] macro MPJRE (mean over joints) = {float(pj[valid].mean()):.2f} deg "
          f"over {int(valid.sum())} joints with GT")


if __name__ == "__main__":
    main()
