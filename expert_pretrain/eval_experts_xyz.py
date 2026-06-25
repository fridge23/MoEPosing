#!/usr/bin/env python3
"""Per-joint held-out test errors for the per-joint experts, reporting BOTH the
orientation error (deg, geodesic) AND the xyz motion-delta error (cm), using the
EXACT same protocol as train_multiexpert.py's final [test] line (fixed per-dataset
visible IMU set). Reproduces the macro deg as a sanity check, then prints per joint.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from masked_dataset import (MaskedMotionDataset, collate, build_fixed_visible_sets,
                            apply_visible_imu_sampling)
from multiexpert_model import MultiExpert
from train_multiexpert import geodesic_sums_per_joint, xyz_dist_sums_per_joint
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import slice_for, split_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/pengfei/Downloads/dynaip/weights/multiexpert_9d64_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="0=read target_fps from metadata.pt when available; used to convert "
                         "joint_delta cm/frame to MPJVE mm/s")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    tkeys = split_keys(a["target"])
    osl = slice_for(tkeys, "joint_orient_r6d")
    xsl = slice_for(tkeys, "joint_delta")
    J = len(CANONICAL_JOINTS)

    model = MultiExpert(len(CANONICAL_IMUS), J, hidden_dim=a["hidden"], nhead=a["nhead"],
                        num_layers=a["layers"], target_dim=a["target_dim"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} target={a['target']}")

    # Rebuild the SAME val+test datasets and fixed visible sets the trainer used.
    val_ds = MaskedMotionDataset(a["data"], augment=False, target_key=a["target"],
                                 mask_key=a["mask_key"], manifest=a["manifest"], split="val",
                                 require_min_imus=True)
    test_ds = MaskedMotionDataset(a["data"], augment=False, target_key=a["target"],
                                  mask_key=a["mask_key"], manifest=a["manifest"], split="test",
                                  require_min_imus=True)
    fps = args.fps
    if fps <= 0:
        try:
            meta = torch.load(f"{a['data'].rstrip('/')}/metadata.pt", map_location="cpu")
            fps = float(meta.get("window", {}).get("target_fps", 0.0))
        except FileNotFoundError:
            fps = 0.0
    vsets = build_fixed_visible_sets([val_ds, test_ds], seed=a["seed"],
                                     min_k=a["min_k"], max_k=a["max_k"], split="val")
    dl = DataLoader(test_ds, batch_size=a["batch_size"], shuffle=False, num_workers=2,
                    collate_fn=collate)
    print(f"[data] {len(test_ds)} test windows; fixed visible sets={vsets}")

    num = torch.zeros(J); den = torch.zeros(J)      # orientation deg sums
    xn = torch.zeros(J); xd = torch.zeros(J)        # xyz cm sums
    with torch.no_grad():
        for b in dl:
            b = apply_visible_imu_sampling(b, epoch=0, batch_idx=0, seed=a["seed"],
                                           min_k=a["min_k"], max_k=a["max_k"], split="val",
                                           fixed_per_dataset=True, visible_sets=vsets)
            imu = b["imu"].to(dev); tgt = b["target"].to(dev)
            im = b["imu_mask"].to(dev); mk = b["mask"].to(dev); ln = b["lengths"].to(dev)
            pred = model(imu, im, ln)
            oa, ob = osl
            n_, d_ = geodesic_sums_per_joint(pred[..., oa:ob].float(), tgt[..., oa:ob].float(), mk, ln)
            xa, xb = xsl
            xn_, xd_ = xyz_dist_sums_per_joint(pred[..., xa:xb].float(), tgt[..., xa:xb].float(), mk, ln)
            num += n_.cpu(); den += d_.cpu(); xn += xn_.cpu(); xd += xd_.cpu()

    deg = num / den.clamp_min(1.0); deg[den == 0] = float("nan")
    cm = xn / xd.clamp_min(1.0); cm[xd == 0] = float("nan")
    v = ~torch.isnan(deg)
    macro_deg = float(deg[v].mean()); macro_cm = float(cm[v].mean())
    mpjve = cm * 10.0 * fps if fps > 0 else torch.full_like(cm, float("nan"))
    macro_mpjve = float(mpjve[v].mean()) if fps > 0 else float("nan")
    print(f"\n[sanity] macro orient={macro_deg:.2f}deg (expect ~22.38)  "
          f"macro xyz={macro_cm:.3f}cm/frame  macro MPJVE={macro_mpjve:.1f}mm/s "
          f"median orient={float(deg[v].median()):.2f}deg "
          f"median xyz={float(cm[v].median()):.3f}cm/frame fps={fps:.1f}\n")

    order = sorted(range(J), key=lambda j: (float("inf") if torch.isnan(cm[j]) else float(cm[j])))
    print(f"{'joint':>18s}  {'test_orient':>11s}  {'test_xyz':>14s}  {'MPJVE':>10s}   (easiest->hardest by xyz)")
    for j in order:
        if v[j]:
            mv = f"{float(mpjve[j]):.1f}mm/s" if fps > 0 else "n/a"
            print(f"{CANONICAL_JOINTS[j]:>18s}  {float(deg[j]):8.2f}deg  "
                  f"{float(cm[j]):10.3f}cm/f  {mv:>10s}")


if __name__ == "__main__":
    main()
