#!/usr/bin/env python3
"""Freeze a deterministic, leak-free train/val/test split MANIFEST.

Splitting is done at the SEQUENCE level (group key = the motion id with its
trailing ``:<window>`` stripped), so overlapping windows of one motion never
straddle train and val. The test set is the standard sparse-IMU benchmark
(DIP / TotalCapture / IMUPoser), matched by motion-name keyword, and is held out
entirely. Everything else is grouped by sequence and ~val_frac of the *windows*
(via whole sequences) is assigned to val.

Output JSON (loaded by every training/eval run -> one reproducible source of
truth):
  {"seq_id": {<id>: "train|val|test"}, "counts": {...}, "meta": {...}}
"""
import argparse
import glob
import json
import random
import re
from collections import Counter

import torch

SEQ_RE = re.compile(r":\d+$")


def seq_id(motion: str) -> str:
    """Underlying sequence id = motion string without the trailing window index."""
    return SEQ_RE.sub("", str(motion))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/pengfei/Downloads/poser_mle_orient")
    ap.add_argument("--out", default="/home/pengfei/Downloads/poser_mle_orient/splits.json")
    ap.add_argument("--test-kw", default="dip,totalcapture,imuposer")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test_kw = [k.strip().lower() for k in args.test_kw.split(",") if k.strip()]

    paths = sorted(glob.glob(f"{args.data}/shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt in {args.data}")

    # sequence -> window count, and sequence -> source (for reporting)
    seq_windows = Counter()
    seq_source = {}
    seq_is_test = {}
    for p in paths:
        for s in torch.load(p, map_location="cpu")["samples"]:
            sid = seq_id(s.get("motion", ""))
            seq_windows[sid] += 1
            seq_source.setdefault(sid, str(s.get("source", "")))
            if sid not in seq_is_test:
                m = str(s.get("motion", "")).lower()
                seq_is_test[sid] = any(k in m for k in test_kw)

    test_seqs = [s for s in seq_windows if seq_is_test[s]]
    pool = [s for s in seq_windows if not seq_is_test[s]]
    random.Random(args.seed).shuffle(pool)

    # greedily fill val with whole sequences up to ~val_frac of the pool's windows
    pool_windows = sum(seq_windows[s] for s in pool)
    target_val = args.val_frac * pool_windows
    val_seqs, cum = [], 0
    for s in pool:
        if cum >= target_val:
            break
        val_seqs.append(s)
        cum += seq_windows[s]
    val_set = set(val_seqs)
    train_seqs = [s for s in pool if s not in val_set]

    assignment = {}
    for s in train_seqs:
        assignment[s] = "train"
    for s in val_seqs:
        assignment[s] = "val"
    for s in test_seqs:
        assignment[s] = "test"

    def wc(seqs):
        return sum(seq_windows[s] for s in seqs)

    # sanity: the three sequence sets are disjoint by construction
    assert not (set(train_seqs) & val_set) and not (set(train_seqs) & set(test_seqs)) \
        and not (val_set & set(test_seqs)), "split overlap!"

    counts = {
        "train": {"seqs": len(train_seqs), "windows": wc(train_seqs)},
        "val": {"seqs": len(val_seqs), "windows": wc(val_seqs)},
        "test": {"seqs": len(test_seqs), "windows": wc(test_seqs)},
    }
    manifest = {
        "seq_id": assignment,
        "counts": counts,
        "meta": {
            "data": args.data, "test_kw": test_kw, "val_frac": args.val_frac,
            "seed": args.seed, "total_sequences": len(seq_windows),
            "total_windows": sum(seq_windows.values()),
            "note": "group key = motion id with trailing ':<window>' stripped; "
                    "splitting at sequence level prevents window leakage.",
        },
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[splits] wrote {args.out}")
    for k, v in counts.items():
        print(f"  {k:5s} {v['seqs']:5d} seqs  {v['windows']:6d} windows")
    # show test composition by source
    tsrc = Counter(seq_source[s] for s in test_seqs)
    print("  test sources:", dict(tsrc))


if __name__ == "__main__":
    main()
