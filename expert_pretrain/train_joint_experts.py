#!/usr/bin/env python3
import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import JointMotionShardDataset, collate_joint_motion
from model import JointExpertTransformer, masked_motion_loss
from schema import CANONICAL_IMUS, CANONICAL_JOINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/expert_pretrain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", default="weights/joint_expert_transformer.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = JointMotionShardDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_joint_motion)
    model = JointExpertTransformer(len(CANONICAL_IMUS), len(CANONICAL_JOINTS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in dl:
            imu = batch["imu"].to(device)
            imu_mask = batch["imu_mask"].to(device)
            target = batch["joint_delta"].to(device)
            joint_mask = batch["joint_mask"].to(device)
            lengths = batch["lengths"].to(device)
            pred = model(imu, imu_mask, lengths)
            loss = masked_motion_loss(pred, target, joint_mask, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        print(f"epoch={epoch + 1} loss={total / max(n, 1):.6f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"model": model.state_dict(), "imus": CANONICAL_IMUS, "joints": CANONICAL_JOINTS}, args.save)


if __name__ == "__main__":
    main()
