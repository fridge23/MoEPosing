import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from masked_dataset import apply_visible_imu_sampling
from multiexpert_model import (
    EXPERT_RAW_OUTPUT_DIM,
    ExpertOutputAssembler,
    MultiExpert,
    SparseIMUJointTokenizer,
    expert_target_loss,
)
from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from train_joint_expert import (
    JointEarlyStopper,
    checkpoint_path,
    save_joint_checkpoint,
    simulate_early_stop_epochs,
)
from train_wholebody import load_joint_expert_bank
from wholebody_model import WholeBodyPoser


B, T = 2, 4
S, J = len(CANONICAL_IMUS), len(CANONICAL_JOINTS)


def fake_batch():
    available = torch.zeros(B, S, dtype=torch.bool)
    available[0, [0, 1, 4, 5, 6, 7]] = True
    available[1, :] = True
    return {
        "imu": torch.randn(B, T, S, 12),
        "imu_mask": available.clone(),
        "available_imu_mask": available,
        "target": torch.randn(B, T, J, EXPERT_RAW_OUTPUT_DIM),
        "mask": torch.ones(B, J, dtype=torch.bool),
        "lengths": torch.full((B,), T, dtype=torch.long),
        "source": ["mobileposer", "dynaip_andy"],
        "sample_idx": torch.arange(B),
    }


class SmokeDesignTest(unittest.TestCase):
    def test_expert_shapes(self):
        batch = fake_batch()
        model = MultiExpert(S, J, hidden_dim=16, nhead=4, num_layers=1, target_dim=EXPERT_RAW_OUTPUT_DIM)
        masked_imu_tokens, visible_joint_mask = model.experts[0].imu_tokenizer(batch["imu"], batch["imu_mask"])
        self.assertEqual(masked_imu_tokens.shape, (B, T, J, 12))
        self.assertEqual(visible_joint_mask.shape, (B, J))
        out = model.forward_dict(batch["imu"], batch["imu_mask"], batch["lengths"])
        self.assertEqual(out["orientation_6d"].shape, (B, T, J, 6))
        self.assertEqual(out["motion_delta"].shape, (B, T, J, 3))

    def test_visible_imu_sampling_changes_and_is_valid(self):
        batch = fake_batch()
        seen = set()
        for batch_idx in range(6):
            sampled = apply_visible_imu_sampling(batch, epoch=0, batch_idx=batch_idx, seed=123)
            for b in range(B):
                keep = sampled["imu_mask"][b].nonzero().flatten().tolist()
                self.assertGreaterEqual(len(keep), 2)
                self.assertLessEqual(len(keep), 5)
                self.assertTrue(sampled["imu_mask"][b].logical_and(~batch["available_imu_mask"][b]).sum().item() == 0)
            seen.add(tuple(sampled["imu_mask"][0].nonzero().flatten().tolist()))
        self.assertGreater(len(seen), 1)

    def test_validation_visible_imu_sampling_is_fixed_and_valid(self):
        batch = fake_batch()
        first = apply_visible_imu_sampling(
            batch, epoch=0, batch_idx=0, seed=123, split="val", fixed_per_dataset=True,
        )
        for epoch in range(3):
            sampled = apply_visible_imu_sampling(
                batch, epoch=epoch, batch_idx=epoch + 10, seed=123, split="val", fixed_per_dataset=True,
            )
            self.assertTrue(torch.equal(sampled["imu_mask"], first["imu_mask"]))
            self.assertTrue(sampled["imu_mask"].logical_and(~batch["available_imu_mask"]).sum().item() == 0)

    def test_expert_output_assembler_shape(self):
        outs = torch.randn(B, T, 3, EXPERT_RAW_OUTPUT_DIM)
        tokens, valid = ExpertOutputAssembler(J)(outs, selected=[0, 4, 27])
        self.assertEqual(tokens.shape, (B, T, J, EXPERT_RAW_OUTPUT_DIM))
        self.assertEqual(valid.shape, (B, J))
        self.assertTrue(valid[:, [0, 4, 27]].all())

    def test_one_fake_expert_training_step(self):
        batch = fake_batch()
        model = MultiExpert(S, J, hidden_dim=16, nhead=4, num_layers=1, target_dim=EXPERT_RAW_OUTPUT_DIM)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pred = model(batch["imu"], batch["imu_mask"], batch["lengths"])
        loss = expert_target_loss(
            pred, batch["target"], batch["mask"], batch["lengths"],
            orientation_slice=(0, 6), motion_delta_slice=(6, 9),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss))

    def test_early_stopping_one_fake_expert(self):
        stopper = JointEarlyStopper(patience=2)
        stops = []
        for epoch, value in enumerate([1.0, 1.1, 1.2]):
            _, should_stop = stopper.step(value, epoch)
            stops.append(should_stop)
        self.assertEqual(stops, [False, False, True])
        self.assertEqual(stopper.best_epoch, 0)

    def test_two_experts_stop_at_different_epochs_and_stop_optimizing(self):
        stop_epochs, update_counts = simulate_early_stop_epochs(
            {
                "LeftHand": [1.0, 1.1, 1.2, 1.3, 1.4],
                "LeftLowerLeg": [1.0, 0.9, 0.8, 0.81, 0.82],
            },
            {"default": 2},
        )
        self.assertEqual(stop_epochs["LeftHand"], 2)
        self.assertEqual(stop_epochs["LeftLowerLeg"], 4)
        self.assertEqual(update_counts["LeftHand"], stop_epochs["LeftHand"] + 1)
        self.assertLess(update_counts["LeftHand"], update_counts["LeftLowerLeg"])

    def test_best_checkpoint_saved_per_expert(self):
        model = MultiExpert(S, J, hidden_dim=16, nhead=4, num_layers=1, target_dim=EXPERT_RAW_OUTPUT_DIM).experts[0]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
        with tempfile.TemporaryDirectory() as tmp:
            p0 = checkpoint_path(tmp, 0, "Pelvis")
            p1 = checkpoint_path(tmp, 1, "LeftUpperLeg")
            save_joint_checkpoint(p0, model, opt, sched, joint_idx=0, joint_name="Pelvis",
                                  epoch=3, val_loss=0.5, args={}, eval_visible_sets={"mobileposer": [0, 1]})
            save_joint_checkpoint(p1, model, opt, sched, joint_idx=1, joint_name="LeftUpperLeg",
                                  epoch=4, val_loss=0.4, args={}, eval_visible_sets={"mobileposer": [0, 1]})
            self.assertTrue(Path(p0).exists())
            self.assertTrue(Path(p1).exists())
            self.assertEqual(torch.load(p0, map_location="cpu")["joint_name"], "Pelvis")
            self.assertEqual(torch.load(p1, map_location="cpu")["joint_idx"], 1)

    def test_recovery_loads_selected_joint_expert_checkpoints(self):
        batch = fake_batch()
        args = {"hidden": 16, "nhead": 4, "layers": 1, "dropout": 0.1, "target_dim": EXPERT_RAW_OUTPUT_DIM}
        with tempfile.TemporaryDirectory() as tmp:
            for joint_idx in [0, 4]:
                model = MultiExpert(S, J, hidden_dim=16, nhead=4, num_layers=1,
                                    target_dim=EXPERT_RAW_OUTPUT_DIM).experts[joint_idx]
                save_joint_checkpoint(
                    checkpoint_path(tmp, joint_idx, CANONICAL_JOINTS[joint_idx]),
                    model, optimizer=None, scheduler=None, joint_idx=joint_idx,
                    joint_name=CANONICAL_JOINTS[joint_idx], epoch=1, val_loss=1.0,
                    args=args, eval_visible_sets={},
                )
            bank = load_joint_expert_bank(tmp, selected=[0, 4], target_dim_value=EXPERT_RAW_OUTPUT_DIM,
                                          device=torch.device("cpu"))
            tokens, valid = bank(batch["imu"], batch["imu_mask"], batch["lengths"])
            self.assertEqual(tokens.shape, (B, T, J, EXPERT_RAW_OUTPUT_DIM))
            self.assertEqual(valid.shape, (B, J))
            self.assertTrue(valid[:, [0, 4]].all())
            self.assertFalse(valid[:, [1, 2, 3]].any())

    def test_recovery_shape_and_training_step(self):
        prior = Path(__file__).resolve().parents[2] / "student_kl_18to21_best_64.pth"
        if not prior.exists():
            self.skipTest(f"missing prior checkpoint: {prior}")
        batch = fake_batch()
        model = WholeBodyPoser(str(prior), dec_layers=1, target_dim=EXPERT_RAW_OUTPUT_DIM, train_prior="lora")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        known = torch.zeros(B, J, dtype=torch.bool)
        known[:, :5] = True
        recovery_input, expert_tokens, masked_imu_tokens, visible_imu_joints = model.make_recovery_tokens(
            batch["target"], known, batch["imu"], batch["imu_mask"],
        )
        self.assertEqual(masked_imu_tokens.shape, (B, T, J, 12))
        self.assertEqual(expert_tokens.shape, (B, T, J, EXPERT_RAW_OUTPUT_DIM))
        self.assertEqual(recovery_input.shape, (B, T, J, model.recovery_input_dim))
        self.assertEqual(visible_imu_joints.shape, (B, J))
        pred_dict = model.forward_dict(batch["target"], known, batch["imu"], batch["imu_mask"], batch["lengths"])
        self.assertEqual(pred_dict["pred_full_body_orientation_6d"].shape, (B, T, J, 6))
        self.assertEqual(pred_dict["pred_full_body_motion_delta"].shape, (B, T, J, 3))
        pred = model(batch["target"], known, batch["imu"], batch["imu_mask"], batch["lengths"])
        loss = expert_target_loss(
            pred, batch["target"], batch["mask"], batch["lengths"],
            orientation_slice=(0, 6), motion_delta_slice=(6, 9), per_joint=False,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
