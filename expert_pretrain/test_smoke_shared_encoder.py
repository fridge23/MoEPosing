#!/usr/bin/env python3
"""Smoke tests for the shared encoder + lightweight expert architecture.

Run:  python test_smoke_shared_encoder.py [--device cuda]

Tests:
  1. SharedEncoder forward pass and output shape
  2. Pretrained first 2 layers loaded correctly
  3. Newly added layers 2-3 initialized (not identical to pretrained)
  4. PretrainDecoder output shape
  5. Stage 1 combined model forward + backward
  6. LightweightJointExpert forward pass and output shape
  7. MultiLightweightExpert with active subset
  8. Stage 2 expert training step (frozen encoder)
  9. Visible IMU sampling: changes across batches (train), fixed (val)
 10. Visible IMU sampling: valid per dataset
 11. Early stopping simulation
 12. Checkpoint save and load
 13. LightweightExpertOutputAssembler
 14. Activity-specific selective inference
 15. No CPU/GPU mismatch
"""
import argparse
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from shared_encoder_model import (
    SharedEncoder,
    PretrainDecoder,
    SharedEncoderPretrainModel,
    LightweightJointExpert,
    MultiLightweightExpert,
    LightweightExpertOutputAssembler,
    activity_joint_indices,
    selective_inference,
    EXPERT_RAW_OUTPUT_DIM,
)
from masked_dataset import sample_visible_set

N_IMUS = len(CANONICAL_IMUS)
N_JOINTS = len(CANONICAL_JOINTS)
D = 64
TARGET_DIM = EXPERT_RAW_OUTPUT_DIM  # 9
PRIOR_PATH = "pretrained/student_kl_18to21_best_64.pth"


def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg, err=None):
    print(f"  [FAIL] {msg}")
    if err:
        print(f"         {err}")
    return False


def test_encoder_forward(device):
    """1. SharedEncoder forward pass and output shape."""
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    B, T = 4, 32
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False  # keep first 5 IMUs
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    out = enc(imu, mask, lengths)
    assert out.shape == (B, T, N_JOINTS, D), \
        f"Expected {(B,T,N_JOINTS,D)}, got {out.shape}"
    assert out.device.type == device.type
    _ok(f"Encoder output shape {tuple(out.shape)} on {device}")
    return True


def test_pretrained_layers_loaded(device):
    """2. Pretrained first 2 layers loaded correctly."""
    if not os.path.exists(PRIOR_PATH):
        return _fail(f"Prior checkpoint not found: {PRIOR_PATH}")
    enc = SharedEncoder(d=D, num_layers=4).to("cpu")
    n = enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2, freeze=False)
    assert n == 2
    ok = enc.verify_pretrained(PRIOR_PATH, num_layers=2)
    assert ok, "Pretrained layers do not match checkpoint"
    _ok("Pretrained layers 0-1 match student_kl checkpoint")
    return True


def test_new_layers_different():
    """3. Newly added layers 2-3 are NOT identical to pretrained 0-1."""
    enc = SharedEncoder(d=D, num_layers=4)
    enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2, freeze=False)
    # layers 2-3 are randomly initialized; should differ from layers 0-1
    sd0 = enc.temporal_layers[0].state_dict()
    sd2 = enc.temporal_layers[2].state_dict()
    differs = any(
        not torch.equal(sd0[k], sd2[k]) for k in sd0 if k in sd2
    )
    assert differs, "Layer 2 should differ from pretrained layer 0"
    _ok("Layers 2-3 are freshly initialized (differ from pretrained)")
    return True


def test_decoder_output(device):
    """4. PretrainDecoder output shape."""
    dec = PretrainDecoder(d=D, num_layers=2, target_dim=TARGET_DIM).to(device)
    B, T = 4, 32
    enc_out = torch.randn(B, T, N_JOINTS, D, device=device)
    pred = dec(enc_out)
    assert pred.shape == (B, T, N_JOINTS, TARGET_DIM), \
        f"Expected {(B,T,N_JOINTS,TARGET_DIM)}, got {pred.shape}"
    _ok(f"Decoder output shape {tuple(pred.shape)}")
    return True


def test_stage1_training_step(device):
    """5. Stage 1 combined model forward + backward."""
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    if os.path.exists(PRIOR_PATH):
        enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2)
    dec = PretrainDecoder(d=D, num_layers=2, target_dim=TARGET_DIM).to(device)
    model = SharedEncoderPretrainModel(enc, dec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, T = 4, 32
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    target = torch.randn(B, T, N_JOINTS, TARGET_DIM, device=device)

    pred = model(imu, mask, lengths)
    loss = (pred - target).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    _ok(f"Stage 1 training step: loss={float(loss):.4f}")
    return True


def test_expert_forward(device):
    """6. LightweightJointExpert forward pass and output shape."""
    expert = LightweightJointExpert(d=D, num_layers=2,
                                    target_dim=TARGET_DIM).to(device)
    B, T = 4, 32
    enc_out = torch.randn(B, T, N_JOINTS, D, device=device)
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    out = expert(enc_out, imu, mask, lengths)
    assert out.shape == (B, T, TARGET_DIM), \
        f"Expected {(B,T,TARGET_DIM)}, got {out.shape}"
    _ok(f"Expert output shape {tuple(out.shape)} (full encoder + masked IMU)")

    # check forward_dict
    d = expert.forward_dict(enc_out, imu, mask, lengths)
    assert d["orientation_6d"].shape == (B, T, 6)
    assert d["motion_delta"].shape == (B, T, 3)
    _ok("Expert forward_dict: orientation_6d [B,T,6], motion_delta [B,T,3]")
    return True


def test_multi_expert_active(device):
    """7. MultiLightweightExpert with active subset."""
    multi = MultiLightweightExpert(n_joints=N_JOINTS, d=D,
                                   num_layers=2).to(device)
    B, T = 4, 32
    shared = torch.randn(B, T, N_JOINTS, D, device=device)
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False
    lengths = torch.full((B,), T, dtype=torch.long, device=device)

    # all experts
    out_all = multi(shared, imu, mask, lengths)
    assert out_all.shape == (B, T, N_JOINTS, TARGET_DIM)

    # selected experts
    active = [0, 3, 7, 20]
    out_sel = multi(shared, imu, mask, lengths, active=active)
    assert out_sel.shape == (B, T, len(active), TARGET_DIM)
    _ok(f"MultiExpert: all={tuple(out_all.shape)}, active({len(active)})={tuple(out_sel.shape)}")
    return True


def test_stage2_training_step(device):
    """8. Stage 2 expert training step (frozen encoder)."""
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    if os.path.exists(PRIOR_PATH):
        enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    expert = LightweightJointExpert(d=D, num_layers=2,
                                    target_dim=TARGET_DIM).to(device)
    opt = torch.optim.AdamW(expert.parameters(), lr=1e-4)

    B, T = 4, 32
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    target_j = torch.randn(B, T, TARGET_DIM, device=device)

    with torch.no_grad():
        shared = enc(imu, mask, lengths)  # [B, T, 28, d]
    pred = expert(shared, imu, mask, lengths)  # full context -> joint output
    loss = (pred - target_j).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # verify encoder params unchanged
    enc_grad = any(p.grad is not None for p in enc.parameters())
    assert not enc_grad, "Encoder should have no gradients (frozen)"
    _ok(f"Stage 2 training step: loss={float(loss):.4f}, encoder frozen")
    return True


def test_visible_sampling_train_changes():
    """9. Training visible set changes across batches."""
    available = list(range(N_IMUS))
    sets = []
    for batch_idx in range(5):
        s = sample_visible_set("test_source", epoch=0, batch_idx=batch_idx,
                               available_imus=available, min_k=2, max_k=5)
        sets.append(tuple(s))
    unique = len(set(sets))
    assert unique > 1, f"Expected different sets across batches, got {unique} unique"
    _ok(f"Train visible sampling: {unique} different sets across 5 batches")

    # fixed for val
    sets_val = []
    for batch_idx in range(5):
        s = sample_visible_set("test_source", epoch=0, batch_idx=0,
                               available_imus=available, split="val",
                               min_k=2, max_k=5)
        sets_val.append(tuple(s))
    assert len(set(sets_val)) == 1, "Val sets should be identical"
    _ok("Val visible sampling: identical across calls")
    return True


def test_visible_sampling_valid_per_dataset():
    """10. Visible sets are valid per dataset (no unavailable IMUs)."""
    partial = [0, 1, 5, 6, 10]  # only 5 IMUs available
    for i in range(20):
        s = sample_visible_set("partial_ds", epoch=i, batch_idx=i,
                               available_imus=partial, min_k=2, max_k=5)
        for imu in s:
            assert imu in partial, f"Sampled unavailable IMU {imu}"
        assert 2 <= len(s) <= 5
    _ok("Visible sampling: all sampled IMUs are available, k in [2,5]")
    return True


def test_early_stopping_simulation():
    """11. Early stopping simulation: two experts converge at different epochs."""
    patience = 3
    best = [float("inf"), float("inf")]
    since = [0, 0]
    converged = [False, False]

    fake_losses = [
        # epoch: expert0, expert1
        [10.0, 10.0],
        [9.0, 9.5],
        [8.5, 9.3],
        [8.5, 9.0],  # expert0 stalls
        [8.5, 8.8],
        [8.5, 8.5],  # expert0 patience=3 -> converge
        [8.5, 8.5],
        [8.5, 8.5],
        [8.5, 8.5],  # expert1 patience=3 -> converge
    ]
    converge_epochs = [None, None]
    for ep, losses in enumerate(fake_losses):
        for j in range(2):
            if converged[j]:
                continue
            if losses[j] < best[j] - 1e-3:
                best[j] = losses[j]
                since[j] = 0
            else:
                since[j] += 1
            if since[j] >= patience:
                converged[j] = True
                converge_epochs[j] = ep + 1
    assert converge_epochs[0] is not None and converge_epochs[1] is not None
    assert converge_epochs[0] < converge_epochs[1], \
        f"Expert 0 should converge before expert 1: {converge_epochs}"
    _ok(f"Early stopping: expert0@epoch{converge_epochs[0]}, "
        f"expert1@epoch{converge_epochs[1]}")
    return True


def test_checkpoint_save_load(device):
    """12. Checkpoint save and load."""
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    if os.path.exists(PRIOR_PATH):
        enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2)
    dec = PretrainDecoder(d=D, num_layers=2, target_dim=TARGET_DIM).to(device)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "epoch": 5,
            "val_orient_deg": 12.34,
            "args": {"hidden": D, "encoder_layers": 4, "nhead": 4, "ff": 128},
            "imus": CANONICAL_IMUS,
            "joints": CANONICAL_JOINTS,
        }, path)

        ckpt = torch.load(path, map_location="cpu")
        enc2 = SharedEncoder(d=D, num_layers=4)
        enc2.load_state_dict(ckpt["encoder"])
        dec2 = PretrainDecoder(d=D, num_layers=2, target_dim=TARGET_DIM)
        dec2.load_state_dict(ckpt["decoder"])

        # verify parameters match
        for (n1, p1), (n2, p2) in zip(enc.named_parameters(),
                                       enc2.named_parameters()):
            assert torch.equal(p1.cpu(), p2.cpu()), f"Mismatch at {n1}"
        assert ckpt["epoch"] == 5
        assert ckpt["val_orient_deg"] == 12.34
        _ok("Checkpoint save/load: encoder+decoder params match")
    finally:
        os.unlink(path)
    return True


def test_expert_checkpoint_save_load(device):
    """12b. Expert checkpoint save and load."""
    expert = LightweightJointExpert(d=D, num_layers=2,
                                    target_dim=TARGET_DIM).to(device)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({
            "model": expert.state_dict(),
            "joint_idx": 5,
            "joint_name": "LeftFoot",
            "epoch": 42,
            "val_orient_deg": 8.76,
        }, path)
        ckpt = torch.load(path, map_location="cpu")
        expert2 = LightweightJointExpert(d=D, num_layers=2,
                                         target_dim=TARGET_DIM)
        expert2.load_state_dict(ckpt["model"])
        for (n1, p1), (n2, p2) in zip(expert.named_parameters(),
                                       expert2.named_parameters()):
            assert torch.equal(p1.cpu(), p2.cpu()), f"Mismatch at {n1}"
        _ok("Expert checkpoint save/load: params match")
    finally:
        os.unlink(path)
    return True


def test_assembler(device):
    """13. LightweightExpertOutputAssembler."""
    asm = LightweightExpertOutputAssembler(N_JOINTS, TARGET_DIM).to(device)
    B, T, k = 4, 32, 5
    selected = [0, 3, 7, 20, 24]
    expert_out = torch.randn(B, T, k, TARGET_DIM, device=device)
    dense, valid = asm(expert_out, selected=selected)
    assert dense.shape == (B, T, N_JOINTS, TARGET_DIM)
    assert valid.shape == (B, N_JOINTS)
    assert valid[:, selected].all()
    assert not valid[:, [1, 2, 4, 5]].any()
    _ok(f"Assembler: dense {tuple(dense.shape)}, "
        f"valid at selected={selected}")
    return True


def test_selective_inference(device):
    """14. Activity-specific selective inference."""
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    multi = MultiLightweightExpert(n_joints=N_JOINTS, d=D,
                                    num_layers=2).to(device)
    asm = LightweightExpertOutputAssembler(N_JOINTS, TARGET_DIM).to(device)

    squat_joints = activity_joint_indices("squat")
    B, T = 2, 16
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    mask[:, 5:] = False
    lengths = torch.full((B,), T, dtype=torch.long, device=device)

    t0 = time.time()
    result = selective_inference(enc, multi, imu, mask, squat_joints,
                                lengths, assembler=asm)
    elapsed = time.time() - t0

    assert result["expert_outputs"].shape == (B, T, len(squat_joints), TARGET_DIM)
    assert result["dense_tokens"].shape == (B, T, N_JOINTS, TARGET_DIM)
    assert len(result["selected_joints"]) == len(squat_joints)
    _ok(f"Selective inference (squat, {len(squat_joints)} experts): "
        f"{elapsed*1000:.1f}ms, "
        f"output {tuple(result['expert_outputs'].shape)}")
    return True


def test_no_cpu_gpu_mismatch(device):
    """15. No CPU/GPU mismatch during forward pass."""
    if device.type != "cuda":
        _ok("Skipped (CPU only)")
        return True
    enc = SharedEncoder(d=D, num_layers=4).to(device)
    if os.path.exists(PRIOR_PATH):
        enc.load_pretrained(PRIOR_PATH, num_layers_to_load=2)
    dec = PretrainDecoder(d=D, num_layers=2, target_dim=TARGET_DIM).to(device)
    model = SharedEncoderPretrainModel(enc, dec)

    B, T = 2, 16
    imu = torch.randn(B, T, N_IMUS, 12, device=device)
    mask = torch.ones(B, N_IMUS, dtype=torch.bool, device=device)
    lengths = torch.full((B,), T, dtype=torch.long, device=device)

    # should not raise any device mismatch errors
    pred = model(imu, mask, lengths)
    target = torch.randn_like(pred)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    # check all model params are on correct device
    for name, p in model.named_parameters():
        assert p.device.type == device.type, \
            f"Parameter {name} on {p.device}, expected {device}"
    # check buffers too
    for name, b in model.named_buffers():
        assert b.device.type == device.type, \
            f"Buffer {name} on {b.device}, expected {device}"
    _ok(f"No CPU/GPU mismatch: all params/buffers on {device}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")
    print()

    tests = [
        ("1. Encoder forward pass", test_encoder_forward),
        ("2. Pretrained layers loaded", test_pretrained_layers_loaded),
        ("3. New layers differ from pretrained", test_new_layers_different),
        ("4. Decoder output shape", test_decoder_output),
        ("5. Stage 1 training step", test_stage1_training_step),
        ("6. Expert forward pass", test_expert_forward),
        ("7. MultiExpert active subset", test_multi_expert_active),
        ("8. Stage 2 training step (frozen encoder)", test_stage2_training_step),
        ("9. Visible sampling changes (train) / fixed (val)",
         test_visible_sampling_train_changes),
        ("10. Visible sampling valid per dataset",
         test_visible_sampling_valid_per_dataset),
        ("11. Early stopping simulation", test_early_stopping_simulation),
        ("12a. Encoder+decoder checkpoint save/load", test_checkpoint_save_load),
        ("12b. Expert checkpoint save/load", test_expert_checkpoint_save_load),
        ("13. Expert output assembler", test_assembler),
        ("14. Selective inference (activity-specific)", test_selective_inference),
        ("15. No CPU/GPU mismatch", test_no_cpu_gpu_mismatch),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            # some tests don't need device arg
            import inspect
            sig = inspect.signature(fn)
            if sig.parameters:
                result = fn(device)
            else:
                result = fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            _fail(name, str(e))
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
