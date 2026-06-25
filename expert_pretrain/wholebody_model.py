"""ViPoser Stage-2: whole-body recovery (Global Poser, mobicom26-paper1003 4.2.2).

A TEMPORAL transformer over per-frame tokens. Each frame token concatenates two
streams, both whole-body with TRAINABLE mask embeddings at the missing slots:
  - joint stream: all 28 joints' target vectors; unknown joints (no expert active) are
    replaced by a learned per-joint placeholder,
  - IMU stream:   raw 15-IMU values mapped onto the canonical 28-joint grid; absent
    joint slots are replaced by a learned IMU mask token,
plus the two binary masks. The token is projected to d=64 and processed by
  encoder = the 2-layer/64-d student Transformer distilled from Sapiens layers
            18-21 (student_kl, FROZEN) with LoRA adapters on its linears,
  decoder = 2 fully-trained layers of the same shape ("4-64" total),
and a head regresses every joint's target vector per frame.

Phase-I feeds GROUND-TRUTH targets at the known joints, so it needs no experts.
Phase-II swaps in the experts' predictions.
"""
import math
from typing import List, Optional

import torch
from torch import nn

from multiexpert_model import SparseIMUJointTokenizer
from schema import CANONICAL_IMUS, CANONICAL_JOINTS

RECOVERY_ORIENTATION_DIM = 6
RECOVERY_MOTION_DELTA_DIM = 3
RECOVERY_RAW_OUTPUT_DIM = RECOVERY_ORIENTATION_DIM + RECOVERY_MOTION_DELTA_DIM


def split_recovery_prediction(pred: torch.Tensor):
    if pred.shape[-1] < RECOVERY_RAW_OUTPUT_DIM:
        raise ValueError(f"Recovery output dim must be at least {RECOVERY_RAW_OUTPUT_DIM}, got {pred.shape[-1]}")
    out = {
        "pred_full_body_orientation_6d": pred[..., :RECOVERY_ORIENTATION_DIM],
        "pred_full_body_motion_delta": pred[..., RECOVERY_ORIENTATION_DIM:RECOVERY_RAW_OUTPUT_DIM],
    }
    if pred.shape[-1] > RECOVERY_RAW_OUTPUT_DIM:
        out["extra"] = pred[..., RECOVERY_RAW_OUTPUT_DIM:]
    return out


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update (BA)."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    # nn.TransformerEncoderLayer's eval fast-path reads .weight/.bias directly,
    # bypassing forward(); expose the EFFECTIVE merged weight so it stays correct.
    @property
    def weight(self):
        return self.base.weight + (self.B @ self.A) * self.scale

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def _encoder_layer(d: int, nhead: int, ff: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                      batch_first=True, dropout=0.1)


def load_student_prior(path: str, d: int = 64, nhead: int = 4, ff: int = 128):
    """Load the 2-layer student_kl prior (weights map 1:1 onto
    nn.TransformerEncoderLayer: self_attn/linear1/linear2/norm1/norm2)."""
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    layers = nn.ModuleList([_encoder_layer(d, nhead, ff) for _ in range(2)])
    for li in range(2):
        prefix = f"student.layers.{li}."
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = layers[li].load_state_dict(sub, strict=False)
        assert not [m for m in missing if "self_attn" in m or "linear" in m or "norm" in m], \
            f"prior layer {li} missing weights: {missing}"
    return layers


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # x [B,T,d]
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class WholeBodyPoser(nn.Module):
    def __init__(self, prior_path: str, n_joints: int = len(CANONICAL_JOINTS),
                 n_imus: int = len(CANONICAL_IMUS), imu_dim: int = 12,
                 d: int = 64, nhead: int = 4, ff: int = 128, lora_r: int = 8,
                 dec_layers: int = 2, target_dim: int = RECOVERY_RAW_OUTPUT_DIM, max_len: int = 1024,
                 train_prior: str = "ln_lora", residual: bool = True):
        super().__init__()
        self.n_joints, self.n_imus = n_joints, n_imus
        self.residual = residual
        # Separate trainable mask embeddings for missing expert-output and IMU slots.
        self.joint_fill = nn.Parameter(torch.zeros(n_joints, target_dim))
        self.imu_tokenizer = SparseIMUJointTokenizer(n_imus, n_joints, imu_dim=imu_dim)
        self.recovery_input_dim = target_dim + 1 + imu_dim + 1
        in_dim = n_joints * self.recovery_input_dim
        self.input_proj = nn.Linear(in_dim, d)
        self.pe = SinusoidalPE(d, max_len)

        # encoder = Sapiens-distilled prior. train_prior controls adaptation:
        #   "lora"    frozen + LoRA on the FFN linears,
        #   "ln_lora" LoRA + trainable LayerNorms (frozen-transformer transfer
        #             needs the norms to adapt to the new token statistics),
        #   "full"    everything trainable (prior as initialization only).
        self.prior = load_student_prior(prior_path, d=d, nhead=nhead, ff=ff)
        if train_prior != "full":
            for p in self.prior.parameters():
                p.requires_grad_(False)
            for layer in self.prior:
                layer.linear1 = LoRALinear(layer.linear1, r=lora_r)
                layer.linear2 = LoRALinear(layer.linear2, r=lora_r)
                if train_prior == "ln_lora":
                    for p in layer.norm1.parameters():
                        p.requires_grad_(True)
                    for p in layer.norm2.parameters():
                        p.requires_grad_(True)

        # decoder = fully-trained layers + whole-body head
        self.decoder = nn.ModuleList([_encoder_layer(d, nhead, ff) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_joints * target_dim))

    def make_recovery_tokens(self, jr6d: torch.Tensor, known: torch.Tensor,
                             imu: torch.Tensor, imu_mask: torch.Tensor):
        """Build per-joint recovery input [B,T,28,D] from IMU + expert tokens."""
        b, t, j, _ = jr6d.shape
        if j != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joints, got {j}")
        kj = known.view(b, 1, j, 1).to(jr6d.dtype)
        expert_tokens = jr6d * kj + self.joint_fill.view(1, 1, j, -1) * (1 - kj)
        imu_tokens, visible_imu_joints = self.imu_tokenizer(imu, imu_mask)
        ki = visible_imu_joints.view(b, 1, j, 1).to(imu.dtype).expand(b, t, j, 1)
        recovery_input = torch.cat([
            expert_tokens,
            kj.expand(b, t, j, 1),
            imu_tokens,
            ki,
        ], dim=-1)
        return recovery_input, expert_tokens, imu_tokens, visible_imu_joints

    def forward(self, jr6d: torch.Tensor, known: torch.Tensor,
                imu: torch.Tensor, imu_mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """jr6d [B,T,J,D] (expert outputs at known joints; rest ignored), known [B,J] bool,
        imu [B,T,S,12], imu_mask [B,S] bool -> target [B,T,J,D] for ALL joints."""
        b, t, j, _ = jr6d.shape
        recovery_input, expert_tokens, _, _ = self.make_recovery_tokens(jr6d, known, imu, imu_mask)
        x = self.pe(self.input_proj(recovery_input.reshape(b, t, -1)))
        pad = None
        if lengths is not None:
            pad = torch.arange(t, device=x.device).view(1, t) >= lengths.view(-1, 1)
        for layer in self.prior:
            x = layer(x, src_key_padding_mask=pad)
        for layer in self.decoder:
            x = layer(x, src_key_padding_mask=pad)
        out = self.head(x).view(b, t, j, -1)
        if self.residual:
            # complete-the-input bias: known joints start at their input value,
            # unknown ones at their learned placeholder; the net adds a correction.
            out = out + expert_tokens
        return out

    def forward_dict(self, jr6d: torch.Tensor, known: torch.Tensor,
                     imu: torch.Tensor, imu_mask: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None):
        pred = self.forward(jr6d, known, imu, imu_mask, lengths)
        return split_recovery_prediction(pred)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
