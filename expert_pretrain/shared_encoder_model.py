"""Shared sparse-IMU encoder + lightweight joint-specific experts.

Architecture:
  SharedEncoder: SparseIMUJointTokenizer (15 IMU -> 28 joint tokens with learned
      mask embeddings) -> input projection + joint-ID embedding -> spatial context
      (mean-pool + broadcast) -> per-joint temporal Transformer (shared weights,
      [B*28, T, d]).  Layers 0-1 initialized from Sapiens-distilled student_kl
      prior; layers 2-3 fresh.  Output: [B, T, 28, d].

  PretrainDecoder (Stage 1): 2-layer per-frame spatial self-attention over the
      28 per-joint encoder representations, then a target head.  Reconstructs
      full-body pose: [B, T, 28, target_dim].

  LightweightJointExpert (Stage 2): 2-layer temporal Transformer + head.  Reads
      a single joint's frozen encoder output [B, T, d] and predicts that joint's
      9D target (6D orientation + 3D motion delta).
"""
import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from multiexpert_model import SparseIMUJointTokenizer, split_expert_prediction
from schema import CANONICAL_IMUS, CANONICAL_JOINTS

EXPERT_ORIENTATION_DIM = 6
EXPERT_MOTION_DELTA_DIM = 3
EXPERT_RAW_OUTPUT_DIM = EXPERT_ORIENTATION_DIM + EXPERT_MOTION_DELTA_DIM


# ---------------------------------------------------------------------------
# Pretrained prior loading (reused from wholebody_model.py)
# ---------------------------------------------------------------------------

def _encoder_layer(d: int, nhead: int, ff: int, dropout: float = 0.1
                   ) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d, nhead=nhead, dim_feedforward=ff,
        batch_first=True, dropout=dropout,
    )


def load_student_layers(path: str, d: int = 64, nhead: int = 4, ff: int = 128,
                        num_layers: int = 2) -> nn.ModuleList:
    """Load the student_kl prior layers (1:1 map to TransformerEncoderLayer)."""
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    layers = nn.ModuleList([_encoder_layer(d, nhead, ff) for _ in range(num_layers)])
    for li in range(num_layers):
        prefix = f"student.layers.{li}."
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = layers[li].load_state_dict(sub, strict=False)
        core_missing = [m for m in missing
                        if "self_attn" in m or "linear" in m or "norm" in m]
        assert not core_missing, (
            f"Prior layer {li} missing core weights: {core_missing}"
        )
    return layers


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]].unsqueeze(0)


# ---------------------------------------------------------------------------
# SharedEncoder
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Sparse-IMU -> per-joint temporal representation [B, T, 28, d].

    Two modes controlled by ``mask_position``:

    **"before"** (mask tokens enter encoder):
      1. SparseIMUJointTokenizer: 15 IMUs -> 28 joint tokens (visible = real,
         absent = learned imu_mask_token).
      2. Linear(imu_dim+1, d) + Embedding(n_joints, d) per token.
      3. Per-frame spatial self-attention (2 layers) over all 28 tokens.
      4. Per-joint temporal (4 layers, shared weights) over all 28.

    **"after"** (encoder sees only visible IMUs):
      1. Work directly on 15 IMU slots — no tokenizer expansion.
      2. Linear(imu_dim, d) + Embedding(n_imus, d) per IMU slot.
      3. Per-frame spatial self-attention (2 layers) over 15 slots with
         src_key_padding_mask excluding non-visible IMUs.
      4. Per-IMU temporal (4 layers, shared weights) over 15 slots.
      5. Scatter visible IMU outputs to their 28 joint positions; fill all
         other positions with per-joint learned mask tokens [n_joints, d].
    """

    def __init__(self, n_imus: int = len(CANONICAL_IMUS),
                 n_joints: int = len(CANONICAL_JOINTS),
                 imu_dim: int = 12, d: int = 64, nhead: int = 4,
                 ff: int = 128, num_layers: int = 4,
                 num_spatial_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 1024,
                 mask_position: str = "before"):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints
        self.d = d
        self.num_layers = num_layers
        self.num_spatial_layers = num_spatial_layers
        self.mask_position = mask_position

        if mask_position == "before":
            self.imu_tokenizer = SparseIMUJointTokenizer(
                n_imus, n_joints, imu_dim=imu_dim,
            )
            self.sensor_value = nn.Linear(imu_dim + 1, d)
            self.sensor_id = nn.Embedding(n_joints, d)
        else:
            self.imu_proj = nn.Linear(imu_dim, d)
            self.imu_embed = nn.Embedding(n_imus, d)
            self.imu_mask_embed = nn.Parameter(torch.randn(n_imus, imu_dim) * 0.02)
            from multiexpert_model import CANONICAL_IMU_TO_JOINT
            self.register_buffer(
                "imu_to_joint",
                torch.tensor(CANONICAL_IMU_TO_JOINT[:n_imus], dtype=torch.long),
            )
            self.post_mask_tokens = nn.Parameter(torch.randn(n_joints, d) * 0.02)

        self.spatial_layers = nn.ModuleList([
            _encoder_layer(d, nhead, ff, dropout)
            for _ in range(num_spatial_layers)
        ])
        self.pe = SinusoidalPE(d, max_len)
        self.temporal_layers = nn.ModuleList([
            _encoder_layer(d, nhead, ff, dropout) for _ in range(num_layers)
        ])

    # -- pretrained initialization ------------------------------------------

    def load_pretrained(self, path: str, num_layers_to_load: int = 2,
                        freeze: bool = False):
        """Replace the first `num_layers_to_load` temporal layers with weights
        from the student_kl checkpoint, optionally freezing them."""
        pretrained = load_student_layers(
            path, d=self.d,
            nhead=self.temporal_layers[0].self_attn.num_heads,
            ff=self.temporal_layers[0].linear1.out_features,
            num_layers=num_layers_to_load,
        )
        for li in range(num_layers_to_load):
            self.temporal_layers[li].load_state_dict(
                pretrained[li].state_dict(),
            )
            if freeze:
                for p in self.temporal_layers[li].parameters():
                    p.requires_grad_(False)
        return num_layers_to_load

    def verify_pretrained(self, path: str, num_layers: int = 2) -> bool:
        """Check that the first `num_layers` match the checkpoint."""
        ref = load_student_layers(
            path, d=self.d,
            nhead=self.temporal_layers[0].self_attn.num_heads,
            ff=self.temporal_layers[0].linear1.out_features,
            num_layers=num_layers,
        )
        for li in range(num_layers):
            for (n1, p1), (n2, p2) in zip(
                self.temporal_layers[li].named_parameters(),
                ref[li].named_parameters(),
            ):
                if n1 != n2 or not torch.equal(p1.data.cpu(), p2.data.cpu()):
                    return False
        return True

    # -- forward -------------------------------------------------------------

    def _forward_before(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        joint_imu, visible = self.imu_tokenizer(imu, imu_mask)
        b, t, j, _ = joint_imu.shape
        present = visible.to(imu.dtype).view(b, 1, j, 1).expand(b, t, j, 1)
        x = self.sensor_value(torch.cat([joint_imu, present], dim=-1))
        x = x + self.sensor_id(
            torch.arange(j, device=imu.device),
        ).view(1, 1, j, -1)

        x = x.reshape(b * t, j, self.d)
        for layer in self.spatial_layers:
            x = layer(x)
        x = x.reshape(b, t, j, self.d)

        x = x.permute(0, 2, 1, 3).reshape(b * j, t, self.d)
        x = self.pe(x)
        pad = None
        if lengths is not None:
            pad = (torch.arange(t, device=x.device).view(1, t)
                   >= lengths.view(-1, 1))
            pad = pad.repeat_interleave(j, dim=0)
        for layer in self.temporal_layers:
            x = layer(x, src_key_padding_mask=pad)

        return x.reshape(b, j, t, self.d).permute(0, 2, 1, 3)

    def _forward_after(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                       lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, s, _ = imu.shape  # s = 15 IMU slots
        vis = imu_mask.bool()   # [B, 15]

        # replace non-visible IMU slots with per-IMU learnable mask at raw [12] level
        imu_in = imu.clone()
        vis_raw = vis.view(b, 1, s, 1).expand(b, t, s, imu.shape[-1])
        mask_raw = self.imu_mask_embed.view(1, 1, s, imu.shape[-1]).expand(b, t, s, imu.shape[-1])
        imu_in = torch.where(vis_raw, imu_in, mask_raw)

        # project all 15 slots (real data + mask tokens) together
        x = self.imu_proj(imu_in)  # [B, T, 15, d]
        x = x + self.imu_embed(
            torch.arange(s, device=imu.device),
        ).view(1, 1, s, -1)

        # spatial self-attention on 15 IMU slots, non-visible masked out
        inv = ~vis  # [B, 15] True = non-visible
        spatial_pad = inv.unsqueeze(1).expand(b, t, s).reshape(b * t, s)
        x = x.reshape(b * t, s, self.d)
        for layer in self.spatial_layers:
            x = layer(x, src_key_padding_mask=spatial_pad)
        x = x.reshape(b, t, s, self.d)

        # per-IMU temporal processing (shared weights across 15 slots)
        x = x.permute(0, 2, 1, 3).reshape(b * s, t, self.d)
        x = self.pe(x)
        pad = None
        if lengths is not None:
            pad = (torch.arange(t, device=x.device).view(1, t)
                   >= lengths.view(-1, 1))
            pad = pad.repeat_interleave(s, dim=0)
        for layer in self.temporal_layers:
            x = layer(x, src_key_padding_mask=pad)
        x = x.reshape(b, s, t, self.d).permute(0, 2, 1, 3)  # [B, T, 15, d]

        # scatter visible IMU outputs to 28 joint positions
        j = self.n_joints
        out = self.post_mask_tokens.view(1, 1, j, self.d).expand(b, t, j, self.d).clone()
        jidx = self.imu_to_joint  # [15] — which joint each IMU maps to
        vis_exp = vis.view(b, 1, s, 1).expand(b, t, s, self.d)
        mask_fallback = self.post_mask_tokens[jidx].view(1, 1, s, self.d).expand(b, t, s, self.d)
        out[:, :, jidx] = torch.where(vis_exp, x, mask_fallback)

        return out

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """imu [B,T,15,12], imu_mask [B,15] -> per-joint repr [B,T,28,d]."""
        if self.mask_position == "before":
            return self._forward_before(imu, imu_mask, lengths)
        return self._forward_after(imu, imu_mask, lengths)


# ---------------------------------------------------------------------------
# PretrainDecoder (Stage 1)
# ---------------------------------------------------------------------------

class PretrainDecoder(nn.Module):
    """Decode per-joint encoder output [B,T,28,d] -> [B,T,28,target_dim].

    Per-frame spatial self-attention lets joints exchange kinematic information
    after independent temporal processing, then a linear head projects to the
    target dimension.
    """

    def __init__(self, n_joints: int = len(CANONICAL_JOINTS), d: int = 64,
                 nhead: int = 4, ff: int = 128, num_layers: int = 2,
                 target_dim: int = EXPERT_RAW_OUTPUT_DIM, dropout: float = 0.1):
        super().__init__()
        self.n_joints = n_joints
        self.d = d
        self.target_dim = target_dim
        self.decoder_layers = nn.ModuleList([
            _encoder_layer(d, nhead, ff, dropout) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, target_dim),
        )

    def forward(self, encoder_output: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """encoder_output [B,T,28,d] -> [B,T,n_joints,target_dim]."""
        b, t, j, d = encoder_output.shape
        x = encoder_output.reshape(b * t, j, d)
        for layer in self.decoder_layers:
            x = layer(x)
        pred = self.head(x)
        return pred.reshape(b, t, j, self.target_dim)


# ---------------------------------------------------------------------------
# Stage 1 combined model
# ---------------------------------------------------------------------------

class SharedEncoderPretrainModel(nn.Module):
    """Encoder + decoder for Stage 1 full-body pretraining."""

    def __init__(self, encoder: SharedEncoder, decoder: PretrainDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        shared = self.encoder(imu, imu_mask, lengths)
        return self.decoder(shared, lengths)

    def forward_dict(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        pred = self.forward(imu, imu_mask, lengths)
        return split_expert_prediction(pred)


# ---------------------------------------------------------------------------
# LightweightJointExpert (Stage 2)
# ---------------------------------------------------------------------------

class LightweightJointExpert(nn.Module):
    """Per-joint expert with access to the FULL shared context.

    To give every joint (including ones whose own IMU is absent — the failure
    mode of ``mask_position="after"``, where most encoder tokens collapse to a
    constant mask token) full cross-joint / cross-sensor knowledge, each expert
    reads BOTH:

      * the full encoder output    [B, T, n_joints, d]
      * the full masked raw IMU    [B, T, n_imus, imu_dim]  (+ visibility mask)

    A single learnable joint query cross-attends over all (n_joints + n_imus)
    context tokens per frame, then a temporal Transformer + head predicts this
    joint's 9D target (6D orientation + 3D motion delta).  Output: [B, T, dim].

    Works identically for both ``before`` and ``after`` encoders, so the same
    Stage-2 code trains experts on either Stage-1 variant.
    """

    def __init__(self, d: int = 64, nhead: int = 4, ff: int = 128,
                 num_layers: int = 2, target_dim: int = EXPERT_RAW_OUTPUT_DIM,
                 dropout: float = 0.1, n_joints: int = len(CANONICAL_JOINTS),
                 n_imus: int = len(CANONICAL_IMUS), imu_dim: int = 12,
                 max_len: int = 1024):
        super().__init__()
        self.d = d
        self.target_dim = target_dim
        self.n_joints = n_joints
        self.n_imus = n_imus
        self.imu_dim = imu_dim
        # raw masked-IMU projection into token space + positional tags so the
        # cross-attention query can tell tokens apart (attention is order-free).
        self.imu_proj = nn.Linear(imu_dim, d)
        self.imu_embed = nn.Embedding(n_imus, d)
        self.joint_embed = nn.Embedding(n_joints, d)
        # this expert's learnable query over the full per-frame context
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.context_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff,
            batch_first=True, dropout=dropout,
        )
        self.pe = SinusoidalPE(d, max_len)
        self.temporal = nn.TransformerEncoder(
            _encoder_layer(d, nhead, ff, dropout),
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d, target_dim),
        )

    def forward(self, encoder_output: torch.Tensor, imu: torch.Tensor,
                imu_mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """encoder_output [B,T,J,d], imu [B,T,S,imu_dim], imu_mask [B,S]
        -> [B,T,target_dim] for this expert's joint."""
        b, t, j, d = encoder_output.shape
        s = imu.shape[2]
        enc = encoder_output + self.joint_embed(
            torch.arange(j, device=encoder_output.device),
        ).view(1, 1, j, d)
        imu_tok = self.imu_proj(imu) + self.imu_embed(
            torch.arange(s, device=imu.device),
        ).view(1, 1, s, d)
        context = torch.cat([enc, imu_tok], dim=2).reshape(b * t, j + s, d)
        # mask non-visible IMU tokens out of cross-attention (zeros carry no
        # signal); encoder tokens are always attendable so memory is never empty.
        mem_pad = torch.zeros(b, t, j + s, dtype=torch.bool, device=imu.device)
        mem_pad[:, :, j:] = ~imu_mask.bool().view(b, 1, s)
        mem_pad = mem_pad.reshape(b * t, j + s)
        query = self.query.view(1, 1, d).expand(b * t, 1, d)
        ctx = self.context_layer(query, context, memory_key_padding_mask=mem_pad)
        x = self.pe(ctx.reshape(b, t, d))
        pad = None
        if lengths is not None:
            pad = (torch.arange(t, device=x.device).view(1, t)
                   >= lengths.view(-1, 1))
        x = self.temporal(x, src_key_padding_mask=pad)
        return self.head(x)

    def forward_dict(self, encoder_output: torch.Tensor, imu: torch.Tensor,
                     imu_mask: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        pred = self.forward(encoder_output, imu, imu_mask, lengths)
        return split_expert_prediction(pred)


# ---------------------------------------------------------------------------
# Multi-expert container
# ---------------------------------------------------------------------------

class MultiLightweightExpert(nn.Module):
    """28 lightweight joint experts sharing a frozen encoder.

    Each expert reads the full encoder output + full masked IMU (see
    :class:`LightweightJointExpert`), so they are interchangeable across
    ``before``/``after`` Stage-1 encoders.
    """

    def __init__(self, n_joints: int = len(CANONICAL_JOINTS), d: int = 64,
                 nhead: int = 4, ff: int = 128, num_layers: int = 2,
                 target_dim: int = EXPERT_RAW_OUTPUT_DIM, dropout: float = 0.1,
                 n_imus: int = len(CANONICAL_IMUS), imu_dim: int = 12):
        super().__init__()
        self.n_joints = n_joints
        self.target_dim = target_dim
        self.experts = nn.ModuleList([
            LightweightJointExpert(d, nhead, ff, num_layers, target_dim, dropout,
                                   n_joints=n_joints, n_imus=n_imus, imu_dim=imu_dim)
            for _ in range(n_joints)
        ])

    def forward(self, encoder_output: torch.Tensor, imu: torch.Tensor,
                imu_mask: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                active: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Run selected (or all) experts on the full shared context.

        encoder_output [B,T,28,d], imu [B,T,15,imu_dim], imu_mask [B,15].
        Returns [B, T, len(active), target_dim].
        """
        idxs = range(self.n_joints) if active is None else active
        outs = [self.experts[j](encoder_output, imu, imu_mask, lengths)
                for j in idxs]
        return torch.stack(outs, dim=2)

    def forward_dict(self, encoder_output: torch.Tensor, imu: torch.Tensor,
                     imu_mask: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None,
                     active: Optional[Sequence[int]] = None
                     ) -> Dict[str, torch.Tensor]:
        pred = self.forward(encoder_output, imu, imu_mask, lengths, active)
        return split_expert_prediction(pred)


# ---------------------------------------------------------------------------
# Expert output assembly (for recovery model compatibility)
# ---------------------------------------------------------------------------

class LightweightExpertOutputAssembler(nn.Module):
    """Pack selected lightweight expert outputs into dense [B,T,28,D] tokens."""

    def __init__(self, n_joints: int = len(CANONICAL_JOINTS),
                 expert_dim: int = EXPERT_RAW_OUTPUT_DIM):
        super().__init__()
        self.n_joints = n_joints
        self.expert_dim = expert_dim
        self.mask_token = nn.Parameter(torch.zeros(n_joints, expert_dim))

    def forward(self, expert_outputs: torch.Tensor,
                selected: Optional[Sequence[int]] = None
                ) -> tuple:
        """expert_outputs [B,T,k,D] -> dense [B,T,28,D], valid [B,28]."""
        b, t, k, _ = expert_outputs.shape
        if selected is None:
            if k != self.n_joints:
                raise ValueError(
                    "selected must be provided when k != n_joints")
            selected_t = torch.arange(self.n_joints,
                                      device=expert_outputs.device)
        else:
            selected_t = torch.as_tensor(
                selected, device=expert_outputs.device, dtype=torch.long)
        tokens = self.mask_token.view(1, 1, self.n_joints, self.expert_dim
                                      ).expand(b, t, -1, -1).clone()
        tokens[:, :, selected_t, :] = expert_outputs
        valid = torch.zeros(b, self.n_joints, dtype=torch.bool,
                            device=expert_outputs.device)
        valid[:, selected_t] = True
        return tokens, valid


# ---------------------------------------------------------------------------
# Activity-specific joint sets
# ---------------------------------------------------------------------------

ACTIVITY_JOINT_SETS = {
    "squat": [
        "Pelvis", "LeftUpperLeg", "RightUpperLeg",
        "LeftLowerLeg", "RightLowerLeg",
        "LeftFoot", "RightFoot",
    ],
    "bench_press": [
        "LeftShoulder", "RightShoulder",
        "LeftUpperArm", "RightUpperArm",
        "LeftForeArm", "RightForeArm",
        "LeftHand", "RightHand",
    ],
    "upper_body": [
        "T8", "Neck", "Head",
        "LeftShoulder", "RightShoulder",
        "LeftUpperArm", "RightUpperArm",
        "LeftForeArm", "RightForeArm",
        "LeftHand", "RightHand",
    ],
    "lower_body": [
        "Pelvis",
        "LeftUpperLeg", "RightUpperLeg",
        "LeftLowerLeg", "RightLowerLeg",
        "LeftFoot", "RightFoot",
        "LeftToe", "RightToe",
    ],
    "full_body": list(CANONICAL_JOINTS),
}


def activity_joint_indices(activity: str) -> List[int]:
    """Return canonical joint indices for a named activity."""
    names = ACTIVITY_JOINT_SETS.get(activity)
    if names is None:
        raise KeyError(
            f"Unknown activity '{activity}'. "
            f"Known: {sorted(ACTIVITY_JOINT_SETS)}"
        )
    return [CANONICAL_JOINTS.index(n) for n in names]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def selective_inference(
    encoder: SharedEncoder,
    experts: MultiLightweightExpert,
    imu: torch.Tensor,
    imu_mask: torch.Tensor,
    selected_joints: Sequence[int],
    lengths: Optional[torch.Tensor] = None,
    assembler: Optional[LightweightExpertOutputAssembler] = None,
) -> Dict[str, torch.Tensor]:
    """Run encoder once + only selected experts.

    Returns dict with:
      shared_repr:    [B, T, 28, d]   (full per-joint encoder output)
      expert_outputs: [B, T, k, 9]    (selected experts only)
      selected_joints: list of joint indices
    If assembler is provided, also returns:
      dense_tokens: [B, T, 28, 9]    (with mask tokens at unselected joints)
      valid_mask:   [B, 28]
    """
    shared = encoder(imu, imu_mask, lengths)
    expert_out = experts(shared, imu, imu_mask, lengths,
                         active=list(selected_joints))
    result = {
        "shared_repr": shared,
        "expert_outputs": expert_out,
        "selected_joints": list(selected_joints),
    }
    if assembler is not None:
        dense, valid = assembler(expert_out, selected=list(selected_joints))
        result["dense_tokens"] = dense
        result["valid_mask"] = valid
    return result
