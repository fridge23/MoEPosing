"""28 (per-joint) INDEPENDENT transformer experts.

Each joint gets its own self-contained 4-layer / 128-dim transformer that:
  - encodes the masked partial IMU input (learned missing-IMU token +
    sensor-id embedding + value linear),
  - pools over the canonical sensor slots per frame,
  - runs a temporal TransformerEncoder,
  - regresses that joint's target vector.

There is NO shared trunk: experts share no parameters, so each can specialize to
"how does my joint move given whatever sparse IMUs are available". They are held in
one nn.Module only for convenient batched training/saving.
"""
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from rotation_utils import normalize_rotation_6d
from schema import CANONICAL_IMUS, CANONICAL_JOINTS

EXPERT_ORIENTATION_DIM = 6
EXPERT_MOTION_DELTA_DIM = 3
EXPERT_RAW_OUTPUT_DIM = EXPERT_ORIENTATION_DIM + EXPERT_MOTION_DELTA_DIM
CANONICAL_IMU_TO_JOINT = [CANONICAL_JOINTS.index(name) for name in CANONICAL_IMUS]


def split_expert_prediction(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split raw expert tokens into 6D orientation and adjacent-frame delta."""
    if pred.shape[-1] < EXPERT_RAW_OUTPUT_DIM:
        raise ValueError(f"Expert output dim must be at least {EXPERT_RAW_OUTPUT_DIM}, got {pred.shape[-1]}")
    out = {
        "orientation_6d": pred[..., :EXPERT_ORIENTATION_DIM],
        "motion_delta": pred[..., EXPERT_ORIENTATION_DIM:EXPERT_RAW_OUTPUT_DIM],
    }
    if pred.shape[-1] > EXPERT_RAW_OUTPUT_DIM:
        out["extra"] = pred[..., EXPERT_RAW_OUTPUT_DIM:]
    return out


class SparseIMUJointTokenizer(nn.Module):
    """Map raw canonical IMUs into a full 28-joint token grid.

    The dataset stores only the physical IMU slots [B,T,15,12]. The model consumes a
    full-body [B,T,28,12] grid: visible sensors are copied into their matching joint
    slot, and all other joint slots use a learned IMU mask token.
    """
    def __init__(self, n_imus: int, n_joints: int, imu_dim: int = 12,
                 imu_to_joint: Optional[Sequence[int]] = None):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints
        self.imu_dim = imu_dim
        if imu_to_joint is None:
            imu_to_joint = CANONICAL_IMU_TO_JOINT[:n_imus]
        if len(imu_to_joint) != n_imus:
            raise ValueError(f"imu_to_joint length {len(imu_to_joint)} != n_imus {n_imus}")
        if any(j < 0 or j >= n_joints for j in imu_to_joint):
            raise ValueError("imu_to_joint contains an out-of-range joint index")
        self.register_buffer("imu_to_joint", torch.tensor(list(imu_to_joint), dtype=torch.long))
        self.imu_mask_token = nn.Parameter(torch.zeros(n_joints, imu_dim))

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor):
        if imu.dim() != 4:
            raise ValueError(f"imu must be [B,T,S,D], got {tuple(imu.shape)}")
        b, t, s, d = imu.shape
        if s != self.n_imus or d != self.imu_dim:
            raise ValueError(f"Expected imu [B,T,{self.n_imus},{self.imu_dim}], got {tuple(imu.shape)}")
        if imu_mask.shape != (b, s):
            raise ValueError(f"Expected imu_mask {(b, s)}, got {tuple(imu_mask.shape)}")
        tokens = self.imu_mask_token.view(1, 1, self.n_joints, d).expand(b, t, -1, -1).clone()
        visible_joint_mask = torch.zeros(b, self.n_joints, dtype=torch.bool, device=imu.device)
        for src in range(s):
            dst = int(self.imu_to_joint[src])
            present = imu_mask[:, src].to(imu.dtype).view(b, 1, 1)
            tokens[:, :, dst, :] = imu[:, :, src, :] * present + tokens[:, :, dst, :] * (1 - present)
            visible_joint_mask[:, dst] |= imu_mask[:, src].bool()
        return tokens, visible_joint_mask


class JointExpert(nn.Module):
    def __init__(self, n_imus: int, imu_dim: int = 12, hidden_dim: int = 128,
                 nhead: int = 4, num_layers: int = 4, dropout: float = 0.1,
                 target_dim: int = EXPERT_RAW_OUTPUT_DIM, n_joints: int = len(CANONICAL_JOINTS)):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.imu_tokenizer = SparseIMUJointTokenizer(n_imus, n_joints, imu_dim=imu_dim)
        self.sensor_value = nn.Linear(imu_dim + 1, hidden_dim)   # +1 = presence flag
        self.sensor_id = nn.Embedding(n_joints, hidden_dim)
        self.frame_fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # imu [B,T,15,12], imu_mask [B,15] -> full joint tokens [B,T,28,12]
        joint_imu, visible_joint_mask = self.imu_tokenizer(imu, imu_mask)
        b, t, j, _ = joint_imu.shape
        present = visible_joint_mask.to(imu.dtype).view(b, 1, j, 1).expand(b, t, j, 1)
        x = self.sensor_value(torch.cat([joint_imu, present], dim=-1))
        x = x + self.sensor_id(torch.arange(j, device=imu.device)).view(1, 1, j, -1)
        x = x.mean(dim=2)
        x = self.frame_fuse(x)
        z = self.temporal(x, src_key_padding_mask=key_padding_mask)
        return self.head(z)

    def forward_dict(self, imu: torch.Tensor, imu_mask: torch.Tensor,
                     key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pred = self.forward(imu, imu_mask, key_padding_mask)
        return split_expert_prediction(pred)


class MultiExpert(nn.Module):
    def __init__(self, n_imus: int, n_joints: int, hidden_dim: int = 128,
                 nhead: int = 4, num_layers: int = 4, dropout: float = 0.1,
                 target_dim: int = EXPERT_RAW_OUTPUT_DIM):
        super().__init__()
        self.n_joints = n_joints
        self.target_dim = target_dim
        self.experts = nn.ModuleList([
            JointExpert(n_imus, n_joints=n_joints, hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers,
                        dropout=dropout, target_dim=target_dim)
            for _ in range(n_joints)
        ])

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor, lengths=None,
                active=None) -> torch.Tensor:
        """active: optional list of joint indices to actually run. When a subset is
        given, ONLY those experts execute (forward+backward), and the output is
        [B,T,len(active),target_dim] aligned to `active`. Used during training to
        drop CONVERGED experts out of compute entirely (their frozen weights are
        already snapshotted). active=None runs all joints (eval / inference)."""
        key_padding_mask = None
        if lengths is not None:
            t = imu.shape[1]
            steps = torch.arange(t, device=imu.device).view(1, t)
            key_padding_mask = steps >= lengths.to(imu.device).view(-1, 1)
        idxs = range(self.n_joints) if active is None else active
        outs = [self.experts[j](imu, imu_mask, key_padding_mask) for j in idxs]
        return torch.stack(outs, dim=2)   # [B,T,len(idxs),target_dim]

    def forward_dict(self, imu: torch.Tensor, imu_mask: torch.Tensor, lengths=None,
                     active=None) -> Dict[str, torch.Tensor]:
        pred = self.forward(imu, imu_mask, lengths, active=active)
        return split_expert_prediction(pred)


class ExpertOutputAssembler(nn.Module):
    """Pack selected expert outputs into dense [B,T,28,9] recovery tokens."""
    def __init__(self, n_joints: int, expert_dim: int = EXPERT_RAW_OUTPUT_DIM):
        super().__init__()
        self.n_joints = n_joints
        self.expert_dim = expert_dim
        self.mask_token = nn.Parameter(torch.zeros(n_joints, expert_dim))

    def forward(self, expert_outputs: torch.Tensor, selected: Optional[Sequence[int]] = None,
                valid_mask: Optional[torch.Tensor] = None):
        if expert_outputs.shape[-1] != self.expert_dim:
            raise ValueError(f"Expected expert dim {self.expert_dim}, got {expert_outputs.shape[-1]}")
        b, t, k, _ = expert_outputs.shape
        if selected is None:
            if k != self.n_joints:
                raise ValueError("selected must be provided when expert_outputs is not dense over all joints")
            selected_t = torch.arange(self.n_joints, device=expert_outputs.device)
        else:
            selected_t = torch.as_tensor(selected, device=expert_outputs.device, dtype=torch.long)
            if selected_t.numel() != k:
                raise ValueError("selected length must match expert_outputs joint dimension")
        tokens = self.mask_token.view(1, 1, self.n_joints, self.expert_dim).expand(b, t, -1, -1).clone()
        tokens[:, :, selected_t, :] = expert_outputs
        if valid_mask is None:
            valid = torch.zeros(b, self.n_joints, dtype=torch.bool, device=expert_outputs.device)
            valid[:, selected_t] = True
        else:
            valid = valid_mask.to(device=expert_outputs.device, dtype=torch.bool)
            if valid.shape != (b, self.n_joints):
                raise ValueError(f"valid_mask must have shape {(b, self.n_joints)}, got {tuple(valid.shape)}")
        return tokens, valid


def _valid_mask(pred, joint_mask, lengths):
    valid = joint_mask.to(pred.dtype).view(pred.shape[0], 1, pred.shape[2], 1)
    if lengths is not None:
        steps = torch.arange(pred.shape[1], device=pred.device).view(1, -1, 1, 1)
        valid = valid * (steps < lengths.to(pred.device).view(-1, 1, 1, 1)).to(pred.dtype)
    return valid


def masked_motion_loss(pred, target, joint_mask, lengths=None):
    """POOLED masked L1: one true ELEMENT-WISE mean over all valid (joint, frame)
    entries. `valid` counts joint-frames, so we also divide by the target dim D to
    get a per-element mean (without the /D this is D-times too large). Joints present
    in fewer samples contribute proportionally smaller gradient (cf. per_joint_*)."""
    valid = _valid_mask(pred, joint_mask, lengths)
    err = (pred - target).abs() * valid
    return err.sum() / valid.sum().clamp_min(1.0) / pred.shape[-1]


def per_joint_motion_loss(pred, target, joint_mask, lengths=None):
    """PER-JOINT-normalized masked L1: average each joint's own mean error, then
    average over the joints present in the batch. Every joint contributes equally
    regardless of how often it appears, removing the implicit per-joint LR bias of
    the pooled loss (experts already have independent params, so this gives each
    expert the gradient it would get if trained on its own)."""
    valid = _valid_mask(pred, joint_mask, lengths).expand_as(pred)   # [B,T,J,D]
    num = ((pred - target).abs() * valid).sum(dim=(0, 1, 3))         # [J]
    den = valid.sum(dim=(0, 1, 3))                                   # [J]
    present = (den > 0).to(pred.dtype)                               # joints in batch
    per_joint = num / den.clamp_min(1.0)                            # [J] mean abs err
    return (per_joint * present).sum() / present.sum().clamp_min(1.0)


def _masked_dim_loss(pred, target, joint_mask, lengths=None, *, squared: bool, per_joint: bool):
    valid = _valid_mask(pred, joint_mask, lengths).expand_as(pred)
    err = pred - target
    err = err.pow(2) if squared else err.abs()
    err = err * valid
    if not per_joint:
        return err.sum() / valid.sum().clamp_min(1.0)
    num = err.sum(dim=(0, 1, 3))
    den = valid.sum(dim=(0, 1, 3))
    present = (den > 0).to(pred.dtype)
    each = num / den.clamp_min(1.0)
    return (each * present).sum() / present.sum().clamp_min(1.0)


def orientation_6d_loss(pred6, target6, joint_mask, lengths=None, *, per_joint: bool = True):
    """MSE on normalized 6D rotations.

    TODO: optionally replace this with geodesic loss after 6D-to-matrix conversion.
    """
    pred6 = normalize_rotation_6d(pred6.float()).to(pred6.dtype)
    target6 = normalize_rotation_6d(target6.float()).to(target6.dtype)
    return _masked_dim_loss(pred6, target6, joint_mask, lengths, squared=True, per_joint=per_joint)


def motion_delta_loss(pred_delta, target_delta, joint_mask, lengths=None, *, per_joint: bool = True):
    return _masked_dim_loss(pred_delta, target_delta, joint_mask, lengths, squared=False, per_joint=per_joint)


def expert_target_loss(pred, target, joint_mask, lengths=None, *, orientation_slice=None,
                       motion_delta_slice=None, lambda_orientation: float = 1.0,
                       lambda_motion_delta: float = 1.0, per_joint: bool = True):
    pieces = []
    if orientation_slice is not None and lambda_orientation:
        a, b = orientation_slice
        pieces.append(lambda_orientation * orientation_6d_loss(
            pred[..., a:b], target[..., a:b], joint_mask, lengths, per_joint=per_joint,
        ))
    if motion_delta_slice is not None and lambda_motion_delta:
        a, b = motion_delta_slice
        pieces.append(lambda_motion_delta * motion_delta_loss(
            pred[..., a:b], target[..., a:b], joint_mask, lengths, per_joint=per_joint,
        ))
    if not pieces:
        return per_joint_motion_loss(pred, target, joint_mask, lengths) if per_joint else masked_motion_loss(
            pred, target, joint_mask, lengths,
        )
    return torch.stack([p if torch.is_tensor(p) else pred.new_tensor(p) for p in pieces]).sum()
