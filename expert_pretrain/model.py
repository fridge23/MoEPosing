import torch
from torch import nn


class JointExpertTransformer(nn.Module):
    def __init__(
        self,
        n_imus: int,
        n_joints: int,
        imu_dim: int = 12,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        target_dim: int = 3,
    ):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints
        self.hidden_dim = hidden_dim

        self.sensor_id = nn.Embedding(n_imus, hidden_dim)
        self.sensor_value = nn.Linear(imu_dim + 1, hidden_dim)
        self.frame_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.joint_id = nn.Embedding(n_joints, hidden_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, target_dim),
            )
            for _ in range(n_joints)
        ])

    def forward(self, imu: torch.Tensor, imu_mask: torch.Tensor, lengths=None):
        """Predict per-joint motion deltas.

        Args:
            imu: [batch, time, n_imus, 12]. Missing sensors should already be zeroed.
            imu_mask: [batch, n_imus] bool/float; 1 means that sensor exists in this sample.
            lengths: optional [batch] valid lengths for padded batches.
        Returns:
            [batch, time, n_joints, 3] joint delta predictions.
        """
        b, t, s, _ = imu.shape
        mask = imu_mask.to(dtype=imu.dtype, device=imu.device).view(b, 1, s, 1).expand(b, t, s, 1)
        x = torch.cat([imu * mask, mask], dim=-1)
        x = self.sensor_value(x)
        sid = torch.arange(s, device=imu.device)
        x = x + self.sensor_id(sid).view(1, 1, s, self.hidden_dim)
        x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        x = self.frame_fuse(x)

        key_padding_mask = None
        if lengths is not None:
            steps = torch.arange(t, device=imu.device).view(1, t)
            key_padding_mask = steps >= lengths.to(imu.device).view(b, 1)
        z = self.temporal(x, src_key_padding_mask=key_padding_mask)

        outs = []
        for j, expert in enumerate(self.experts):
            jid = self.joint_id.weight[j].view(1, 1, -1).expand(b, t, -1)
            outs.append(expert(torch.cat([z, jid], dim=-1)))
        return torch.stack(outs, dim=2)


def masked_motion_loss(pred, target, joint_mask, lengths=None):
    valid = joint_mask.to(dtype=pred.dtype, device=pred.device).view(pred.shape[0], 1, pred.shape[2], 1)
    if lengths is not None:
        steps = torch.arange(pred.shape[1], device=pred.device).view(1, -1, 1, 1)
        valid = valid * (steps < lengths.to(pred.device).view(-1, 1, 1, 1)).to(pred.dtype)
    err = (pred - target).abs() * valid
    return err.sum() / valid.sum().clamp_min(1.0)
