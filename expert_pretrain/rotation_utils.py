"""Rotation helpers for the 6D orientation representation."""

import torch
import torch.nn.functional as F


def normalize_rotation_6d(r6d: torch.Tensor) -> torch.Tensor:
    """Normalize a 6D rotation representation as two orthonormal columns."""
    a1, a2 = r6d[..., :3], r6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    return torch.cat([b1, b2], dim=-1)


def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation [..., 6] to matrices [..., 3, 3]."""
    r6d = normalize_rotation_6d(r6d)
    b1, b2 = r6d[..., :3], r6d[..., 3:]
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)
