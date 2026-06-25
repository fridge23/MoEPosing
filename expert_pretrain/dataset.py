import glob
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class JointMotionShardDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(glob.glob(f"{root}/shard_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {root}")
        self.index = []
        self._cache = {}
        for shard_i, path in enumerate(self.paths):
            n = len(torch.load(path, map_location="cpu")["samples"])
            self.index.extend((shard_i, sample_i) for sample_i in range(n))

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_i: int):
        if shard_i not in self._cache:
            self._cache[shard_i] = torch.load(self.paths[shard_i], map_location="cpu")["samples"]
        return self._cache[shard_i]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_i, sample_i = self.index[idx]
        return self._load_shard(shard_i)[sample_i]


def collate_joint_motion(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([item["imu"].shape[0] for item in batch], dtype=torch.long)
    out = {
        "imu": pad_sequence([item["imu"] for item in batch], batch_first=True),
        "joint_delta": pad_sequence([item["joint_delta"] for item in batch], batch_first=True),
        "joint_displacement": pad_sequence([item["joint_displacement"] for item in batch], batch_first=True),
        "joint_velocity": pad_sequence([item["joint_velocity"] for item in batch], batch_first=True),
        "joint_step_distance": pad_sequence([item["joint_step_distance"] for item in batch], batch_first=True),
        "imu_mask": torch.stack([item["imu_mask"] for item in batch]),
        "joint_mask": torch.stack([item["joint_mask"] for item in batch]),
        "lengths": lengths,
    }
    return out
