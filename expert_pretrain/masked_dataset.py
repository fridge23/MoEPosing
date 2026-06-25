"""Shard dataset plus deterministic dataset-aware visible-IMU sampling.

The dataset returns the full available IMU signal and its original availability mask.
Training code can then call :func:`apply_visible_imu_sampling` once per batch, using
the current epoch/batch index to reproducibly keep a legal 2-5 IMU subset for each
sample's source dataset. This is per-sample rather than per-batch because the
shuffled dataloader may mix sources with different available sensor locations.

Targets (joint_orient_r6d, joint_delta, etc.) and joint masks are untouched.
"""
import glob
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from schema import CANONICAL_IMUS, CANONICAL_JOINTS
from target_spec import mask_keys_for, split_keys

_SEQ_RE = re.compile(r":\d+$")


def seq_id(motion: str) -> str:
    """Underlying sequence id = motion string without trailing ``:<window>``."""
    return _SEQ_RE.sub("", str(motion))


def _resolve_shard_paths(root: str, split: Optional[str]) -> List[str]:
    root_path = Path(root)
    flat = sorted(glob.glob(str(root_path / "shard_*.pt")))
    if flat:
        return flat
    if split:
        nested = sorted(glob.glob(str(root_path / "*" / split / "shard_*.pt")))
        if nested:
            return nested
    nested = sorted(glob.glob(str(root_path / "*" / "*" / "shard_*.pt")))
    if nested:
        return nested
    raise FileNotFoundError(f"No shard_*.pt in {root}")


def _validate_canonical_schema(root: str) -> None:
    for name in ("metadata.pt", "manifest.json"):
        path = Path(root) / name
        if not path.exists():
            continue
        obj = torch.load(path, map_location="cpu") if path.suffix == ".pt" else json.load(open(path))
        imus = obj.get("canonical_imus")
        joints = obj.get("canonical_joints")
        if imus is not None:
            assert list(imus) == CANONICAL_IMUS, "Dataset IMU order differs from schema.CANONICAL_IMUS"
        if joints is not None:
            assert list(joints) == CANONICAL_JOINTS, "Dataset joint order differs from schema.CANONICAL_JOINTS"
        return


def _stable_seed(seed: int, epoch: int, batch_idx: int, dataset_name: str,
                 sample_idx: int, split: str) -> int:
    payload = f"{seed}|{split}|{epoch}|{batch_idx}|{dataset_name}|{sample_idx}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _visible_pool(available_imus: Sequence[int], min_k: int, max_k: int) -> List[Tuple[int, ...]]:
    available = tuple(sorted(int(i) for i in available_imus))
    upper = min(int(max_k), len(available))
    if len(available) < int(min_k):
        raise ValueError(f"Need at least {min_k} available IMUs, got {len(available)}: {available}")
    pool: List[Tuple[int, ...]] = []
    for k in range(int(min_k), upper + 1):
        pool.extend(itertools.combinations(available, k))
    return pool


def sample_visible_set(
    dataset_name: str,
    epoch: int,
    batch_idx: int,
    available_imus: Sequence[int],
    *,
    seed: int = 0,
    sample_idx: int = 0,
    min_k: int = 2,
    max_k: int = 5,
    split: str = "train",
) -> List[int]:
    """Sample one legal visible-IMU set for a dataset/sample.

    The sample is reproducible from global seed, epoch, batch index, dataset name,
    and sample index. Passing the sample's own available IMU list prevents sampling
    a location that is unavailable in that dataset window.
    """
    pool = _visible_pool(available_imus, min_k=min_k, max_k=max_k)
    rng = random.Random(_stable_seed(seed, epoch, batch_idx, dataset_name, sample_idx, split))
    return list(pool[rng.randrange(len(pool))])


def apply_visible_imu_sampling(
    batch: Dict,
    *,
    epoch: int,
    batch_idx: int,
    seed: int = 0,
    min_k: int = 2,
    max_k: int = 5,
    split: str = "train",
    fixed_per_dataset: bool = False,
    visible_sets: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict:
    """Return a batch with dataset-aware sparse visible IMUs applied."""
    imu = batch["imu"].clone()
    available = batch.get("available_imu_mask", batch["imu_mask"]).bool().clone()
    visible = torch.zeros_like(available)
    sources = batch.get("source", ["unknown"] * imu.shape[0])
    sample_indices = batch.get("sample_idx", torch.arange(imu.shape[0]))
    visible_indices = []
    for b in range(imu.shape[0]):
        source = str(sources[b])
        sample_idx = 0 if fixed_per_dataset else int(sample_indices[b])
        sample_epoch = 0 if fixed_per_dataset else epoch
        sample_batch_idx = 0 if fixed_per_dataset else batch_idx
        present = available[b].nonzero().flatten().tolist()
        if visible_sets is not None and source in visible_sets:
            keep = [int(i) for i in visible_sets[source]]
            unavailable = sorted(set(keep) - set(present))
            if unavailable:
                raise ValueError(f"Fixed visible set for {source} selects unavailable IMUs: {unavailable}")
        else:
            keep = sample_visible_set(
                source, sample_epoch, sample_batch_idx, present, seed=seed, sample_idx=sample_idx,
                min_k=min_k, max_k=max_k, split=split,
            )
        visible[b, keep] = True
        imu[b, :, ~visible[b]] = 0.0
        visible_indices.append(keep)
    out = dict(batch)
    out["imu"] = imu
    out["imu_mask"] = visible
    out["visible_imu_indices"] = visible_indices
    return out


def _random_rotation(sigma_deg: float) -> torch.Tensor:
    """A random 3x3 rotation, angle ~ N(0, sigma_deg) about a random axis (Rodrigues)."""
    axis = torch.randn(3)
    axis = axis / axis.norm().clamp_min(1e-8)
    ang = torch.randn(()) * (sigma_deg * math.pi / 180.0)
    x, y, z = axis
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return torch.eye(3) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


class MaskedMotionDataset(Dataset):
    def __init__(self, root: str, min_k: int = 2, max_k: int = 5, augment: bool = True,
                 target_key: str = "joint_delta", mask_key: str = "joint_mask",
                 exclude_kw=None, include_only_kw=None, acc_noise_std: float = 0.0,
                 ori_bias_deg: float = 0.0, max_cached_shards: int = 8,
                 manifest: Optional[str] = None, split: Optional[str] = None,
                 require_min_imus: bool = False, preload: bool = False):
        _validate_canonical_schema(root)
        self.paths = _resolve_shard_paths(root, split)
        self.min_k, self.max_k = min_k, max_k
        self.augment = augment
        self.require_min_imus = require_min_imus
        self.target_keys = split_keys(target_key)
        self.mask_keys = mask_keys_for(self.target_keys, mask_key)
        self.target_key = ",".join(self.target_keys)
        self.mask_key = ",".join(self.mask_keys)
        self.acc_noise_std = acc_noise_std
        self.ori_bias_deg = ori_bias_deg
        self.max_cached_shards = max_cached_shards
        exclude_kw = [k.lower() for k in (exclude_kw or [])]
        include_only_kw = [k.lower() for k in (include_only_kw or [])]
        # Preferred path: a frozen split MANIFEST keyed on sequence id, so no motion
        # straddles train/val (leak-free). Falls back to keyword filters if absent.
        split_ids = None
        if manifest is not None and split is not None:
            with open(manifest) as f:
                assign = json.load(f)["seq_id"]
            split_ids = {sid for sid, sp in assign.items() if sp == split}
        self._cache: Dict[int, list] = {}
        self._order: List[int] = []
        # preload=True keeps every qualified sample's extracted tensors in RAM (built
        # once in the MAIN process; DataLoader workers inherit it via copy-on-write
        # fork, so there is no per-worker disk reload). Turns disk-bound epochs into
        # compute-bound ones. ~160KB/window -> ~9GB for the 53k train split.
        self.preload = preload
        self._mem: Optional[List[Dict[str, torch.Tensor]]] = [] if preload else None
        self.source_available_imus: Dict[str, set] = {}
        self.source_common_imus: Dict[str, Optional[set]] = {}
        self.source_counts: Dict[str, int] = {}
        self.source_valid_joints: Dict[str, set] = {}
        # index only samples that carry all requested targets + masks
        self.index = []
        for si, p in enumerate(self.paths):
            for i, s in enumerate(torch.load(p, map_location="cpu")["samples"]):
                if any(k not in s for k in self.target_keys) or any(k not in s for k in self.mask_keys):
                    continue
                imu_mask = s["imu_mask"].bool()
                assert imu_mask.numel() == len(CANONICAL_IMUS), "IMU mask length does not match canonical IMU order"
                present = imu_mask.nonzero().flatten().tolist()
                if self.require_min_imus and len(present) < self.min_k:
                    continue
                valid_mask = None
                for key in self.mask_keys:
                    m = s[key].bool()
                    assert m.numel() == len(CANONICAL_JOINTS), "Joint mask length does not match canonical joint order"
                    valid_mask = m if valid_mask is None else valid_mask & m
                if valid_mask is None or not bool(valid_mask.any()):
                    continue
                motion = str(s.get("motion", "")).lower()
                if split_ids is not None and seq_id(s.get("motion", "")) not in split_ids:
                    continue
                if exclude_kw and any(k in motion for k in exclude_kw):
                    continue
                if include_only_kw and not any(k in motion for k in include_only_kw):
                    continue
                source = str(s.get("source", Path(p).parent.parent.name))
                self.source_available_imus.setdefault(source, set()).update(present)
                current = set(present)
                if source not in self.source_common_imus:
                    self.source_common_imus[source] = current
                else:
                    self.source_common_imus[source] &= current
                self.source_counts[source] = self.source_counts.get(source, 0) + 1
                self.source_valid_joints.setdefault(source, set()).update(valid_mask.nonzero().flatten().tolist())
                self.index.append((si, i))
                if self.preload:
                    self._mem.append(self._extract_base(s, source))

    def __len__(self):
        return len(self.index)

    def _shard(self, si: int):
        if si not in self._cache:
            self._cache[si] = torch.load(self.paths[si], map_location="cpu")["samples"]
            self._order.append(si)
            if len(self._order) > self.max_cached_shards:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[si]

    def _extract_base(self, item, source) -> Dict[str, torch.Tensor]:
        """Pull the model-ready tensors out of a raw shard sample (owns its own
        copies, so the shard can be freed). Shared by preload and lazy paths."""
        imu = item["imu"].float()
        imu_mask = item["imu_mask"].bool()
        assert imu.shape[1] == len(CANONICAL_IMUS), "IMU tensor does not match canonical IMU order"
        targets = [item[key].float() for key in self.target_keys]
        target = targets[0] if len(targets) == 1 else torch.cat(targets, dim=-1)
        assert target.shape[1] == len(CANONICAL_JOINTS), "Target tensor does not match canonical joint order"
        mask = item[self.mask_keys[0]].bool().clone()
        for key in self.mask_keys[1:]:
            mask &= item[key].bool()
        return {"imu": imu, "imu_mask": imu_mask, "target": target, "mask": mask, "source": str(source)}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._mem is not None:
            base = self._mem[idx]
        else:
            si, i = self.index[idx]
            item = self._shard(si)[i]
            source = str(item.get("source", Path(self.paths[si]).parent.parent.name))
            base = self._extract_base(item, source)
        imu, imu_mask = base["imu"], base["imu_mask"]
        available_imu_mask = imu_mask          # original availability (pre-subsample)
        if self.augment:
            imu = imu.clone()
            present = imu_mask.nonzero().flatten().tolist()
            if len(present) >= self.min_k:
                k = random.randint(self.min_k, min(self.max_k, len(present)))
                keep = set(random.sample(present, k))
                new_mask = torch.zeros_like(imu_mask)
                for s in present:
                    if s in keep:
                        new_mask[s] = True
                    else:
                        imu[:, s] = 0.0
                imu_mask = new_mask
            # NOTE: sim-to-real IMU noise is applied on the GPU in the training step,
            # NOT here — torch ops inside forked DataLoader workers deadlock.
        return {
            "imu": imu,
            "imu_mask": imu_mask,
            "available_imu_mask": available_imu_mask,
            "target": base["target"],
            "mask": base["mask"],
            "source": base["source"],
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }

    def sample_visible_set(self, dataset_name: str, epoch: int, batch_idx: int,
                           sample_idx: int = 0) -> List[int]:
        available = sorted(self.source_available_imus.get(dataset_name, ()))
        return sample_visible_set(dataset_name, epoch, batch_idx, available,
                                  seed=0, sample_idx=sample_idx,
                                  min_k=self.min_k, max_k=self.max_k)

    def describe_sources(self) -> List[str]:
        """Human-readable source summary for the qualified indexed samples."""
        lines = []
        for source in sorted(self.source_counts):
            available = sorted(self.source_available_imus.get(source, ()))
            common = sorted(self.source_common_imus.get(source) or ())
            valid_joints = sorted(self.source_valid_joints.get(source, ()))
            missing_joints = [CANONICAL_JOINTS[i] for i in range(len(CANONICAL_JOINTS)) if i not in valid_joints]
            imu_names = [CANONICAL_IMUS[i] for i in common or available]
            lines.append(
                f"{source}: windows={self.source_counts[source]} "
                f"imus={len(common or available)}/{len(CANONICAL_IMUS)} {imu_names} "
                f"supervised_joints={len(valid_joints)}/{len(CANONICAL_JOINTS)} "
                f"missing_joints={missing_joints}"
            )
        return lines


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["imu"].shape[0] for b in batch], dtype=torch.long)
    return {
        "imu": pad_sequence([b["imu"] for b in batch], batch_first=True),
        "target": pad_sequence([b["target"] for b in batch], batch_first=True),
        "imu_mask": torch.stack([b["imu_mask"] for b in batch]),
        "available_imu_mask": torch.stack([b["available_imu_mask"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "lengths": lengths,
        "source": [b["source"] for b in batch],
        "sample_idx": torch.stack([b["sample_idx"] for b in batch]),
    }


def _base_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def build_fixed_visible_sets(datasets: Iterable, *, seed: int = 0, min_k: int = 2,
                             max_k: int = 5, split: str = "val",
                             provided: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, List[int]]:
    """Choose one reproducible fixed visible set per dataset/source for eval."""
    fixed: Dict[str, List[int]] = {}
    for ds in datasets:
        if ds is None:
            continue
        base = _base_dataset(ds)
        common = getattr(base, "source_common_imus", {})
        available = getattr(base, "source_available_imus", {})
        for source in sorted(available):
            if source in fixed:
                continue
            if provided and source in provided and provided[source] is not None:
                keep = [int(i) for i in provided[source]]
                source_avail = set(available[source])
                unavailable = sorted(set(keep) - source_avail)
                if unavailable:
                    raise ValueError(f"Provided eval visible set for {source} has unavailable IMUs: {unavailable}")
                if not (min_k <= len(keep) <= max_k):
                    raise ValueError(f"Provided eval visible set for {source} must have {min_k}-{max_k} IMUs")
                fixed[source] = keep
                continue
            pool_source = sorted(common.get(source) or available[source])
            fixed[source] = sample_visible_set(
                source, 0, 0, pool_source, seed=seed, sample_idx=0,
                min_k=min_k, max_k=max_k, split=split,
            )
    return fixed
