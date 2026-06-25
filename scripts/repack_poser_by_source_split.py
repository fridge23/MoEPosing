#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


EVAL_MOBILEPOSER_KEYWORDS = ("totalcapture", "dip", "imuposer")

SOURCE_UNIT_ASSUMPTIONS = {
    "mobileposer": {
        "input_position": "processed SMPL joint tensor; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "processed MobilePoser acceleration tensor; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from MobilePoser preprocessing",
    },
    "dynaip_andy": {
        "input_position": "DynaIP extracted Xsens joint position; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "DynaIP free acceleration; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "dynaip_cip": {
        "input_position": "DynaIP extracted CIP/Xsens joint position; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "DynaIP free acceleration; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "dynaip_emokine": {
        "input_position": "DynaIP extracted joint position; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "DynaIP free acceleration; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "dynaip_unipd": {
        "input_position": "DynaIP extracted Xsens joint position; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "DynaIP free acceleration; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "dynaip_virginia": {
        "input_position": "DynaIP extracted Xsens joint position; treated as meters",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "DynaIP free acceleration; not rescaled in repack",
        "orientation": "9D rotation matrix",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "assumed from DynaIP extraction pipeline",
    },
    "ceti_age_kinematics": {
        "input_position": "CeTI centimeters were converted to meters before the input shards were created",
        "conversion": "position_cm / 100.0 in build_unified_pose_data.py",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "not available; IMU tensor is zero-filled",
        "orientation": "CeTI joint angles are not used as IMU orientation",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "explicit units from CeTI *_channels.tsv",
    },
    "vidimu": {
        "input_position": "VIDIMU video/IK CSV millimeter coordinates were converted to meters before input shards",
        "conversion": "position_mm / 1000.0 in build_unified_pose_data.py",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "not available from current .raw reader; orientation-only IMUs when present",
        "orientation": ".raw quaternion converted to 9D rotation matrix before input shards",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "scale inferred from file magnitudes/OpenSim format",
    },
    "ultra_mocap": {
        "input_position": "ULTRA .trc millimeter markers were converted to meters before input shards",
        "conversion": "marker_mm / 1000.0 in build_unified_pose_data.py",
        "output_position": "meters, root-relative targets inherited from input shards",
        "imu_acceleration": "ACC columns divided by 1000.0 before input shards; sensor scaling is not fully verified",
        "orientation": "identity placeholder; gyro stored separately as imu_gyro",
        "fps": "60 Hz in poser_mle_unified_60hz",
        "confidence": "marker units explicit; IMU scaling uncertain",
    },
}


def normalized_split(sample: Dict) -> str:
    source = str(sample.get("source", "unknown"))
    motion = str(sample.get("motion", "")).lower()
    old_split = str(sample.get("split", "train")).lower()
    if source == "mobileposer" and any(key in motion for key in EVAL_MOBILEPOSER_KEYWORDS):
        return "eval"
    if old_split in {"eval", "test", "validation", "val"}:
        return "eval"
    return "train"


def route_for(sample: Dict) -> Tuple[str, str]:
    source = str(sample.get("source", "unknown")).replace("/", "_")
    split = normalized_split(sample)
    return source, split


class RouteWriter:
    def __init__(self, output: Path, shard_size: int):
        self.output = output
        self.shard_size = shard_size
        self.buffers: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        self.shard_idx: Counter = Counter()
        self.window_counts: Counter = Counter()
        self.sequence_ids: Dict[Tuple[str, str], set] = defaultdict(set)
        self.quality_counts: Counter = Counter()
        self.source_split_quality_counts: Counter = Counter()

    def add(self, sample: Dict):
        source, split = route_for(sample)
        sample = dict(sample)
        sample["split"] = split
        if source in SOURCE_UNIT_ASSUMPTIONS:
            sample["unit_info"] = SOURCE_UNIT_ASSUMPTIONS[source]
        key = (source, split)
        self.buffers[key].append(sample)
        self.window_counts[key] += 1
        self.sequence_ids[key].add(str(sample.get("motion", "")))
        quality = str(sample.get("quality", "unknown"))
        self.quality_counts[quality] += 1
        self.source_split_quality_counts[(source, split, quality)] += 1
        if len(self.buffers[key]) >= self.shard_size:
            self.flush(key)

    def flush(self, key: Tuple[str, str]):
        samples = self.buffers[key]
        if not samples:
            return
        source, split = key
        out_dir = self.output / source / split
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = self.shard_idx[key]
        path = out_dir / f"shard_{idx:05d}.pt"
        torch.save({"samples": samples}, path)
        print(f"[save] {path} ({len(samples)} samples)")
        self.buffers[key] = []
        self.shard_idx[key] += 1

    def close(self):
        for key in list(self.buffers):
            self.flush(key)


def load_manifest(input_root: Path) -> Dict:
    path = input_root / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_metadata(output: Path, input_root: Path, writer: RouteWriter, input_manifest: Dict, total_seen: int):
    source_split_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    source_counts: Counter = Counter()
    split_counts: Counter = Counter()
    source_sequence_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    source_quality_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (source, split), count in sorted(writer.window_counts.items()):
        source_split_counts[source][split] = count
        source_counts[source] += count
        split_counts[split] += count
        source_sequence_counts[source][split] = len(writer.sequence_ids[(source, split)])

    for (source, split, quality), count in sorted(writer.source_split_quality_counts.items()):
        source_quality_counts[source][f"{split}:{quality}"] += count

    manifest = dict(input_manifest)
    manifest.update(
        {
            "schema_version": input_manifest.get("schema_version", "poser_mle_unified_v1"),
            "layout": "source/split/shard_XXXXX.pt",
            "input_root": str(input_root),
            "total_windows": sum(writer.window_counts.values()),
            "input_windows_seen": total_seen,
            "split_window_counts": dict(sorted(split_counts.items())),
            "source_window_counts": dict(sorted(source_counts.items())),
            "source_split_window_counts": {k: dict(sorted(v.items())) for k, v in sorted(source_split_counts.items())},
            "source_split_sequence_counts": {k: dict(sorted(v.items())) for k, v in sorted(source_sequence_counts.items())},
            "quality_window_counts": dict(sorted(writer.quality_counts.items())),
            "source_quality_counts": {k: dict(sorted(v.items())) for k, v in sorted(source_quality_counts.items())},
            "source_unit_assumptions": SOURCE_UNIT_ASSUMPTIONS,
            "split_rule": {
                "mobileposer_eval_keywords": list(EVAL_MOBILEPOSER_KEYWORDS),
                "meaning": "MobilePoser motions whose path/name contains TotalCapture, DIP, or IMUPoser are forced to eval; all other non-eval samples are train.",
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    torch.save(manifest, output / "metadata.pt")

    lines = [
        "# Dataset Index",
        "",
        f"Input: `{input_root}`",
        f"Total windows: {manifest['total_windows']}",
        "",
        "| source | train windows | eval windows | train seqs | eval seqs |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in sorted(source_counts):
        split_map = source_split_counts.get(source, {})
        seq_map = source_sequence_counts.get(source, {})
        lines.append(
            f"| `{source}` | {split_map.get('train', 0)} | {split_map.get('eval', 0)} | "
            f"{seq_map.get('train', 0)} | {seq_map.get('eval', 0)} |"
        )
    lines.extend(
        [
            "",
            "MobilePoser eval-only rule: `TotalCapture`, `dip`, and `imuposer` are placed under `mobileposer/eval/`.",
            "Other datasets are kept in their own source folders.",
            "",
            "## Unit Assumptions",
            "",
            "| source | position handling | IMU handling | orientation | confidence |",
            "|---|---|---|---|---|",
        ]
    )
    for source in sorted(source_counts):
        info = SOURCE_UNIT_ASSUMPTIONS.get(source, {})
        position = info.get("conversion") or info.get("input_position", "not documented")
        imu = info.get("imu_acceleration", "not documented")
        orientation = info.get("orientation", "not documented")
        confidence = info.get("confidence", "not documented")
        lines.append(f"| `{source}` | {position} | {imu} | {orientation} | {confidence} |")
    lines.extend(
        [
            "",
            "Important: the repack step does not rescale tensors. Unit conversion happens earlier in `dynaip/expert_pretrain/build_unified_pose_data.py`; this index records the assumptions attached to each sample as `unit_info`.",
            "For full processing details, see `../DATA_PROCESSING_REVIEW.md`.",
        ]
    )
    (output / "DATASET_INDEX.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/pengfei/Downloads/poser_mle_unified_60hz")
    parser.add_argument("--output", default="/home/pengfei/Downloads/poser_mle_unified_60hz_by_dataset")
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()

    input_root = Path(args.input)
    output = Path(args.output)
    writer = RouteWriter(output, args.shard_size)
    manifest = load_manifest(input_root)
    total_seen = 0

    for path in sorted(input_root.glob("shard_*.pt")):
        obj = torch.load(path, map_location="cpu")
        samples = obj.get("samples", [])
        print(f"[read] {path} ({len(samples)} samples)")
        for sample in samples:
            writer.add(sample)
            total_seen += 1

    writer.close()
    write_metadata(output, input_root, writer, manifest, total_seen)
    print(f"[done] wrote {sum(writer.window_counts.values())} windows to {output}")


if __name__ == "__main__":
    main()
