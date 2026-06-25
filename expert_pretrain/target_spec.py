from typing import Dict, Iterable, List, Tuple, Union


TARGET_DIMS = {
    "joint_orient_r6d": 6,
    "joint_global_orient_r6d": 6,
    "joint_rot_delta_r6d": 6,
    "joint_delta": 3,
    "joint_delta_local": 3,
    "joint_velocity": 3,
    "joint_displacement": 3,
    "joint_position_root_relative": 3,
    "joint_step_distance": 1,
}

DEFAULT_MASK_KEYS = {
    "joint_orient_r6d": "orient_mask",
    "joint_global_orient_r6d": "global_orient_mask",
    "joint_rot_delta_r6d": "rot_delta_mask",
    "joint_delta": "joint_mask",
    "joint_delta_local": "joint_delta_local_mask",
    "joint_velocity": "joint_mask",
    "joint_displacement": "joint_mask",
    "joint_position_root_relative": "joint_mask",
    "joint_step_distance": "joint_mask",
}


def split_keys(keys: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(keys, str):
        return [k.strip() for k in keys.split(",") if k.strip()]
    return [str(k).strip() for k in keys if str(k).strip()]


def target_dim(keys: Union[str, Iterable[str]]) -> int:
    total = 0
    for key in split_keys(keys):
        if key not in TARGET_DIMS:
            known = ", ".join(sorted(TARGET_DIMS))
            raise KeyError(f"Unknown target key '{key}'. Known target keys: {known}")
        total += TARGET_DIMS[key]
    return total


def target_slices(keys: Union[str, Iterable[str]]) -> Dict[str, Tuple[int, int]]:
    out = {}
    start = 0
    for key in split_keys(keys):
        width = TARGET_DIMS[key]
        out[key] = (start, start + width)
        start += width
    return out


def slice_for(keys: Union[str, Iterable[str]], key: str):
    return target_slices(keys).get(key)


def mask_keys_for(target_keys: Union[str, Iterable[str]], mask_keys: Union[str, Iterable[str]]) -> List[str]:
    targets = split_keys(target_keys)
    masks = split_keys(mask_keys)
    if not masks or masks == ["auto"]:
        return [DEFAULT_MASK_KEYS[k] for k in targets]
    if len(masks) == 1:
        return masks * len(targets)
    if len(masks) != len(targets):
        raise ValueError("mask_key must be 'auto', one key, or one mask key per target key")
    return masks
