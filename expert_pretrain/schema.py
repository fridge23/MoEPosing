CANONICAL_IMUS = [
    "Pelvis",
    "Head",
    "LeftHand",
    "RightHand",
    "LeftForeArm",
    "RightForeArm",
    "LeftLowerLeg",
    "RightLowerLeg",
    "LeftFoot",
    "RightFoot",
    "T8",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftUpperLeg",
    "RightUpperLeg",
]

XSENS_JOINTS = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe",
]

SMPL_JOINTS = [
    "Pelvis", "LeftUpperLeg", "RightUpperLeg", "Spine1",
    "LeftLowerLeg", "RightLowerLeg", "Spine2", "LeftFoot", "RightFoot",
    "Spine3", "LeftToe", "RightToe", "Neck", "LeftShoulder",
    "RightShoulder", "Head", "LeftUpperArm", "RightUpperArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "LeftHandIndex", "RightHandIndex",
]

CANONICAL_JOINTS = [
    "Pelvis",
    "LeftUpperLeg", "RightUpperLeg",
    "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot",
    "LeftToe", "RightToe",
    "L5", "L3", "T12", "T8", "Spine1", "Spine2", "Spine3",
    "Neck", "Head",
    "LeftShoulder", "RightShoulder",
    "LeftUpperArm", "RightUpperArm",
    "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand",
    "LeftHandIndex", "RightHandIndex",
]

ALIASES = {
    "Root": "Pelvis",
    "LeftWrist": "LeftHand",
    "RightWrist": "RightHand",
    "LeftElbow": "LeftForeArm",
    "RightElbow": "RightForeArm",
    "LeftKnee": "LeftLowerLeg",
    "RightKnee": "RightLowerLeg",
    "LeftAnkle": "LeftFoot",
    "RightAnkle": "RightFoot",
}

DIP_REDUCED_IMUS = ["Pelvis", "LeftLowerLeg", "RightLowerLeg", "Head", "LeftForeArm", "RightForeArm"]


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)
