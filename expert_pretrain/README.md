# Joint Expert Pretraining Pipeline

This folder adds a sequence pretraining pipeline on top of DynaIP data.

Goal:
- Use arbitrary IMU combinations as input.
- Missing IMUs are zero-filled and marked by `imu_mask`.
- Predict every joint's temporal motion signal rather than relying on absolute 3D position.
- Each joint has its own expert head on top of a shared transformer encoder.

## Canonical sample format

Each generated training window contains:

- `imu`: `[T, n_imus, 12]`
  - first 9 dims: IMU orientation as rotation matrix
  - last 3 dims: free acceleration
  - missing IMUs are zero
- `imu_mask`: `[n_imus]`
- `joint_delta`: `[T, n_joints, 3]`
  - root-relative per-frame displacement
  - this is the main training target
- `joint_velocity`: `[T, n_joints, 3]`
- `joint_displacement`: `[T, n_joints, 3]`
  - cumulative displacement from the start of the window
- `joint_step_distance`: `[T, n_joints, 1]`
- `joint_mask`: `[n_joints]`
  - missing joints are zero-filled

## Workflow

1. Download raw public data:

```bash
scripts/start_public_downloads.sh
scripts/download_status.sh
```

2. Stop downloads if needed:

```bash
scripts/stop_public_downloads.sh
```

3. Extract archives manually or with a future extraction helper, then run DynaIP extraction/processing:

```bash
python datasets/extract.py
python datasets/process.py
```

`datasets/process.py` requires SMPL files for DIP-IMU processing. SMPL and DIP-IMU require license/login and are not downloaded by the public script.

4. Build the expert pretraining windows from extracted full-sensor Xsens-style data:

```bash
python expert_pretrain/build_joint_motion_dataset.py --mode extract --output datasets/expert_pretrain
```

If only DynaIP `datasets/work` exists, build from the reduced 6-IMU representation:

```bash
python expert_pretrain/build_joint_motion_dataset.py --mode work --output datasets/expert_pretrain_work
```

5. Train the shared transformer + per-joint experts:

```bash
python expert_pretrain/train_joint_experts.py --data datasets/expert_pretrain --batch-size 16 --epochs 20
```
