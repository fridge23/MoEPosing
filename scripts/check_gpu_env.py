#!/usr/bin/env python3
import os
import subprocess
import sys

import torch


def run(cmd):
    try:
        out = subprocess.run(cmd, check=False, text=True, capture_output=True)
        print(f"$ {' '.join(cmd)}")
        if out.stdout:
            print(out.stdout.rstrip())
        if out.stderr:
            print(out.stderr.rstrip())
        print(f"exit={out.returncode}")
    except FileNotFoundError:
        print(f"$ {' '.join(cmd)}")
        print("not found")


def main():
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"device {i}: {props.name}, cc={props.major}.{props.minor}, "
            f"memory={props.total_memory / 1024**3:.2f} GiB"
        )
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("cuda matmul ok:", float(y[0, 0].detach().cpu()))
    print()
    run(["nvidia-smi"])


if __name__ == "__main__":
    main()
