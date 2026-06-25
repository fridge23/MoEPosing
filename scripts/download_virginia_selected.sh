#!/usr/bin/env bash
set -euo pipefail

dest="${1:-datasets/raw/virginia}"
mkdir -p "$dest"

download() {
  local name="$1"
  local url="$2"
  wget -nv -c -O "$dest/$name" "$url"
}

download compressed_mvnx_dataset_P1_Day_1.zip https://ndownloader.figshare.com/files/27383579
download compressed_mvnx_dataset_P2_Day_1.zip https://ndownloader.figshare.com/files/27383609
download compressed_mvnx_dataset_P3_Day_1.zip https://ndownloader.figshare.com/files/27384026
download compressed_mvnx_dataset_P4_Day_1.zip https://ndownloader.figshare.com/files/27384053
download compressed_mvnx_dataset_P5_Day_1.zip https://ndownloader.figshare.com/files/27408479
download compressed_mvnx_dataset_P6_Day_2.zip https://ndownloader.figshare.com/files/27408605
download compressed_mvnx_dataset_P10_Day_1.zip https://ndownloader.figshare.com/files/27429866
download compressed_mvnx_dataset_P11_Day_1.zip https://ndownloader.figshare.com/files/27431270
download compressed_mvnx_dataset_P13_Day_2.zip https://ndownloader.figshare.com/files/27442799
