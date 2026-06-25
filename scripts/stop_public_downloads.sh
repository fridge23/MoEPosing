#!/usr/bin/env bash
set -euo pipefail

pkill -f "scripts/download_public_raw.sh" || true
pkill -f "scripts/download_unipd_osf.sh" || true
pkill -f "scripts/download_virginia_selected.sh" || true
pkill -f "wget -nv -c -O datasets/raw" || true
echo "stopped public download processes"
