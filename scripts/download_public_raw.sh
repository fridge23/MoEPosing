#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets/raw/andy datasets/raw/cip datasets/raw/emokine datasets/raw/unipd datasets/raw/virginia logs

echo "[download] AnDy xsens mvnx"
wget -nv -c -O datasets/raw/andy/xens_mnvx.zip \
  https://zenodo.org/api/records/3254403/files/xens_mnvx.zip/content

echo "[download] CIP MTwAwinda"
wget -nv -c -O datasets/raw/cip/MTwAwinda.zip \
  https://zenodo.org/api/records/5801928/files/MTwAwinda.zip/content

echo "[download] Emokine"
wget -nv -c -O datasets/raw/emokine/EmokineDataset_v1.0.zip \
  https://zenodo.org/api/records/7821844/files/EmokineDataset_v1.0.zip/content

echo "[download] UNIPD single_person mvnx"
scripts/download_unipd_osf.sh datasets/raw/unipd

echo "[download] Virginia selected mvnx archives"
scripts/download_virginia_selected.sh datasets/raw/virginia

echo "[download] Public downloads complete"
