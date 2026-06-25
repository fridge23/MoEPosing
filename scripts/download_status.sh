#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
pgrep -af "download_public_raw|download_unipd|download_virginia|wget -nv -c -O datasets/raw" || true
echo
echo "== raw sizes =="
du -sh datasets/raw datasets/raw/* 2>/dev/null || true
echo
echo "== tail log =="
tail -40 logs/download_public_raw.log 2>/dev/null || true
