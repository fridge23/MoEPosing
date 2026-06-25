#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
log="logs/download_public_raw.log"

if pgrep -af "scripts/download_public_raw.sh" >/dev/null; then
  echo "download_public_raw.sh is already running"
  pgrep -af "scripts/download_public_raw.sh"
  exit 0
fi

nohup scripts/download_public_raw.sh >> "$log" 2>&1 &
echo $! > logs/download_public_raw.pid
echo "started public raw download pid=$(cat logs/download_public_raw.pid)"
echo "log: $log"
