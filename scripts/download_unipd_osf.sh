#!/usr/bin/env bash
set -euo pipefail

dest="${1:-datasets/raw/unipd}"
root_url="https://api.osf.io/v2/nodes/yj9q4/files/onedrive/013T5F34D77OR65DWXZBCKIHE5IBOJFWDO/?page%5Bsize%5D=100"

mkdir -p "$dest"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

download_folder() {
  local api_url="$1"
  local rel_dir="$2"
  local json="$tmpdir/list-$(printf '%s' "$api_url" | md5sum | awk '{print $1}').json"

  curl -fsSL "$api_url" -o "$json"

  jq -r '.data[] | select(.attributes.kind == "folder") |
    [.attributes.name, .relationships.files.links.related.href] | @tsv' "$json" |
  while IFS=$'\t' read -r name child_url; do
    download_folder "$child_url?page%5Bsize%5D=100" "$rel_dir/$name"
  done

  jq -r '.data[] | select(.attributes.kind == "file" and (.attributes.name | endswith(".mvnx"))) |
    [.attributes.name, .links.download] | @tsv' "$json" |
  while IFS=$'\t' read -r name url; do
    mkdir -p "$dest/$rel_dir"
    wget -nv -c -O "$dest/$rel_dir/$name" "$url"
  done
}

download_folder "$root_url" ""
