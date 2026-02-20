#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <bdd100k_root>"
  exit 1
fi

python scripts/training/prepare_bdd100k.py --bdd-root "$1"
