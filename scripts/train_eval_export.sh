#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_best.pt>"
  exit 1
fi

python scripts/training/eval_export.py --weights "$1" --export-onnx
