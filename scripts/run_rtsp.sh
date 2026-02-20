#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <stream_url>"
  exit 1
fi

python -m src.main --source "$1"
