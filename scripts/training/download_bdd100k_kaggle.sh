#!/usr/bin/env bash
set -euo pipefail

# Requires Kaggle API token at ~/.kaggle/kaggle.json
# Example datasets (community mirrors):
#   solesensei/bdd100k
# Adjust if you use a different source.

OUT_DIR="${1:-datasets/raw}"
mkdir -p "$OUT_DIR"

kaggle datasets download -d solesensei/bdd100k -p "$OUT_DIR" --force

cd "$OUT_DIR"
unzip -o bdd100k.zip -d bdd100k

echo "BDD100K downloaded to: $OUT_DIR/bdd100k"
