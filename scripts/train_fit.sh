#!/usr/bin/env bash
set -euo pipefail

python scripts/training/train_yolo.py --train-config configs/training/train_bdd100k.yaml
