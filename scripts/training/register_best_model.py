#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_registry import register_best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy best trained weights to FlowTrack model registry")
    parser.add_argument("--best", required=True, type=str, help="Path to trained best.pt")
    parser.add_argument("--target", default="models/flowtrack_best.pt", type=str, help="Target model path")
    args = parser.parse_args()

    out = register_best_model(best_weights=args.best, target=args.target)
    print(f"Registered model: {out}")


if __name__ == "__main__":
    main()
