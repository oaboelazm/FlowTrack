#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for FlowTrack")
    parser.add_argument("--train-config", type=str, default="configs/training/train_bdd100k.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.train_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg.pop("model", "yolov8n.pt")

    model = YOLO(model_name)
    model.train(**cfg)


if __name__ == "__main__":
    main()
