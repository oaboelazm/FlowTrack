#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and export trained YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, default="configs/training/traffic_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--export-openvino", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("Validation complete")
    print(metrics)

    if args.export_onnx:
        onnx_path = model.export(format="onnx", imgsz=args.imgsz)
        print(f"Exported ONNX: {onnx_path}")

    if args.export_openvino:
        ov_path = model.export(format="openvino", imgsz=args.imgsz)
        print(f"Exported OpenVINO: {ov_path}")


if __name__ == "__main__":
    main()
