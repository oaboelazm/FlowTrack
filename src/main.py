from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.app.pipeline import FlowTrackPipeline
from src.utils.config import load_yaml


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = config

    if args.source is not None:
        cfg["source"]["input"] = args.source
    if args.weights is not None:
        cfg["model"]["weights"] = args.weights
    if args.conf is not None:
        cfg["model"]["conf"] = args.conf
    if args.iou is not None:
        cfg["model"]["iou"] = args.iou
    if args.imgsz is not None:
        cfg["model"]["imgsz"] = args.imgsz
    if args.device is not None:
        cfg["model"]["device"] = args.device
    if args.half:
        cfg["model"]["half"] = True
    if args.no_tracking:
        cfg["tracking"]["enabled"] = False
    if args.tracker is not None:
        cfg["tracking"]["tracker"] = args.tracker
    if args.no_display:
        cfg["app"]["display"] = False
    if args.print_counts:
        cfg["app"]["print_counts"] = True
    if args.show_heatmap:
        cfg["app"]["show_heatmap"] = True

    if args.line_x1 is not None:
        cfg["line_counter"]["x1"] = args.line_x1
    if args.line_y1 is not None:
        cfg["line_counter"]["y1"] = args.line_y1
    if args.line_x2 is not None:
        cfg["line_counter"]["x2"] = args.line_x2
    if args.line_y2 is not None:
        cfg["line_counter"]["y2"] = args.line_y2

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowTrack - Real-Time Traffic Monitoring")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--source", type=str, help="Webcam index (e.g., 0) or stream URL")

    parser.add_argument("--weights", type=str, help="YOLO weights path")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument("--device", type=str, help="cuda:0 / cpu / auto(empty)")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference")

    parser.add_argument("--no-tracking", action="store_true", help="Disable tracker (detection only)")
    parser.add_argument("--tracker", type=str, help="Tracker yaml (bytetrack.yaml or botsort.yaml)")

    parser.add_argument("--line-x1", type=int, help="Line point 1 x")
    parser.add_argument("--line-y1", type=int, help="Line point 1 y")
    parser.add_argument("--line-x2", type=int, help="Line point 2 x")
    parser.add_argument("--line-y2", type=int, help="Line point 2 y")

    parser.add_argument("--show-heatmap", action="store_true", help="Overlay movement heatmap")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    parser.add_argument("--print-counts", action="store_true", help="Print per-frame counts to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg = apply_overrides(cfg, args)

    pipeline = FlowTrackPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
