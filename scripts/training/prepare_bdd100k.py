#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

# BDD100K labels that map to FlowTrack classes
CLASS_MAP = {
    "pedestrian": 0,
    "rider": 0,
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
}


def yolo_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def convert_split(
    bdd_root: Path,
    split: str,
    out_root: Path,
) -> int:
    labels_json = bdd_root / "labels" / f"bdd100k_labels_images_{split}.json"
    images_dir = bdd_root / "images" / "100k" / split

    if not labels_json.exists():
        raise FileNotFoundError(f"Missing labels JSON: {labels_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    items = json.loads(labels_json.read_text(encoding="utf-8"))

    converted = 0
    for item in tqdm(items, desc=f"Converting {split}"):
        name = item.get("name")
        if not name:
            continue

        src_img = images_dir / name
        if not src_img.exists():
            continue

        labels = item.get("labels", [])
        img_w = int(item.get("attributes", {}).get("W", 1280))
        img_h = int(item.get("attributes", {}).get("H", 720))

        rows: List[str] = []
        for lb in labels:
            category = str(lb.get("category", "")).lower().strip()
            cls_id = CLASS_MAP.get(category)
            if cls_id is None:
                continue

            box = lb.get("box2d")
            if not box:
                continue

            x1 = max(0.0, float(box.get("x1", 0)))
            y1 = max(0.0, float(box.get("y1", 0)))
            x2 = min(float(img_w), float(box.get("x2", img_w)))
            y2 = min(float(img_h), float(box.get("y2", img_h)))

            if x2 <= x1 or y2 <= y1:
                continue

            cx, cy, bw, bh = yolo_box(x1, y1, x2, y2, img_w, img_h)
            rows.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Keep frame only if at least one target object exists
        if not rows:
            continue

        shutil.copy2(src_img, out_images / name)
        txt_name = f"{Path(name).stem}.txt"
        (out_labels / txt_name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        converted += 1

    return converted


def write_dataset_yaml(out_root: Path, yaml_path: Path) -> None:
    text = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: person\n"
        "  1: bicycle\n"
        "  2: car\n"
        "  3: motorcycle\n"
        "  4: bus\n"
        "  5: truck\n"
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLO format for FlowTrack classes")
    parser.add_argument("--bdd-root", type=str, required=True, help="BDD100K root containing images/ and labels/")
    parser.add_argument("--out-root", type=str, default="datasets/traffic_yolo", help="Output YOLO dataset root")
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="configs/training/traffic_dataset.yaml",
        help="Generated YOLO dataset yaml path",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="BDD splits to convert",
    )
    args = parser.parse_args()

    bdd_root = Path(args.bdd_root)
    out_root = Path(args.out_root)

    total = 0
    for split in args.splits:
        converted = convert_split(bdd_root=bdd_root, split=split, out_root=out_root)
        print(f"{split}: converted {converted} frames")
        total += converted

    write_dataset_yaml(out_root=out_root, yaml_path=Path(args.dataset_yaml))
    print(f"Done. Total converted frames: {total}")
    print(f"Dataset YAML written to: {args.dataset_yaml}")


if __name__ == "__main__":
    main()
