from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
from ultralytics import YOLO

from src.core.class_names import canonical_class_name
from src.core.entities import SegmentationMask


@dataclass
class SegmenterConfig:
    weights: str
    device: str
    conf: float
    iou: float
    imgsz: int
    half: bool
    max_det: int
    include_classes: List[str]
    strict_classes: bool = True


class YoloSegmenter:
    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.include_classes: Set[str] = {canonical_class_name(c) for c in cfg.include_classes}

    def infer(self, frame: np.ndarray) -> List[SegmentationMask]:
        if self.cfg.strict_classes and not self.include_classes:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            max_det=self.cfg.max_det,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        masks = result.masks
        boxes = result.boxes
        if masks is None or boxes is None or masks.xy is None:
            return []

        names = result.names if hasattr(result, "names") else {}
        items: List[SegmentationMask] = []
        for idx, poly in enumerate(masks.xy):
            if idx >= len(boxes):
                break
            box = boxes[idx]
            cls_id = int(box.cls.item())
            raw_name = str(names.get(cls_id, str(cls_id)))
            class_name = canonical_class_name(raw_name)
            if self.include_classes and class_name not in self.include_classes:
                continue

            polygon = np.asarray(poly, dtype=np.int32)
            if polygon.size < 6:
                continue

            items.append(
                SegmentationMask(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    polygon_xy=polygon,
                )
            )
        return items
