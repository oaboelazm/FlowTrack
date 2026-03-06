from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
from ultralytics import YOLO

from src.core.class_names import canonical_class_name
from src.core.entities import Detection


@dataclass
class DetectorConfig:
    weights: str
    device: str
    conf: float
    iou: float
    imgsz: int
    half: bool
    max_det: int
    include_classes: List[str]
    strict_classes: bool = True


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.include_classes: Set[str] = {canonical_class_name(c) for c in cfg.include_classes}

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if self.strict_classes and not self.include_classes:
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
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names if hasattr(result, "names") else {}
        detections: List[Detection] = []

        for box in boxes:
            cls_id = int(box.cls.item())
            raw_name = str(names.get(cls_id, str(cls_id)))
            class_name = canonical_class_name(raw_name)
            if self.include_classes and class_name not in self.include_classes:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=[x1, y1, x2, y2],
                )
            )
        return detections

    @staticmethod
    def counts_by_class(detections: List[Detection]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
