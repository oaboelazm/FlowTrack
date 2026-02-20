from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
from ultralytics import YOLO

from src.core.entities import Detection


COCO_TRAFFIC_MAP = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


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


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        valid = [c for c in cfg.include_classes if c in COCO_TRAFFIC_MAP]
        self.target_ids: Set[int] = {COCO_TRAFFIC_MAP[c] for c in valid}

    def infer(self, frame: np.ndarray) -> List[Detection]:
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
            if self.target_ids and cls_id not in self.target_ids:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=str(names.get(cls_id, str(cls_id))),
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
