from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
from ultralytics import YOLO

from src.core.entities import TrackedObject
from src.detection.yolo_detector import COCO_TRAFFIC_MAP


@dataclass
class ByteTrackConfig:
    weights: str
    tracker: str
    device: str
    conf: float
    iou: float
    imgsz: int
    half: bool
    max_det: int
    include_classes: List[str]


class ByteTrackTracker:
    def __init__(self, cfg: ByteTrackConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        valid = [c for c in cfg.include_classes if c in COCO_TRAFFIC_MAP]
        self.target_ids: Set[int] = {COCO_TRAFFIC_MAP[c] for c in valid}

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.cfg.tracker,
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
        tracks: List[TrackedObject] = []

        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, ids):
            cls_id = int(box.cls.item())
            if self.target_ids and cls_id not in self.target_ids:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            tracks.append(
                TrackedObject(
                    class_id=cls_id,
                    class_name=str(names.get(cls_id, str(cls_id))),
                    confidence=float(box.conf.item()),
                    bbox_xyxy=[x1, y1, x2, y2],
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        return tracks

    @staticmethod
    def counts_by_class(tracks: List[TrackedObject]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tr in tracks:
            counts[tr.class_name] = counts.get(tr.class_name, 0) + 1
        return counts
