from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]


@dataclass
class TrackedObject(Detection):
    track_id: Optional[int]

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str
    timestamp: float


@dataclass
class PipelineOutput:
    frame_bgr: np.ndarray
    fps: float
    counts_per_frame: Dict[str, int]
    total_tracks_in_frame: int
    crossing_events: List[CrossingEvent]
    metrics: Dict[str, float]


@dataclass
class ChunkPlaybackOutput:
    video_path: str
    fps: float
    total_frames: int
    counts_per_frame: Dict[str, int]
    crossing_events: List[CrossingEvent]
    metrics: Dict[str, float]
