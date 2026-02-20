from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from src.core.entities import CrossingEvent, TrackedObject


VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
TRACKED_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


@dataclass
class AnalyticsConfig:
    congestion_vehicle_threshold: int = 18
    congestion_hold_frames: int = 25
    stop_speed_px_per_sec: float = 10.0
    stop_duration_sec: float = 8.0
    meters_per_pixel: float = 0.05
    enable_heatmap: bool = True


class TrafficAnalytics:
    def __init__(self, cfg: AnalyticsConfig):
        self.cfg = cfg
        self.crossings_1m: Deque[Tuple[float, str]] = deque()
        self.crossings_1h: Deque[Tuple[float, str]] = deque()
        self.crossing_total = Counter()

        self.track_last_pos: Dict[int, Tuple[int, int]] = {}
        self.track_last_ts: Dict[int, float] = {}
        self.track_stop_start: Dict[int, float] = {}

        self.last_speeds_kmh: Dict[int, float] = {}
        self.stopped_alerts = 0

        self.congestion_frames = 0
        self.congestion_alert = False

        self.heatmap: np.ndarray | None = None

    def _trim_deques(self, now: float) -> None:
        while self.crossings_1m and (now - self.crossings_1m[0][0]) > 60:
            self.crossings_1m.popleft()
        while self.crossings_1h and (now - self.crossings_1h[0][0]) > 3600:
            self.crossings_1h.popleft()

    def _update_speed_and_stop(self, tracks: List[TrackedObject], now: float) -> None:
        for tr in tracks:
            if tr.track_id is None:
                continue

            tid = tr.track_id
            center = tr.center
            prev_pos = self.track_last_pos.get(tid)
            prev_ts = self.track_last_ts.get(tid)

            if prev_pos is None or prev_ts is None:
                self.track_last_pos[tid] = center
                self.track_last_ts[tid] = now
                continue

            dt = max(now - prev_ts, 1e-6)
            dx = center[0] - prev_pos[0]
            dy = center[1] - prev_pos[1]
            speed_px = ((dx * dx + dy * dy) ** 0.5) / dt

            speed_mps = speed_px * self.cfg.meters_per_pixel
            speed_kmh = speed_mps * 3.6
            self.last_speeds_kmh[tid] = speed_kmh

            if speed_px < self.cfg.stop_speed_px_per_sec and tr.class_name in VEHICLE_CLASSES:
                if tid not in self.track_stop_start:
                    self.track_stop_start[tid] = now
                elif (now - self.track_stop_start[tid]) >= self.cfg.stop_duration_sec:
                    self.stopped_alerts += 1
                    self.track_stop_start[tid] = now + 9999
            else:
                self.track_stop_start.pop(tid, None)

            self.track_last_pos[tid] = center
            self.track_last_ts[tid] = now

    def _update_congestion(self, tracks: List[TrackedObject]) -> None:
        vehicles_now = sum(1 for t in tracks if t.class_name in VEHICLE_CLASSES)
        if vehicles_now >= self.cfg.congestion_vehicle_threshold:
            self.congestion_frames += 1
        else:
            self.congestion_frames = max(0, self.congestion_frames - 1)

        self.congestion_alert = self.congestion_frames >= self.cfg.congestion_hold_frames

    def _update_heatmap(self, tracks: List[TrackedObject], frame_shape: Tuple[int, int, int]) -> None:
        if not self.cfg.enable_heatmap:
            return

        h, w = frame_shape[:2]
        if self.heatmap is None or self.heatmap.shape != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        for tr in tracks:
            x, y = tr.center
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(self.heatmap, (x, y), 16, 2.5, thickness=-1)

        cv2.GaussianBlur(self.heatmap, (0, 0), 3.0, dst=self.heatmap)

    def update(
        self,
        tracks: List[TrackedObject],
        crossing_events: List[CrossingEvent],
        frame_shape: Tuple[int, int, int],
        ts: float | None = None,
    ) -> Dict[str, float]:
        now = ts if ts is not None else time.time()

        for ev in crossing_events:
            self.crossings_1m.append((ev.timestamp, ev.class_name))
            self.crossings_1h.append((ev.timestamp, ev.class_name))
            self.crossing_total[ev.class_name] += 1

        self._trim_deques(now)
        self._update_speed_and_stop(tracks, now)
        self._update_congestion(tracks)
        self._update_heatmap(tracks, frame_shape)

        in_frame_counter = Counter(t.class_name for t in tracks)
        total_in_frame = len(tracks)
        vehicles_in_frame = sum(v for k, v in in_frame_counter.items() if k in VEHICLE_CLASSES)

        speeds = list(self.last_speeds_kmh.values())
        avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

        metrics: Dict[str, float] = {
            "vehicles_per_min": float(sum(1 for _, c in self.crossings_1m if c in VEHICLE_CLASSES)),
            "vehicles_per_hour": float(sum(1 for _, c in self.crossings_1h if c in VEHICLE_CLASSES)),
            "pedestrians_per_min": float(sum(1 for _, c in self.crossings_1m if c == "person")),
            "traffic_density": float(vehicles_in_frame),
            "avg_speed_kmh": avg_speed,
            "congestion": float(1 if self.congestion_alert else 0),
            "abnormal_stops": float(self.stopped_alerts),
            "active_tracks": float(total_in_frame),
        }

        for cls_name in TRACKED_CLASSES:
            key = f"dist_{cls_name}"
            metrics[key] = float(in_frame_counter.get(cls_name, 0))

        return metrics

    def heatmap_overlay(self, frame_bgr: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        if self.heatmap is None:
            return frame_bgr

        norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(cmap, alpha, frame_bgr, 1.0 - alpha, 0)
