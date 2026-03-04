from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.core.entities import SegmentationMask, TrackedObject


COLOR_MAP = {
    "person": (36, 255, 12),
    "bicycle": (255, 153, 51),
    "car": (0, 215, 255),
    "motorcycle": (204, 102, 255),
    "bus": (255, 99, 71),
    "truck": (0, 140, 255),
}


def draw_tracks(frame: np.ndarray, tracks: List[TrackedObject]) -> np.ndarray:
    canvas = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox_xyxy
        color = COLOR_MAP.get(tr.class_name, (255, 255, 255))
        tid = f"ID {tr.track_id}" if tr.track_id is not None else "ID ?"
        label = f"{tid} | {tr.class_name} {tr.confidence:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, tr.center, 3, color, -1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def draw_segmentations(frame: np.ndarray, masks: List[SegmentationMask], alpha: float = 0.35) -> np.ndarray:
    if not masks:
        return frame

    canvas = frame.copy()
    overlay = frame.copy()

    for item in masks:
        color = COLOR_MAP.get(item.class_name, (200, 200, 200))
        poly = item.polygon_xy.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=2)

        x, y = int(item.polygon_xy[0][0]), int(item.polygon_xy[0][1])
        label = f"{item.class_name} seg {item.confidence:.2f}"
        cv2.putText(canvas, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


def draw_line(frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> np.ndarray:
    canvas = frame.copy()
    cv2.line(canvas, p1, p2, (0, 255, 255), 2)
    cv2.putText(canvas, "Count Line", (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def draw_dashboard(
    frame: np.ndarray,
    fps: float,
    counts: Dict[str, int],
    line_counts: Dict[str, int],
    metrics: Dict[str, float],
) -> np.ndarray:
    canvas = frame.copy()
    panel_w = 380
    panel_h = 280

    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    y = 34
    cv2.putText(canvas, "FlowTrack Analytics", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 26
    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
    y += 24

    total = sum(counts.values())
    cv2.putText(canvas, f"In-frame objects: {total}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 22
    cv2.putText(canvas, f"Incoming: {line_counts.get('incoming', 0)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 2)
    y += 22
    cv2.putText(canvas, f"Outgoing: {line_counts.get('outgoing', 0)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 120), 2)
    y += 22

    cv2.putText(
        canvas,
        f"Veh/min: {metrics.get('vehicles_per_min', 0):.0f} | Ped/min: {metrics.get('pedestrians_per_min', 0):.0f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
    )
    y += 22
    cv2.putText(
        canvas,
        f"Veh/hour: {metrics.get('vehicles_per_hour', 0):.0f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
    )
    y += 22
    cv2.putText(
        canvas,
        f"Density: {metrics.get('traffic_density', 0):.0f} | Avg speed: {metrics.get('avg_speed_kmh', 0):.1f} km/h",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
    )
    y += 22

    cong = "YES" if metrics.get("congestion", 0) > 0 else "NO"
    cv2.putText(
        canvas,
        f"Congestion: {cong} | Abnormal stops: {metrics.get('abnormal_stops', 0):.0f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 220, 220),
        2,
    )

    x = 20
    y = 220
    for cls_name, count in sorted(counts.items()):
        label = f"{cls_name}: {count}"
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP.get(cls_name, (240, 240, 240)), 2)
        y += 20
        if y > 270:
            x = 200
            y = 220

    return canvas
