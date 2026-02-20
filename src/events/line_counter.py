from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.core.entities import CrossingEvent, TrackedObject


@dataclass
class LineCounterConfig:
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    cooldown_sec: float = 1.5


class LineCounter:
    def __init__(self, cfg: LineCounterConfig):
        self.cfg = cfg
        self.prev_side: Dict[int, float] = {}
        self.last_count_ts: Dict[int, float] = {}
        self.in_count = 0
        self.out_count = 0

    def _side_value(self, point: Tuple[int, int]) -> float:
        x, y = point
        x1, y1 = self.cfg.p1
        x2, y2 = self.cfg.p2
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def _direction(self, prev: float, curr: float) -> str:
        if prev < 0 <= curr:
            return "incoming"
        return "outgoing"

    def update(self, tracks: List[TrackedObject], ts: float | None = None) -> List[CrossingEvent]:
        now = ts if ts is not None else time.time()
        events: List[CrossingEvent] = []

        for tr in tracks:
            if tr.track_id is None:
                continue

            side = self._side_value(tr.center)
            if abs(side) < 1e-5:
                continue

            prev = self.prev_side.get(tr.track_id)
            self.prev_side[tr.track_id] = side
            if prev is None or prev * side > 0:
                continue

            last_ts = self.last_count_ts.get(tr.track_id, 0.0)
            if (now - last_ts) < self.cfg.cooldown_sec:
                continue

            direction = self._direction(prev, side)
            if direction == "incoming":
                self.in_count += 1
            else:
                self.out_count += 1

            self.last_count_ts[tr.track_id] = now
            events.append(
                CrossingEvent(
                    track_id=tr.track_id,
                    class_name=tr.class_name,
                    direction=direction,
                    timestamp=now,
                )
            )

        return events

    def summary(self) -> Dict[str, int]:
        return {"incoming": self.in_count, "outgoing": self.out_count}
