from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from src.core.entities import CrossingEvent


class CsvWriter:
    def __init__(self, metrics_path: str = "outputs/metrics.csv", events_path: str = "outputs/crossings.csv"):
        self.metrics_path = Path(metrics_path)
        self.events_path = Path(events_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def append_metrics(self, row: Dict[str, float]) -> None:
        file_exists = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def append_events(self, events: Iterable[CrossingEvent]) -> None:
        events_list: List[CrossingEvent] = list(events)
        if not events_list:
            return

        file_exists = self.events_path.exists()
        with self.events_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "track_id", "class_name", "direction"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for ev in events_list:
                writer.writerow(
                    {
                        "timestamp": ev.timestamp,
                        "track_id": ev.track_id,
                        "class_name": ev.class_name,
                        "direction": ev.direction,
                    }
                )
