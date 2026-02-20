from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class SourceConfig:
    input_source: str
    reconnect_delay_sec: float = 2.0


class VideoSourceManager:
    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None

    def _parse_source(self):
        src = str(self.cfg.input_source).strip()
        if src.isdigit():
            return int(src)
        return src

    def connect(self) -> bool:
        source = self._parse_source()
        self.cap = cv2.VideoCapture(source)
        return bool(self.cap and self.cap.isOpened())

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def reconnect(self) -> bool:
        self.release()
        time.sleep(self.cfg.reconnect_delay_sec)
        return self.connect()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
