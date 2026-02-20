from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class SourceConfig:
    input_source: str
    reconnect_delay_sec: float = 2.0
    user_agent: str = ""
    referer: str = ""


class VideoSourceManager:
    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None

    def _parse_source(self):
        src = str(self.cfg.input_source).strip()
        # Normalize pasted URLs that may include newlines/spaces.
        if src.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            src = "".join(src.split())
        if src.isdigit():
            return int(src)
        return src

    def _apply_ffmpeg_options(self, source) -> None:
        if not isinstance(source, str):
            return

        src = source.lower()
        if not src.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            return

        user_agent = self.cfg.user_agent.strip()
        referer = self.cfg.referer.strip()

        # EarthCam HLS often requires request headers to avoid 403.
        if "earthcam.com" in src:
            if not user_agent:
                user_agent = "Mozilla/5.0"
            if not referer:
                referer = "https://www.earthcam.com/"

        opts = []
        if user_agent:
            opts.append(f"user_agent;{user_agent}")
        if referer:
            opts.append(f"referer;{referer}")

        if opts:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    def connect(self) -> bool:
        source = self._parse_source()
        self._apply_ffmpeg_options(source)
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
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
