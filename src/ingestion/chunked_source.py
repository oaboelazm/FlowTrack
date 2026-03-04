from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Deque, Optional
from uuid import uuid4

import cv2

from src.ingestion.source_manager import SourceConfig, VideoSourceManager


@dataclass
class ChunkedSourceConfig:
    input_source: str
    reconnect_delay_sec: float = 2.0
    user_agent: str = ""
    referer: str = ""
    chunk_seconds: float = 30.0
    max_queue_size: int = 3
    temp_dir: str = "outputs/chunks"
    writer_fps: float = 20.0
    warmup_timeout_sec: float = 45.0


class ChunkedVideoSourceManager:
    """
    Layered buffered source:
    - Producer thread reads the live stream and writes short video chunks.
    - Consumer reads frames from completed chunks sequentially.
    - Played chunks are deleted immediately after consumption.
    """

    def __init__(self, cfg: ChunkedSourceConfig):
        self.cfg = cfg
        self.temp_dir = Path(cfg.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._source = VideoSourceManager(
            SourceConfig(
                input_source=cfg.input_source,
                reconnect_delay_sec=cfg.reconnect_delay_sec,
                user_agent=cfg.user_agent,
                referer=cfg.referer,
            )
        )

        self._segments: Deque[Path] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None

        self._current_segment_cap: Optional[cv2.VideoCapture] = None
        self._current_segment_path: Optional[Path] = None

    def _cleanup_temp_dir(self) -> None:
        for p in self.temp_dir.glob("chunk_*.mp4"):
            p.unlink(missing_ok=True)

    def connect(self) -> bool:
        self.release()
        self._stop_event.clear()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_temp_dir()

        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="flowtrack-segment-producer",
            daemon=True,
        )
        self._producer_thread.start()

        deadline = time.time() + max(3.0, float(self.cfg.warmup_timeout_sec))
        while time.time() < deadline and not self._stop_event.is_set():
            with self._cv:
                if self._segments:
                    return True
            time.sleep(0.1)
        return False

    def _make_chunk_path(self) -> Path:
        ts_ms = int(time.time() * 1000)
        return self.temp_dir / f"chunk_{ts_ms}_{uuid4().hex[:8]}.mp4"

    def _push_segment(self, path: Path) -> None:
        with self._cv:
            while len(self._segments) >= max(1, int(self.cfg.max_queue_size)):
                stale = self._segments.popleft()
                stale.unlink(missing_ok=True)
            self._segments.append(path)
            self._cv.notify_all()

    def _producer_loop(self) -> None:
        writer: Optional[cv2.VideoWriter] = None
        chunk_path: Optional[Path] = None
        chunk_start: Optional[float] = None
        frames_written = 0

        try:
            while not self._stop_event.is_set():
                if self._source.cap is None or not self._source.cap.isOpened():
                    if not self._source.connect():
                        time.sleep(max(0.2, float(self.cfg.reconnect_delay_sec)))
                        continue

                ok, frame = self._source.read()
                if not ok or frame is None:
                    self._source.reconnect()
                    continue

                if writer is None:
                    h, w = frame.shape[:2]
                    src_fps = float(self._source.cap.get(cv2.CAP_PROP_FPS)) if self._source.cap is not None else 0.0
                    out_fps = src_fps if src_fps > 1.0 else float(self.cfg.writer_fps)
                    chunk_path = self._make_chunk_path()
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(chunk_path), fourcc, out_fps, (w, h))
                    chunk_start = time.monotonic()
                    frames_written = 0

                writer.write(frame)
                frames_written += 1

                elapsed = time.monotonic() - (chunk_start or time.monotonic())
                if elapsed >= max(1.0, float(self.cfg.chunk_seconds)):
                    writer.release()
                    writer = None
                    if chunk_path is not None and frames_written > 0:
                        self._push_segment(chunk_path)
                    chunk_path = None
                    chunk_start = None
                    frames_written = 0
        finally:
            if writer is not None:
                writer.release()
                if chunk_path is not None and frames_written > 0:
                    self._push_segment(chunk_path)
            self._source.release()

    def _pop_segment(self, timeout_sec: float = 0.4) -> Optional[Path]:
        deadline = time.monotonic() + max(0.05, timeout_sec)
        with self._cv:
            while not self._segments and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)
            if self._segments:
                return self._segments.popleft()
        return None

    def _open_next_segment(self) -> bool:
        path = self._pop_segment(timeout_sec=0.8)
        if path is None:
            return False
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            path.unlink(missing_ok=True)
            return False
        self._current_segment_cap = cap
        self._current_segment_path = path
        return True

    def pop_chunk(self, timeout_sec: float = 0.8) -> Optional[Path]:
        """
        Pop next ready chunk path for external chunk-level processing.
        Caller owns deletion of the returned file path.
        """
        return self._pop_segment(timeout_sec=timeout_sec)

    def _close_current_segment(self) -> None:
        if self._current_segment_cap is not None:
            self._current_segment_cap.release()
            self._current_segment_cap = None
        if self._current_segment_path is not None:
            self._current_segment_path.unlink(missing_ok=True)
            self._current_segment_path = None

    def read(self):
        tries = 0
        while tries < 4 and not self._stop_event.is_set():
            if self._current_segment_cap is None:
                if not self._open_next_segment():
                    return False, None

            assert self._current_segment_cap is not None
            ok, frame = self._current_segment_cap.read()
            if ok and frame is not None:
                return True, frame

            self._close_current_segment()
            tries += 1
        return False, None

    def reconnect(self) -> bool:
        # Producer handles live source reconnect. Consumer waits for next ready segment.
        return True

    def release(self) -> None:
        self._stop_event.set()
        with self._cv:
            self._cv.notify_all()

        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
        self._producer_thread = None

        self._close_current_segment()

        with self._cv:
            while self._segments:
                p = self._segments.popleft()
                p.unlink(missing_ok=True)
