from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import torch

from src.analytics.traffic_analytics import AnalyticsConfig, TrafficAnalytics
from src.core.entities import ChunkPlaybackOutput, CrossingEvent, PipelineOutput, TrackedObject
from src.detection.yolo_detector import DetectorConfig, YoloDetector
from src.events.line_counter import LineCounter, LineCounterConfig
from src.ingestion.chunked_source import ChunkedSourceConfig, ChunkedVideoSourceManager
from src.ingestion.source_manager import SourceConfig, VideoSourceManager
from src.storage.csv_writer import CsvWriter
from src.tracking.bytetrack_tracker import ByteTrackConfig, ByteTrackTracker
from src.utils.logger import setup_logger
from src.visualization.overlay import draw_dashboard, draw_line, draw_tracks


@dataclass
class RuntimeConfig:
    frame_skip: int
    resize_width: int
    resize_height: int


class FlowTrackPipeline:
    @staticmethod
    def _resolve_weights(model_cfg: Dict[str, Any], logger) -> str:
        requested = str(model_cfg.get("weights", "yolov8n.pt")).strip()
        if not requested:
            return "yolov8n.pt"

        p = Path(requested)
        if p.exists():
            return requested

        is_local_like = ("/" in requested) or ("\\" in requested) or requested.startswith(".")
        is_custom_pt = requested.endswith(".pt") and not requested.lower().startswith("yolo")
        if is_local_like or is_custom_pt:
            fallback = "yolov8n.pt"
            logger.warning("Weights '%s' not found. Falling back to '%s'.", requested, fallback)
            return fallback

        return requested

    @staticmethod
    def _resolve_device(model_cfg: Dict[str, Any], logger) -> str:
        requested = str(model_cfg.get("device", "")).strip().lower()
        has_cuda = torch.cuda.is_available()

        if not requested:
            selected = "cuda:0" if has_cuda else "cpu"
            logger.info("Model device auto-selected: %s", selected)
            return selected

        if requested in {"cuda", "cuda:0", "0", "gpu"}:
            if has_cuda:
                return "cuda:0"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if requested in {"cpu", "-1"}:
            return "cpu"

        return requested

    def __init__(self, config: Dict[str, Any]):
        self.log = setup_logger()
        self.cfg = config

        source_cfg = SourceConfig(
            input_source=str(config["source"]["input"]),
            reconnect_delay_sec=float(config["source"].get("reconnect_delay_sec", 2.0)),
            user_agent=str(config["source"].get("user_agent", "")),
            referer=str(config["source"].get("referer", "")),
        )
        runtime_cfg = RuntimeConfig(
            frame_skip=int(config["runtime"].get("frame_skip", 0)),
            resize_width=int(config["runtime"].get("resize_width", 1280)),
            resize_height=int(config["runtime"].get("resize_height", 720)),
        )

        include_classes = list(config.get("classes", {}).get("include", []))
        model_cfg = config.get("model", {})
        resolved_weights = self._resolve_weights(model_cfg, self.log)
        resolved_device = self._resolve_device(model_cfg, self.log)
        self._detector_cfg = DetectorConfig(
            weights=resolved_weights,
            device=resolved_device,
            conf=float(model_cfg.get("conf", 0.35)),
            iou=float(model_cfg.get("iou", 0.45)),
            imgsz=int(model_cfg.get("imgsz", 960)),
            half=bool(model_cfg.get("half", False)),
            max_det=int(model_cfg.get("max_det", 300)),
            include_classes=include_classes,
        )

        self.tracking_enabled = bool(config.get("tracking", {}).get("enabled", True))
        if self.tracking_enabled:
            self.tracker = ByteTrackTracker(
                ByteTrackConfig(
                    weights=resolved_weights,
                    tracker=str(config.get("tracking", {}).get("tracker", "bytetrack.yaml")),
                    device=resolved_device,
                    conf=float(model_cfg.get("conf", 0.35)),
                    iou=float(model_cfg.get("iou", 0.45)),
                    imgsz=int(model_cfg.get("imgsz", 960)),
                    half=bool(model_cfg.get("half", False)),
                    max_det=int(model_cfg.get("max_det", 300)),
                    include_classes=include_classes,
                )
            )
            self.detector = None
        else:
            self.detector = YoloDetector(self._detector_cfg)
            self.tracker = None

        line_cfg = config.get("line_counter", {})
        self.line_counter = LineCounter(
            LineCounterConfig(
                p1=(int(line_cfg.get("x1", 100)), int(line_cfg.get("y1", 350))),
                p2=(int(line_cfg.get("x2", 1180)), int(line_cfg.get("y2", 350))),
                cooldown_sec=float(line_cfg.get("cooldown_sec", 1.5)),
            )
        )

        analytics_cfg = config.get("analytics", {})
        self.analytics = TrafficAnalytics(
            AnalyticsConfig(
                congestion_vehicle_threshold=int(analytics_cfg.get("congestion_vehicle_threshold", 18)),
                congestion_hold_frames=int(analytics_cfg.get("congestion_hold_frames", 25)),
                stop_speed_px_per_sec=float(analytics_cfg.get("stop_speed_px_per_sec", 10.0)),
                stop_duration_sec=float(analytics_cfg.get("stop_duration_sec", 8.0)),
                meters_per_pixel=float(analytics_cfg.get("meters_per_pixel", 0.05)),
                enable_heatmap=bool(analytics_cfg.get("enable_heatmap", True)),
            )
        )

        storage_cfg = config.get("storage", {})
        self.storage_enabled = bool(storage_cfg.get("enabled", True))
        self.storage_interval_sec = float(storage_cfg.get("write_interval_sec", 1.0))
        self.csv = CsvWriter(
            metrics_path=str(storage_cfg.get("metrics_csv", "outputs/metrics.csv")),
            events_path=str(storage_cfg.get("events_csv", "outputs/crossings.csv")),
        )

        source_settings = config.get("source", {})
        self.chunk_mode = bool(source_settings.get("chunk_mode", False))
        if self.chunk_mode:
            self.log.info(
                "Chunk mode enabled: chunk_seconds=%s, queue_size=%s, temp_dir=%s",
                source_settings.get("chunk_seconds", 30),
                source_settings.get("chunk_queue_size", 3),
                source_settings.get("chunk_tmp_dir", "outputs/chunks"),
            )
            self.source = ChunkedVideoSourceManager(
                ChunkedSourceConfig(
                    input_source=source_cfg.input_source,
                    reconnect_delay_sec=source_cfg.reconnect_delay_sec,
                    user_agent=source_cfg.user_agent,
                    referer=source_cfg.referer,
                    chunk_seconds=float(source_settings.get("chunk_seconds", 30.0)),
                    max_queue_size=int(source_settings.get("chunk_queue_size", 3)),
                    temp_dir=str(source_settings.get("chunk_tmp_dir", "outputs/chunks")),
                )
            )
        else:
            self.source = VideoSourceManager(source_cfg)
        self.runtime = runtime_cfg
        self.window_name = str(config.get("app", {}).get("window_name", "FlowTrack"))
        self.display = bool(config.get("app", {}).get("display", True))
        self.print_counts = bool(config.get("app", {}).get("print_counts", False))
        self.show_heatmap = bool(config.get("app", {}).get("show_heatmap", False))
        self.segment_playback_mode = bool(config.get("app", {}).get("segment_playback_mode", False))
        self.processed_chunks_dir = Path(str(source_settings.get("processed_chunk_dir", "outputs/processed_chunks")))
        self.processed_chunks_dir.mkdir(parents=True, exist_ok=True)

        self.frame_idx = 0
        self.fps = 0.0
        self.t_prev = time.time()
        self.t_last_store = 0.0

    def start(self) -> None:
        if not self.source.connect():
            raise RuntimeError(f"Unable to connect source: {self.cfg['source']['input']}")
        self.log.info("Source connected: %s", self.cfg["source"]["input"])

    def _process_tracks(self, frame) -> list[TrackedObject]:
        if self.tracker is not None:
            try:
                return self.tracker.track(frame)
            except ModuleNotFoundError as e:
                if e.name == "lap":
                    self.log.warning("Tracking dependency 'lap' is missing. Falling back to detection-only mode.")
                    self.tracker = None
                    if self.detector is None:
                        self.detector = YoloDetector(self._detector_cfg)
                else:
                    raise

        assert self.detector is not None
        dets = self.detector.infer(frame)
        return [
            TrackedObject(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy,
                track_id=None,
            )
            for d in dets
        ]

    @staticmethod
    def _counts_by_class(tracks: list[TrackedObject]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tr in tracks:
            counts[tr.class_name] = counts.get(tr.class_name, 0) + 1
        return counts

    def _visualize_frame(self, frame, tracks: list[TrackedObject], counts: Dict[str, int], metrics: Dict[str, float]):
        vis = frame
        if self.show_heatmap:
            vis = self.analytics.heatmap_overlay(vis)
        vis = draw_tracks(vis, tracks)
        vis = draw_line(vis, self.line_counter.cfg.p1, self.line_counter.cfg.p2)
        vis = draw_dashboard(vis, self.fps, counts, self.line_counter.summary(), metrics)
        return vis

    def process_next(self) -> Optional[PipelineOutput]:
        ok, frame = self.source.read()
        if not ok or frame is None:
            if self.chunk_mode:
                return None
            self.log.warning("Frame read failed. Reconnecting...")
            if not self.source.reconnect():
                return None
            return None

        self.frame_idx += 1
        if self.runtime.frame_skip > 0 and self.frame_idx % (self.runtime.frame_skip + 1) != 0:
            return None

        frame = cv2.resize(frame, (self.runtime.resize_width, self.runtime.resize_height))
        tracks = self._process_tracks(frame)
        counts = self._counts_by_class(tracks)

        now = time.time()
        dt = max(now - self.t_prev, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.t_prev = now

        crossing_events: list[CrossingEvent] = self.line_counter.update(tracks, now)
        metrics = self.analytics.update(tracks, crossing_events, frame.shape, now)

        vis = self._visualize_frame(frame, tracks, counts, metrics)

        if self.storage_enabled and (now - self.t_last_store >= self.storage_interval_sec):
            metrics_row: Dict[str, float] = {"timestamp": now, "fps": float(self.fps)}
            metrics_row.update(metrics)
            metrics_row["line_incoming"] = float(self.line_counter.summary().get("incoming", 0))
            metrics_row["line_outgoing"] = float(self.line_counter.summary().get("outgoing", 0))
            self.csv.append_metrics(metrics_row)
            self.csv.append_events(crossing_events)
            self.t_last_store = now

        if self.print_counts:
            self.log.info("Counts: %s | Line: %s", counts, self.line_counter.summary())

        return PipelineOutput(
            frame_bgr=vis,
            fps=self.fps,
            counts_per_frame=counts,
            total_tracks_in_frame=len(tracks),
            crossing_events=crossing_events,
            metrics=metrics,
        )

    def process_next_chunk(self) -> Optional[ChunkPlaybackOutput]:
        if not self.chunk_mode:
            return None
        if not isinstance(self.source, ChunkedVideoSourceManager):
            return None

        chunk_path = self.source.pop_chunk(timeout_sec=1.0)
        if chunk_path is None:
            return None

        cap = cv2.VideoCapture(str(chunk_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            chunk_path.unlink(missing_ok=True)
            return None

        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        out_fps = src_fps if src_fps > 1.0 else 20.0

        out_path = self.processed_chunks_dir / f"{chunk_path.stem}_det.mp4"
        writer = None

        total_frames = 0
        counts: Dict[str, int] = {}
        metrics: Dict[str, float] = {}
        crossing_events_all: list[CrossingEvent] = []
        start_ts = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame = cv2.resize(frame, (self.runtime.resize_width, self.runtime.resize_height))
                tracks = self._process_tracks(frame)
                counts = self._counts_by_class(tracks)

                now = time.time()
                dt = max(now - self.t_prev, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
                self.t_prev = now

                crossing_events = self.line_counter.update(tracks, now)
                crossing_events_all.extend(crossing_events)
                metrics = self.analytics.update(tracks, crossing_events, frame.shape, now)

                vis = self._visualize_frame(frame, tracks, counts, metrics)

                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        out_fps,
                        (w, h),
                    )
                writer.write(vis)
                total_frames += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            chunk_path.unlink(missing_ok=True)

        if total_frames == 0:
            out_path.unlink(missing_ok=True)
            return None

        end_ts = time.time()
        if self.storage_enabled and (end_ts - self.t_last_store >= self.storage_interval_sec):
            metrics_row: Dict[str, float] = {"timestamp": end_ts, "fps": float(self.fps)}
            metrics_row.update(metrics)
            metrics_row["line_incoming"] = float(self.line_counter.summary().get("incoming", 0))
            metrics_row["line_outgoing"] = float(self.line_counter.summary().get("outgoing", 0))
            metrics_row["chunk_process_sec"] = float(end_ts - start_ts)
            metrics_row["chunk_frames"] = float(total_frames)
            self.csv.append_metrics(metrics_row)
            self.csv.append_events(crossing_events_all)
            self.t_last_store = end_ts

        return ChunkPlaybackOutput(
            video_path=str(out_path),
            fps=self.fps,
            total_frames=total_frames,
            counts_per_frame=counts,
            crossing_events=crossing_events_all,
            metrics=metrics,
        )

    def run(self) -> None:
        self.start()
        self.log.info("Press 'q' to quit")
        try:
            while True:
                output = self.process_next()
                if output is None:
                    continue

                if self.display:
                    cv2.imshow(self.window_name, output.frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        finally:
            self.close()

    def close(self) -> None:
        self.source.release()
        if self.display:
            cv2.destroyAllWindows()
        self.log.info("Pipeline stopped")
