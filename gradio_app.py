from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
import os
from typing import Deque, Dict, Iterator, List, Tuple
import warnings

import cv2
import gradio as gr
import pandas as pd
import torch

from src.app.pipeline import FlowTrackPipeline
from src.utils.config import load_yaml

# Gradio warns when input video is not browser-compatible.
# Pipeline tries to generate browser-friendly MP4, so this warning is noisy.
warnings.filterwarnings(
    "ignore",
    message="Video does not have browser-compatible container or codec. Converting to mp4.",
    category=UserWarning,
)


def _build_cfg(
    source: str,
    weights: str,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    half: bool,
    tracking_enabled: bool,
    show_heatmap: bool,
    frame_skip: int,
    resize_width: int,
    resize_height: int,
    target_fps: int,
    chunk_mode: bool,
    chunk_seconds: int,
    first_chunk_seconds: int,
    chunk_queue_size: int,
    segment_playback_mode: bool,
) -> Dict:
    cfg = deepcopy(load_yaml("configs/default.yaml"))
    cfg["source"]["input"] = str(source).strip()
    cfg["source"]["chunk_mode"] = bool(chunk_mode)
    cfg["source"]["chunk_seconds"] = int(chunk_seconds)
    cfg["source"]["first_chunk_seconds"] = int(first_chunk_seconds)
    cfg["source"]["chunk_queue_size"] = int(chunk_queue_size)
    cfg["model"]["weights"] = str(weights).strip()
    cfg["model"]["device"] = "" if device == "auto" else str(device).strip()
    cfg["model"]["conf"] = float(conf)
    cfg["model"]["iou"] = float(iou)
    cfg["model"]["imgsz"] = int(imgsz)
    cfg["model"]["half"] = bool(half)
    cfg["tracking"]["enabled"] = bool(tracking_enabled)
    cfg["app"]["show_heatmap"] = bool(show_heatmap)
    cfg["app"]["segment_playback_mode"] = bool(segment_playback_mode)
    cfg["app"]["display"] = False
    cfg["runtime"]["frame_skip"] = int(frame_skip)
    cfg["runtime"]["resize_width"] = int(resize_width)
    cfg["runtime"]["resize_height"] = int(resize_height)
    cfg["app"]["target_fps"] = int(target_fps)
    return cfg


def _metrics_row(output, line_summary: Dict[str, int]) -> Dict[str, float]:
    return {
        "fps": round(float(output.fps), 2),
        "tracks_in_frame": int(output.total_tracks_in_frame),
        "vehicles_per_min": round(float(output.metrics.get("vehicles_per_min", 0.0)), 2),
        "pedestrians_per_min": round(float(output.metrics.get("pedestrians_per_min", 0.0)), 2),
        "traffic_density": round(float(output.metrics.get("traffic_density", 0.0)), 4),
        "avg_speed_kmh": round(float(output.metrics.get("avg_speed_kmh", 0.0)), 2),
        "line_incoming": int(line_summary.get("incoming", 0)),
        "line_outgoing": int(line_summary.get("outgoing", 0)),
    }


def _status_text(selected_device: str) -> str:
    cuda = torch.cuda.is_available()
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        return (
            f"PyTorch {torch.__version__} | CUDA available: True | GPU: {gpu_name} | "
            f"Requested device: {selected_device}"
        )
    return f"PyTorch {torch.__version__} | CUDA available: False | Requested device: {selected_device}"


def run_stream(
    source: str,
    weights: str,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    half: bool,
    tracking_enabled: bool,
    show_heatmap: bool,
    frame_skip: int,
    resize_width: int,
    resize_height: int,
    target_fps: int,
    chunk_mode: bool,
    chunk_seconds: int,
    first_chunk_seconds: int,
    chunk_queue_size: int,
    segment_playback_mode: bool,
) -> Iterator[Tuple]:
    cfg = _build_cfg(
        source=source,
        weights=weights,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        half=half,
        tracking_enabled=tracking_enabled,
        show_heatmap=show_heatmap,
        frame_skip=frame_skip,
        resize_width=resize_width,
        resize_height=resize_height,
        target_fps=target_fps,
        chunk_mode=chunk_mode,
        chunk_seconds=chunk_seconds,
        first_chunk_seconds=first_chunk_seconds,
        chunk_queue_size=chunk_queue_size,
        segment_playback_mode=segment_playback_mode,
    )

    runner = None
    events: Deque[Dict[str, str]] = deque(maxlen=120)
    selected_device = cfg["model"]["device"] or "auto"
    status = _status_text(selected_device)

    empty_metrics = pd.DataFrame([{"fps": 0.0, "tracks_in_frame": 0}])
    empty_events = pd.DataFrame(columns=["time", "track_id", "class", "direction"])
    recent_segment_videos: Deque[str] = deque()

    try:
        runner = FlowTrackPipeline(cfg)
        runner.start()

        while True:
            loop_start = time.time()
            if segment_playback_mode and chunk_mode:
                chunk_output = runner.process_next_chunk()
                if chunk_output is None:
                    continue

                metrics_df = pd.DataFrame(
                    [
                        {
                            "fps": round(float(chunk_output.fps), 2),
                            "tracks_in_frame": int(sum(chunk_output.counts_per_frame.values())),
                            "vehicles_per_min": round(float(chunk_output.metrics.get("vehicles_per_min", 0.0)), 2),
                            "pedestrians_per_min": round(float(chunk_output.metrics.get("pedestrians_per_min", 0.0)), 2),
                            "traffic_density": round(float(chunk_output.metrics.get("traffic_density", 0.0)), 4),
                            "avg_speed_kmh": round(float(chunk_output.metrics.get("avg_speed_kmh", 0.0)), 2),
                            "chunk_frames": int(chunk_output.total_frames),
                        }
                    ]
                )

                if chunk_output.crossing_events:
                    for event in chunk_output.crossing_events:
                        events.appendleft(
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
                                "track_id": str(event.track_id),
                                "class": event.class_name,
                                "direction": event.direction,
                            }
                        )

                recent_segment_videos.append(chunk_output.video_path)
                while len(recent_segment_videos) > 4:
                    stale = recent_segment_videos.popleft()
                    if os.path.exists(stale):
                        os.remove(stale)

                events_df = pd.DataFrame(list(events)) if events else empty_events
                yield chunk_output.video_path, metrics_df, events_df, status
            else:
                output = runner.process_next()
                if output is None:
                    continue

                line_summary = runner.line_counter.summary()
                metrics_df = pd.DataFrame([_metrics_row(output, line_summary)])

                if output.crossing_events:
                    for event in output.crossing_events:
                        events.appendleft(
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
                                "track_id": str(event.track_id),
                                "class": event.class_name,
                                "direction": event.direction,
                            }
                        )

                events_df = pd.DataFrame(list(events)) if events else empty_events
                yield None, metrics_df, events_df, f"{status} | Segment Playback is OFF"

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / max(1, int(target_fps))) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as exc:
        error_df = pd.DataFrame([{"error": str(exc)}])
        yield None, empty_metrics, empty_events, f"{status} | ERROR: {exc}"
    finally:
        if runner is not None:
            runner.close()


with gr.Blocks(title="FlowTrack GPU Monitor") as demo:
    gr.Markdown("## FlowTrack | Real-Time Traffic Monitoring (Gradio)")
    gr.Markdown(
        "Use this app when you need stable live rendering in the same Python process as the model "
        "(better for Colab GPU sessions than Streamlit rerun style)."
    )

    with gr.Row():
        with gr.Column(scale=2):
            source = gr.Textbox(label="Source (0 / RTSP / HLS URL)", value="0")
            weights = gr.Textbox(label="Weights", value="yolov8n.pt")
            device = gr.Dropdown(label="Device", choices=["auto", "cuda:0", "cpu"], value="auto")
            with gr.Row():
                conf = gr.Slider(label="Confidence", minimum=0.1, maximum=0.9, value=0.35, step=0.01)
                iou = gr.Slider(label="IoU", minimum=0.1, maximum=0.9, value=0.45, step=0.01)
            with gr.Row():
                imgsz = gr.Dropdown(label="Image Size", choices=[640, 800, 960, 1280], value=640)
                half = gr.Checkbox(label="FP16 Half", value=False)
            with gr.Row():
                tracking_enabled = gr.Checkbox(label="Enable ByteTrack", value=True)
                show_heatmap = gr.Checkbox(label="Show Heatmap", value=False)
            with gr.Row():
                frame_skip = gr.Slider(label="Frame Skip", minimum=0, maximum=6, value=1, step=1)
                target_fps = gr.Slider(label="UI Target FPS", minimum=2, maximum=20, value=8, step=1)
            with gr.Row():
                chunk_mode = gr.Checkbox(label="Chunked Stream Buffer Mode", value=True)
                chunk_seconds = gr.Slider(label="Chunk Duration (sec)", minimum=5, maximum=60, value=12, step=1)
                first_chunk_seconds = gr.Slider(
                    label="First Chunk Duration (sec)",
                    minimum=5,
                    maximum=90,
                    value=20,
                    step=1,
                )
                chunk_queue_size = gr.Slider(label="Chunk Queue Size", minimum=1, maximum=6, value=3, step=1)
            segment_playback_mode = gr.Checkbox(
                label="Segment Playback Mode (Smooth video, requires chunk mode)",
                value=True,
            )
            with gr.Row():
                resize_width = gr.Slider(label="Resize Width", minimum=640, maximum=1920, value=960, step=32)
                resize_height = gr.Slider(label="Resize Height", minimum=360, maximum=1080, value=540, step=18)

            run_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop")

        with gr.Column(scale=3):
            segment_video = gr.Video(
                label="Segment Playback (Smooth)",
                autoplay=True,
                loop=False,
                height=540,
                show_download_button=False,
                show_share_button=False,
            )
            metrics_table = gr.Dataframe(label="Live Metrics", interactive=False)
            events_table = gr.Dataframe(label="Crossing Events", interactive=False)
            status = gr.Textbox(label="Runtime Status", interactive=False)

    stream_event = run_btn.click(
        fn=run_stream,
        inputs=[
            source,
            weights,
            device,
            conf,
            iou,
            imgsz,
            half,
            tracking_enabled,
            show_heatmap,
            frame_skip,
            resize_width,
            resize_height,
            target_fps,
            chunk_mode,
            chunk_seconds,
            first_chunk_seconds,
            chunk_queue_size,
            segment_playback_mode,
        ],
        outputs=[segment_video, metrics_table, events_table, status],
    )
    stop_btn.click(fn=None, cancels=[stream_event])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
    )
