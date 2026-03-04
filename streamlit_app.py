from __future__ import annotations

import time
from copy import deepcopy

import cv2
import pandas as pd
import streamlit as st
import torch

from src.app.pipeline import FlowTrackPipeline
from src.utils.config import load_yaml

st.set_page_config(page_title="FlowTrack", page_icon="🚦", layout="wide")


def init_state() -> None:
    if "runner" not in st.session_state:
        st.session_state.runner = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_frame_rgb" not in st.session_state:
        st.session_state.last_frame_rgb = None
    if "last_video_path" not in st.session_state:
        st.session_state.last_video_path = None
    if "recent_video_paths" not in st.session_state:
        st.session_state.recent_video_paths = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {
            "fps": 0.0,
            "vehicles_per_min": 0.0,
            "pedestrians_per_min": 0.0,
            "line_in": 0,
            "line_out": 0,
            "avg_speed_kmh": 0.0,
        }


def build_config() -> dict:
    cfg = load_yaml("configs/default.yaml")
    cfg = deepcopy(cfg)

    st.sidebar.header("FlowTrack Controls")
    source = st.sidebar.text_input("Camera Source", value=str(cfg["source"].get("input", "0")))
    chunk_mode = st.sidebar.checkbox("Chunked Stream Buffer Mode", value=bool(cfg["source"].get("chunk_mode", False)))
    chunk_seconds = st.sidebar.slider(
        "Chunk Duration (seconds)",
        5,
        60,
        int(cfg["source"].get("chunk_seconds", 30)),
        1,
    )
    chunk_queue_size = st.sidebar.slider(
        "Chunk Queue Size",
        1,
        6,
        int(cfg["source"].get("chunk_queue_size", 3)),
        1,
    )
    weights = st.sidebar.text_input("Weights", value=str(cfg["model"].get("weights", "yolov8n.pt")))
    default_device = str(cfg["model"].get("device", "")).strip()
    if not default_device:
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = st.sidebar.text_input("Device (cpu / cuda:0)", value=default_device)

    conf = st.sidebar.slider("Confidence", 0.1, 0.9, float(cfg["model"].get("conf", 0.35)), 0.01)
    iou = st.sidebar.slider("IoU", 0.1, 0.9, float(cfg["model"].get("iou", 0.45)), 0.01)
    imgsz = st.sidebar.selectbox("Image Size", [640, 800, 960, 1280], index=2)
    use_half = st.sidebar.checkbox("FP16 (Half)", value=bool(cfg["model"].get("half", False)))

    tracking_enabled = st.sidebar.checkbox("Enable Tracking (ByteTrack)", value=bool(cfg["tracking"].get("enabled", True)))
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=bool(cfg["app"].get("show_heatmap", False)))
    segment_playback_mode = st.sidebar.checkbox(
        "Segment Playback Mode (Smooth, requires chunk mode)",
        value=bool(cfg["app"].get("segment_playback_mode", False)),
    )
    smooth_mode = st.sidebar.checkbox("Smooth Stream Mode (Recommended)", value=True)
    refresh_ms = st.sidebar.slider("UI Refresh (ms)", 80, 500, int(cfg["app"].get("refresh_ms", 140)), 20)

    st.sidebar.subheader("Counting Line")
    x1 = st.sidebar.number_input("x1", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("x1", 100)))
    y1 = st.sidebar.number_input("y1", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("y1", 360)))
    x2 = st.sidebar.number_input("x2", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("x2", 1180)))
    y2 = st.sidebar.number_input("y2", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("y2", 360)))

    cfg["source"]["input"] = str(source).strip()
    cfg["source"]["chunk_mode"] = bool(chunk_mode)
    cfg["source"]["chunk_seconds"] = int(chunk_seconds)
    cfg["source"]["chunk_queue_size"] = int(chunk_queue_size)
    cfg["model"]["weights"] = weights
    cfg["model"]["conf"] = conf
    cfg["model"]["iou"] = iou
    cfg["model"]["imgsz"] = imgsz
    cfg["model"]["half"] = use_half
    cfg["model"]["device"] = device
    cfg["tracking"]["enabled"] = tracking_enabled
    cfg["app"]["show_heatmap"] = show_heatmap
    cfg["app"]["segment_playback_mode"] = segment_playback_mode
    cfg["app"]["display"] = False
    cfg["app"]["refresh_ms"] = int(refresh_ms)
    cfg["line_counter"]["x1"] = int(x1)
    cfg["line_counter"]["y1"] = int(y1)
    cfg["line_counter"]["x2"] = int(x2)
    cfg["line_counter"]["y2"] = int(y2)

    if smooth_mode:
        cfg["app"]["show_heatmap"] = False
        cfg["runtime"]["frame_skip"] = max(int(cfg["runtime"].get("frame_skip", 0)), 1)
        cfg["model"]["imgsz"] = min(int(cfg["model"]["imgsz"]), 640)
        cfg["app"]["refresh_ms"] = max(int(cfg["app"]["refresh_ms"]), 140)

    return cfg


def start_runner(cfg: dict) -> None:
    if st.session_state.runner is not None:
        try:
            st.session_state.runner.close()
        except Exception:
            pass

    try:
        runner = FlowTrackPipeline(cfg)
        runner.start()
        st.session_state.runner = runner
        st.session_state.running = True
    except FileNotFoundError as e:
        fallback_cfg = deepcopy(cfg)
        fallback_cfg["model"]["weights"] = "yolov8n.pt"
        st.warning(f"{e}. Falling back to yolov8n.pt")
        runner = FlowTrackPipeline(fallback_cfg)
        runner.start()
        st.session_state.runner = runner
        st.session_state.running = True
    except RuntimeError as e:
        st.error(
            f"{e}\n\nTips:\n"
            "- For URL streams, paste the URL in one single line.\n"
            "- For EarthCam/HLS token URLs, refresh and use a new tokenized link.\n"
            "- On Streamlit Cloud, webcam source 0 is not available."
        )


def stop_runner() -> None:
    if st.session_state.runner is not None:
        st.session_state.runner.close()
    st.session_state.runner = None
    st.session_state.running = False


def main() -> None:
    init_state()
    st.title("FlowTrack | Smart Traffic Monitoring")

    cfg = build_config()
    selected_device = str(cfg["model"].get("device", "")).strip()
    cuda_available = torch.cuda.is_available()
    if selected_device.startswith("cuda") and not cuda_available:
        st.warning("CUDA is not available in this runtime. Inference will run on CPU.")
    st.caption(
        f"Runtime device: `{selected_device or 'auto'}` | "
        f"CUDA available: `{cuda_available}` | "
        f"PyTorch: `{torch.__version__}`"
    )

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Start", width="stretch"):
        start_runner(cfg)
    if c2.button("Stop", width="stretch"):
        stop_runner()
    c3.caption("Phase 1-5: detection, tracking, line counting, analytics, congestion/stop alerts, heatmap")
    st.caption(
        f"Live settings: tracking={cfg['tracking']['enabled']} | "
        f"imgsz={cfg['model']['imgsz']} | frame_skip={cfg['runtime']['frame_skip']} | "
        f"chunk_mode={cfg['source'].get('chunk_mode', False)} | "
        f"chunk_seconds={cfg['source'].get('chunk_seconds', 30)}"
    )

    frame_placeholder = st.empty()
    m1, m2, m3, m4, m5 = st.columns(5)

    chart_placeholder = st.empty()
    events_placeholder = st.empty()

    output = None
    chunk_output = None
    should_rerun = False
    if st.session_state.running and st.session_state.runner is not None:
        if cfg["source"].get("chunk_mode", False) and cfg["app"].get("segment_playback_mode", False):
            chunk_output = st.session_state.runner.process_next_chunk()
            if chunk_output is not None:
                st.session_state.last_video_path = chunk_output.video_path
                st.session_state.recent_video_paths.append(chunk_output.video_path)
                st.session_state.recent_video_paths = st.session_state.recent_video_paths[-4:]

                for old_path in st.session_state.recent_video_paths[:-2]:
                    try:
                        from pathlib import Path

                        Path(old_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                line_in = st.session_state.runner.line_counter.summary().get("incoming", 0)
                line_out = st.session_state.runner.line_counter.summary().get("outgoing", 0)
                st.session_state.last_metrics = {
                    "fps": float(chunk_output.fps),
                    "vehicles_per_min": float(chunk_output.metrics.get("vehicles_per_min", 0)),
                    "pedestrians_per_min": float(chunk_output.metrics.get("pedestrians_per_min", 0)),
                    "line_in": int(line_in),
                    "line_out": int(line_out),
                    "avg_speed_kmh": float(chunk_output.metrics.get("avg_speed_kmh", 0)),
                }

                row = {
                    "ts": time.time(),
                    "vehicles_per_min": chunk_output.metrics.get("vehicles_per_min", 0.0),
                    "pedestrians_per_min": chunk_output.metrics.get("pedestrians_per_min", 0.0),
                    "traffic_density": chunk_output.metrics.get("traffic_density", 0.0),
                    "avg_speed_kmh": chunk_output.metrics.get("avg_speed_kmh", 0.0),
                }
                st.session_state.history.append(row)
                st.session_state.history = st.session_state.history[-300:]

                if chunk_output.crossing_events:
                    for ev in chunk_output.crossing_events:
                        st.session_state.events.append(
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(ev.timestamp)),
                                "track_id": ev.track_id,
                                "class": ev.class_name,
                                "direction": ev.direction,
                            }
                        )
                    st.session_state.events = st.session_state.events[-100:]
            should_rerun = True
        else:
            output = st.session_state.runner.process_next()
            if output is not None:
                frame_rgb = cv2.cvtColor(output.frame_bgr, cv2.COLOR_BGR2RGB)
                st.session_state.last_frame_rgb = frame_rgb

                line_in = st.session_state.runner.line_counter.summary().get("incoming", 0)
                line_out = st.session_state.runner.line_counter.summary().get("outgoing", 0)

                st.session_state.last_metrics = {
                    "fps": float(output.fps),
                    "vehicles_per_min": float(output.metrics.get("vehicles_per_min", 0)),
                    "pedestrians_per_min": float(output.metrics.get("pedestrians_per_min", 0)),
                    "line_in": int(line_in),
                    "line_out": int(line_out),
                    "avg_speed_kmh": float(output.metrics.get("avg_speed_kmh", 0)),
                }

                row = {
                    "ts": time.time(),
                    "vehicles_per_min": output.metrics.get("vehicles_per_min", 0.0),
                    "pedestrians_per_min": output.metrics.get("pedestrians_per_min", 0.0),
                    "traffic_density": output.metrics.get("traffic_density", 0.0),
                    "avg_speed_kmh": output.metrics.get("avg_speed_kmh", 0.0),
                }
                st.session_state.history.append(row)
                st.session_state.history = st.session_state.history[-300:]

                if output.crossing_events:
                    for ev in output.crossing_events:
                        st.session_state.events.append(
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(ev.timestamp)),
                                "track_id": ev.track_id,
                                "class": ev.class_name,
                                "direction": ev.direction,
                            }
                        )
                    st.session_state.events = st.session_state.events[-100:]
            should_rerun = True

    # Keep UI stable: always show the last good frame and last metrics.
    if cfg["source"].get("chunk_mode", False) and cfg["app"].get("segment_playback_mode", False):
        if st.session_state.last_video_path:
            frame_placeholder.video(st.session_state.last_video_path)
        else:
            frame_placeholder.info("Waiting for first processed segment...")
    elif st.session_state.last_frame_rgb is not None:
        frame_placeholder.image(
            st.session_state.last_frame_rgb,
            channels="RGB",
            width="stretch",
            output_format="JPEG",
        )
    else:
        frame_placeholder.info("Waiting for first valid frame...")

    lm = st.session_state.last_metrics
    m1.metric("FPS", f"{lm['fps']:.1f}")
    m2.metric("Vehicles/min", int(lm["vehicles_per_min"]))
    m3.metric("Pedestrians/min", int(lm["pedestrians_per_min"]))
    m4.metric("Line In/Out", f"{lm['line_in']}/{lm['line_out']}")
    m5.metric("Avg Speed km/h", f"{lm['avg_speed_kmh']:.1f}")

    hist_df = pd.DataFrame(st.session_state.history)
    if not hist_df.empty:
        hist_df = hist_df.set_index("ts")
        chart_placeholder.line_chart(hist_df[["vehicles_per_min", "pedestrians_per_min", "traffic_density", "avg_speed_kmh"]])
    else:
        chart_placeholder.info("No analytics data yet.")

    if st.session_state.events:
        events_placeholder.dataframe(pd.DataFrame(st.session_state.events[::-1]), width="stretch", height=220)
    else:
        events_placeholder.info("No crossing events yet.")

    if should_rerun:
        delay_ms = float(cfg["app"].get("refresh_ms", 140))
        if cfg["source"].get("chunk_mode", False) and cfg["app"].get("segment_playback_mode", False):
            delay_ms = max(delay_ms, float(cfg["source"].get("chunk_seconds", 30)) * 400.0)
        time.sleep(delay_ms / 1000.0)
        st.rerun()


if __name__ == "__main__":
    main()
