from __future__ import annotations

import time
from copy import deepcopy

import cv2
import pandas as pd
import streamlit as st

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
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {
            "fps": 0.0,
            "vehicles_per_min": 0.0,
            "pedestrians_per_min": 0.0,
            "avg_speed_kmh": 0.0,
            "line_in": 0,
            "line_out": 0,
        }


def build_config() -> dict:
    cfg = load_yaml("configs/default.yaml")
    cfg = deepcopy(cfg)

    st.sidebar.header("FlowTrack Controls")
    source = st.sidebar.text_input("Camera Source", value=str(cfg["source"].get("input", "0")))
    weights = st.sidebar.text_input("Weights", value=str(cfg["model"].get("weights", "yolov8n.pt")))
    device = st.sidebar.text_input("Device (cpu / cuda:0)", value=str(cfg["model"].get("device", "")))

    conf = st.sidebar.slider("Confidence", 0.1, 0.9, float(cfg["model"].get("conf", 0.35)), 0.01)
    iou = st.sidebar.slider("IoU", 0.1, 0.9, float(cfg["model"].get("iou", 0.45)), 0.01)
    imgsz = st.sidebar.selectbox("Image Size", [416, 512, 640, 800, 960, 1280], index=2)
    use_half = st.sidebar.checkbox("FP16 (Half)", value=bool(cfg["model"].get("half", False)))

    tracking_enabled = st.sidebar.checkbox("Enable Tracking (ByteTrack)", value=bool(cfg["tracking"].get("enabled", True)))
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=bool(cfg["app"].get("show_heatmap", False)))
    perf_mode = st.sidebar.checkbox("CPU Performance Mode (Recommended)", value=True)
    frame_skip = st.sidebar.slider("Inference Frame Skip", 0, 6, int(cfg["runtime"].get("frame_skip", 0)))
    rw = st.sidebar.selectbox("Render Width", [640, 800, 960, 1024, 1280], index=2)
    rh = st.sidebar.selectbox("Render Height", [360, 480, 540, 720], index=2)

    st.sidebar.subheader("Counting Line")
    x1 = st.sidebar.number_input("x1", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("x1", 100)))
    y1 = st.sidebar.number_input("y1", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("y1", 360)))
    x2 = st.sidebar.number_input("x2", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("x2", 1180)))
    y2 = st.sidebar.number_input("y2", min_value=0, max_value=4000, value=int(cfg["line_counter"].get("y2", 360)))

    cfg["source"]["input"] = str(source).strip()
    cfg["model"]["weights"] = weights
    cfg["model"]["conf"] = conf
    cfg["model"]["iou"] = iou
    cfg["model"]["imgsz"] = imgsz
    cfg["model"]["half"] = use_half
    cfg["model"]["device"] = device
    cfg["tracking"]["enabled"] = tracking_enabled
    cfg["app"]["show_heatmap"] = show_heatmap
    cfg["app"]["display"] = False
    cfg["runtime"]["frame_skip"] = int(frame_skip)
    cfg["runtime"]["resize_width"] = int(rw)
    cfg["runtime"]["resize_height"] = int(rh)
    cfg["line_counter"]["x1"] = int(x1)
    cfg["line_counter"]["y1"] = int(y1)
    cfg["line_counter"]["x2"] = int(x2)
    cfg["line_counter"]["y2"] = int(y2)

    if perf_mode:
        cfg["tracking"]["enabled"] = False
        cfg["app"]["show_heatmap"] = False
        cfg["model"]["imgsz"] = min(int(cfg["model"]["imgsz"]), 640)
        cfg["runtime"]["frame_skip"] = max(int(cfg["runtime"]["frame_skip"]), 2)
        cfg["runtime"]["resize_width"] = min(int(cfg["runtime"]["resize_width"]), 960)
        cfg["runtime"]["resize_height"] = min(int(cfg["runtime"]["resize_height"]), 540)
        if not str(cfg["model"].get("device", "")).strip():
            cfg["model"]["device"] = "cpu"

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

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Start", use_container_width=True):
        start_runner(cfg)
    if c2.button("Stop", use_container_width=True):
        stop_runner()
    c3.caption("Phase 1-5: detection, tracking, line counting, analytics, congestion/stop alerts, heatmap")
    if cfg["tracking"]["enabled"] is False:
        st.info("Performance profile active: detection-only mode for smoother stream on CPU.")

    frame_placeholder = st.empty()
    m1, m2, m3, m4, m5 = st.columns(5)

    chart_placeholder = st.empty()
    events_placeholder = st.empty()

    if st.session_state.running and st.session_state.runner is not None:
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
                "avg_speed_kmh": float(output.metrics.get("avg_speed_kmh", 0)),
                "line_in": int(line_in),
                "line_out": int(line_out),
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

        time.sleep(0.03)
        st.rerun()

    # Always render last available frame to avoid flicker/blank frames between reruns.
    if st.session_state.last_frame_rgb is not None:
        frame_placeholder.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
    else:
        frame_placeholder.info("No frame yet. Press Start and wait for stream initialization.")

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
        chart_placeholder.info("No analytics points yet.")

    if st.session_state.events:
        events_placeholder.dataframe(pd.DataFrame(st.session_state.events[::-1]), use_container_width=True, height=220)
    else:
        events_placeholder.info("No crossing events yet.")


if __name__ == "__main__":
    main()
