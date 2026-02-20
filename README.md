# FlowTrack

FlowTrack is a real-time traffic monitoring and analytics system built with YOLO + ByteTrack + OpenCV, with a Streamlit dashboard for live monitoring.

## Current Status
- End-to-end pipeline implemented for Phases 1 to 5.
- Streamlit app launch verified on **February 20, 2026**.
- Trained model artifact is available in this repo:
  - `models/flowtrack_best.pt`
  - `models/flowtrack_best.onnx`

## Features
- Real-time multi-class detection (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`)
- Multi-object tracking with unique IDs (ByteTrack)
- Directional line crossing counts (`incoming` / `outgoing`)
- Traffic analytics (flow, density, distribution, speed estimate)
- Congestion and abnormal stop indicators
- Movement heatmap overlay
- CSV persistence for metrics and crossing events
- Live Streamlit dashboard

## Project Structure
- `src/ingestion` stream readers and reconnect handling
- `src/detection` YOLO detection module
- `src/tracking` ByteTrack module
- `src/events` line crossing logic
- `src/analytics` traffic analytics and heatmaps
- `src/storage` CSV persistence
- `src/visualization` overlays and HUD rendering
- `src/app/pipeline.py` unified runtime engine
- `src/main.py` CLI runtime entrypoint
- `streamlit_app.py` web dashboard
- `configs/default.yaml` runtime config
- `configs/training/*` training configs
- `docs/` technical documentation and model report

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
CLI (webcam):
```bash
python -m src.main --source 0
```

CLI (RTSP):
```bash
python -m src.main --source "rtsp://user:pass@host:554/stream"
```

Streamlit:
```bash
streamlit run streamlit_app.py
```

## Training
Quick training scripts:
```bash
./scripts/train_fit.sh
./scripts/train_eval_export.sh runs/detect/runs/flowtrack/visdrone_smoke5/weights/best.pt
```

BDD100K conversion/training workflow:
- `docs/TRAINING.md`
- `scripts/training/prepare_bdd100k.py`
- `configs/training/train_bdd100k.yaml`

## Outputs
- Runtime analytics: `outputs/metrics.csv`, `outputs/crossings.csv`
- Training artifacts: `runs/detect/runs/flowtrack/*`
- Documentation figures: `docs/assets/*`

## Documentation
- Full system documentation: `docs/PROJECT_DOCUMENTATION.md`
- Model training report with metrics and figures: `docs/MODEL_REPORT.md`
- Training guide: `docs/TRAINING.md`
