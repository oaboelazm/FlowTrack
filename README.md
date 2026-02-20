# FlowTrack

FlowTrack is an end-to-end real-time traffic monitoring and analytics system using YOLO + ByteTrack, with CLI runtime and Streamlit dashboard.

## Implemented Phases
- Phase 1: Real-time detection on webcam / RTSP / URL streams
- Phase 2: Multi-object tracking with ByteTrack IDs
- Phase 3: Virtual line crossing + directional counting (incoming/outgoing)
- Phase 4: Traffic analytics (flow rates, class distribution, density, speed estimate, CSV storage)
- Phase 5: Congestion alert, abnormal stopping alert, movement heatmap, edge-ready modular design

## Supported Classes
- `car`, `bus`, `truck`, `motorcycle`, `bicycle`, `person`

## Project Structure
- `src/ingestion`: stream capture + reconnect
- `src/detection`: YOLO detection module
- `src/tracking`: ByteTrack tracking module
- `src/events`: line crossing engine
- `src/analytics`: flow/density/speed/congestion/stop analytics + heatmap
- `src/storage`: CSV persistence
- `src/visualization`: overlays for IDs, line, and stats panel
- `src/app/pipeline.py`: central real-time engine
- `src/main.py`: CLI entrypoint
- `streamlit_app.py`: web interface

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (CLI)
Webcam:
```bash
python -m src.main --source 0
```

RTSP/HLS:
```bash
python -m src.main --source "rtsp://user:pass@host:554/stream"
```

With heatmap:
```bash
python -m src.main --source 0 --show-heatmap
```

## Run (Streamlit UI)
```bash
streamlit run streamlit_app.py
```

Or:
```bash
./scripts/run_streamlit.sh
```

## Outputs
- `outputs/metrics.csv`: periodic metrics snapshots
- `outputs/crossings.csv`: line crossing events with direction

## Notes
- Default config: `configs/default.yaml`
- By default tracking is enabled using `bytetrack.yaml`
- First run may download YOLO weights automatically
- For better accuracy in production, fine-tune on BDD100K / UA-DETRAC / custom city data
