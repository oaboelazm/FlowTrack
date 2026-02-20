# FlowTrack Project Documentation

## 1) Objective
Build a production-oriented real-time traffic intelligence pipeline that can process live camera streams and provide:
- object detection
- multi-object tracking
- directional counting
- traffic analytics
- dashboard visualization

## 2) Implemented Scope
### Phase 1: Real-Time Detection
- YOLO inference on webcam/RTSP stream
- target classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`
- class-wise and total counts per frame

### Phase 2: Tracking
- ByteTrack integration
- per-object stable IDs
- reduced double counting via temporal tracking

### Phase 3: Line Crossing
- configurable virtual line (`x1,y1,x2,y2`)
- directional events (`incoming`, `outgoing`)
- cooldown-based anti-duplicate logic

### Phase 4: Analytics
- vehicles/minute, vehicles/hour
- pedestrian flow/minute
- class distribution in frame
- traffic density proxy
- speed estimate (pixel displacement + calibration factor)
- persistent storage to CSV

### Phase 5: Advanced Indicators
- congestion indicator (threshold + hold-frames)
- abnormal stop indicator
- movement heatmap overlay
- ONNX export support for edge deployment

## 3) Architecture
### Pipeline flow
`Source -> Detect/Track -> Events -> Analytics -> Storage -> Visualization`

### Main modules
- `src/ingestion/source_manager.py`
- `src/detection/yolo_detector.py`
- `src/tracking/bytetrack_tracker.py`
- `src/events/line_counter.py`
- `src/analytics/traffic_analytics.py`
- `src/storage/csv_writer.py`
- `src/app/pipeline.py`
- `streamlit_app.py`

## 4) Runtime Modes
### CLI mode
- command-line execution with optional display and heatmap overlays

### Streamlit mode
- live frame rendering
- real-time KPI cards
- metric trend charts
- latest crossing events table

## 5) Configuration
Primary runtime configuration:
- `configs/default.yaml`

Current default model:
- `models/flowtrack_best.pt`

Training configurations:
- `configs/training/train_visdrone_smoke.yaml`
- `configs/training/train_visdrone_smoke5.yaml`
- `configs/training/train_visdrone_full.yaml`
- `configs/training/train_bdd100k.yaml`

## 6) Storage and Artifacts
### Runtime outputs
- `outputs/metrics.csv`
- `outputs/crossings.csv`

### Training outputs
- `runs/detect/runs/flowtrack/*`
- selected exported artifacts in `models/`

## 7) Known Constraints
- Current trained checkpoint is a **smoke model** (quick CPU training profile), not final production accuracy.
- Higher-quality deployment needs full training on larger subsets/full datasets (BDD100K/VisDrone/custom city cameras).
- Speed estimates require scene-specific meter/pixel calibration for physical accuracy.

## 8) Recommended Production Next Steps
1. Train longer profile (`train_visdrone_full.yaml`) or full BDD100K pipeline.
2. Add periodic evaluation on held-out city-specific validation data.
3. Calibrate speed/line geometry per camera.
4. Add DB backend (PostgreSQL/TimescaleDB) for long-term analytics.
5. Package as Docker services (inference + dashboard + storage).
