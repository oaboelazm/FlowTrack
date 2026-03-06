# FlowTrack Project Documentation

## 1) Objective

Build a production-oriented real-time traffic intelligence pipeline that can process live camera streams and provide:
- Object detection
- Multi-object tracking
- Directional counting
- Traffic analytics
- Dashboard visualization

## 2) Implemented Scope

### Phase 1: Real-Time Detection
- **YOLO** inference on webcam/RTSP stream.
- Target classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
- Outputs class-wise and total counts per frame.

### Phase 2: Tracking
- **ByteTrack** integration.
- Assigns per-object stable unique IDs.
- Significantly reduces double counting through robust temporal tracking.

### Phase 3: Line Crossing
- Configurable virtual line defined by coordinates (`x1, y1, x2, y2`).
- Detects directional events (`incoming`, `outgoing`).
- Incorporates a cooldown-based anti-duplicate logic to ensure accurate counts.

### Phase 4: Analytics
- Calculates **Flow**: vehicles/minute, vehicles/hour.
- Calculates **Pedestrian Flow**: pedestrian flow/minute.
- Analyzes class distribution within the frame.
- Provides a **Traffic Density Proxy** based on object bounding box areas.
- **Speed Estimate**: Calculated using pixel displacement over time and a calibration factor (`meters_per_pixel`).
- Persistent storage of all calculated metrics and events to **CSV files**.

### Phase 5: Advanced Indicators
- **Congestion Indicator**: Triggered based on a vehicle count threshold and sustained over a hold-frames period.
- **Abnormal Stop Indicator**: Detects vehicles that stop for unusually long durations in unexpected areas.
- **Movement Heatmap Overlay**: Visualizes areas of high activity and traffic flow patterns.
- **ONNX Export Support**: Allows exporting the model for optimized edge deployment.

### Phase 6: Instance Segmentation
- Integrated optional **YOLO Segmentation** models.
- Extracts precise object masks for enhanced visualization and potentially more accurate analytics in the future.

## 3) Architecture

### Pipeline Flow
`Source -> Detect/Track/Segment -> Events -> Analytics -> Storage -> Visualization`

### Main Modules
- **`src/ingestion/source_manager.py`**: Handles stream reading, reconnection logic, and video chunking.
- **`src/detection/yolo_detector.py`**: Wraps the YOLO object detection model.
- **`src/segmentation/yolo_segmentor.py`**: Wraps the YOLO instance segmentation model.
- **`src/tracking/bytetrack_tracker.py`**: Manages the ByteTrack object tracker.
- **`src/events/line_counter.py`**: Contains the logic for detecting line crossing events.
- **`src/analytics/traffic_analytics.py`**: Computes density, speed, congestion, and generates heatmaps.
- **`src/storage/csv_writer.py`**: Handles persistent logging of data to CSV.
- **`src/app/pipeline.py`**: The central runtime engine that orchestrates all the modules.
- **`streamlit_app.py`**: The Streamlit-based web dashboard.
- **`gradio_app.py`**: The Gradio-based web dashboard.

## 4) Runtime Modes

### CLI Mode (`src/main.py`)
- Command-line execution.
- Optional display windows (OpenCV) and heatmap overlays.
- Good for headless processing or quick local testing.

### Streamlit Mode (`streamlit_app.py`)
- Live frame rendering in the browser.
- Real-time KPI cards for counts, speeds, and density.
- Metric trend charts.
- Latest crossing events table.

### Gradio Mode (`gradio_app.py`)
- GPU-friendly runtime loop (ideal for Google Colab/Kaggle).
- Incorporates a smooth chunked video playback system to display processed segments seamlessly.

## 5) Configuration

The primary runtime configuration is handled via **`configs/default.yaml`**.

Key sections include:
- `app`: Window settings, display toggles, segment playback mode.
- `source`: Input URL (webcam or RTSP), chunking parameters (`chunk_mode`, `chunk_seconds`).
- `model`: YOLO weights path, confidence, IoU, image size.
- `tracking`: Toggle tracking and select tracker config (`bytetrack.yaml`).
- `line_counter`: Set line coordinates and cooldown.
- `analytics`: Thresholds for congestion, abnormal stops, and pixel-to-meter calibration.

Current default model:
- `PretrainedYolo26/Detect.pt` (Detection)
- `PretrainedYolo26/Segment.pt` (Segmentation)

Training configurations (for custom model training):
- `configs/training/train_visdrone_smoke.yaml`
- `configs/training/train_visdrone_smoke5.yaml`
- `configs/training/train_visdrone_full.yaml`
- `configs/training/train_bdd100k.yaml`

## 6) Storage and Artifacts

### Runtime Outputs
- `outputs/metrics.csv`: Time-series data of traffic analytics.
- `outputs/crossings.csv`: Log of every line crossing event.
- `outputs/chunks/` & `outputs/processed_chunks/`: Temporary and processed video segments when chunk mode is enabled.

### Training Outputs
- `runs/detect/runs/flowtrack/*`: Contains model weights, evaluation metrics, and plots from training runs.
- Selected exported artifacts are typically moved to `models/` or `PretrainedYolo26/`.

## 7) Known Constraints

- The default trained models might require fine-tuning for specific camera angles and lighting conditions to achieve maximum accuracy.
- **Speed estimates** are currently based on a global `meters_per_pixel` calibration factor, which assumes a flat plane parallel to the camera. For accurate physical speed measurements, scene-specific perspective calibration is required.
- Running both detection and segmentation simultaneously on high-resolution streams requires significant computational power (GPU recommended).

## 8) Recommended Production Next Steps

1. **Train longer profiles** (e.g., `train_visdrone_full.yaml`) or a full BDD100K pipeline for improved model robustness.
2. Add **periodic evaluation** on held-out, city-specific validation data to ensure continued performance.
3. Implement a tool for easy **perspective calibration** to improve the accuracy of speed and distance estimations for different camera angles.
4. Integrate a **Database Backend** (e.g., PostgreSQL + TimescaleDB) to replace CSV storage for scalable, long-term analytics and querying.
5. Package the entire system into **Docker services** (Inference Node, Dashboard Node, Database Node) for easy deployment and scaling.
