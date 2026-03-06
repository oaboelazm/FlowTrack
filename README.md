# FlowTrack

FlowTrack is a real-time traffic analytics system for **object detection + instance segmentation + tracking** on live video streams.

It provides:
- Real-time detection (vehicles and people)
- Optional instance segmentation overlay
- Multi-object tracking (ByteTrack)
- Line-crossing counts (incoming / outgoing)
- Traffic analytics (flow, density, speed proxy)
- Live dashboards with Streamlit and Gradio

## One-Click Notebooks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oaboelazm/FlowTrack/blob/main/notebooks/FlowTrack_Colab_Training_and_Stream.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/oaboelazm/FlowTrack/main/notebooks/FlowTrack_Kaggle_Training_and_Stream.ipynb)

## Model and Training Summary

- Detection dataset: **VisDrone**
- Segmentation dataset: **COCO**
- Base model used for both tasks: **YOLO26m**
- Training setup: **15 epochs** (same setup used in the project notebook)
- Runtime weights in this repo:
  - `PretrainedYolo26/Detect.pt`
  - `PretrainedYolo26/Segment.pt`

## Key Validation Results

### Segmentation (COCO-trained YOLO26m)

Overall:
- Box: `P=0.717`, `R=0.598`, `mAP50=0.664`, `mAP50-95=0.495`
- Mask: `P=0.716`, `R=0.576`, `mAP50=0.639`, `mAP50-95=0.408`

Important classes (Mask mAP50-95):
- `person`: `0.501`
- `car`: `0.400`
- `bus`: `0.675`
- `truck`: `0.379`

### Detection (VisDrone-trained YOLO26m)

Overall:
- Box: `P=0.385`, `R=0.287`, `mAP50=0.275`, `mAP50-95=0.160`

Important classes (mAP50-95):
- `car`: `0.493`
- `pedestrian`: `0.214`
- `bus`: `0.244`
- `truck`: `0.093`

## Main Features

- Real-time video input from webcam, RTSP, file, or URL
- Detection and optional segmentation in one pipeline
- ByteTrack integration for stable track IDs
- Configurable virtual line for directional counting
- CSV logging for metrics and crossing events
- Chunked streaming and smooth segment playback for web apps

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Project

### 1) CLI

```bash
python -m src.main --source 0
```

RTSP example:

```bash
python -m src.main --source "rtsp://user:pass@host:554/stream"
```

### 2) Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

### 3) Gradio Dashboard

```bash
python gradio_app.py
```

## Configuration

Main runtime config: `configs/default.yaml`

Important defaults:
- Detection weights: `PretrainedYolo26/Detect.pt`
- Segmentation weights: `PretrainedYolo26/Segment.pt`
- Chunk mode: enabled
- Segment playback mode: enabled

You can enable/disable detection, tracking, and segmentation from the UI or config.

## Training Files

- Notebook workflows:
  - `notebooks/FlowTrack_Colab_Training_and_Stream.ipynb`
  - `notebooks/FlowTrack_Kaggle_Training_and_Stream.ipynb`
- Script entrypoint: `scripts/training/train_yolo.py`
- Training configs: `configs/training/`

## Output Files

- Runtime metrics: `outputs/metrics.csv`
- Crossing events: `outputs/crossings.csv`
- Processed stream chunks: `outputs/processed_chunks/`

## Project Structure

- `src/ingestion`: video source and chunked ingestion
- `src/detection`: YOLO detection
- `src/segmentation`: YOLO segmentation
- `src/tracking`: ByteTrack tracking
- `src/events`: line crossing logic
- `src/analytics`: traffic analytics
- `src/storage`: CSV writing
- `src/visualization`: overlays and rendering
- `src/app/pipeline.py`: core runtime pipeline

