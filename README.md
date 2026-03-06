# FlowTrack 🚗🚦

<div align="center">
  <p><strong>Real-Time Traffic Monitoring and Analytics System</strong></p>
</div>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oaboelazm/FlowTrack/blob/main/notebooks/FlowTrack_Colab_Training_and_Stream.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/oaboelazm/FlowTrack/main/notebooks/FlowTrack_Kaggle_Training_and_Stream.ipynb)

---

## 🌟 Overview

**FlowTrack** is an end-to-end, real-time traffic intelligence pipeline built using **YOLO**, **ByteTrack**, and **OpenCV**. It provides robust object detection, multi-object tracking, directional line crossing counts, and advanced traffic analytics (speed estimation, congestion detection, etc.). The project includes live monitoring dashboards built with **Streamlit** and **Gradio**.

---

## ✨ Features

*   **Real-time Multi-Class Detection:** Detects `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.
*   **Multi-Object Tracking:** Assigns unique, stable IDs to objects using ByteTrack.
*   **Directional Line Crossing:** Tracks objects crossing virtual lines (`incoming` / `outgoing`) with anti-duplicate logic.
*   **Traffic Analytics:** Calculates flow (vehicles/min), density, class distribution, and estimates speed.
*   **Advanced Indicators:** Congestion detection, abnormal stop detection, and movement heatmap overlay.
*   **Segmentation Support:** Optional instance segmentation using YOLO segmentation models.
*   **Dashboards:** Live web dashboards using Streamlit and a GPU-friendly Gradio app.
*   **Smooth Playback Mode:** Chunked stream buffer mode for stable, video-like playback without frame-by-frame flickering.
*   **Data Persistence:** Exports metrics and crossing events to CSV files for further analysis.

---

## 📂 Project Structure

```text
FlowTrack/
├── configs/                  # YAML configuration files (default.yaml, training configs)
├── docs/                     # Technical documentation, model reports, and training guides
├── models/                   # Directory for trained models (.pt, .onnx)
├── notebooks/                # Colab & Kaggle quickstart notebooks
├── PretrainedYolo26/         # Pretrained YOLO models (Detect.pt, Segment.pt)
├── scripts/                  # Bash and Python scripts for training, evaluation, etc.
├── src/                      # Source code
│   ├── analytics/            # Traffic analytics, speed estimation, heatmaps
│   ├── app/                  # Unified runtime engine (pipeline.py)
│   ├── core/                 # Core entities, class names normalization
│   ├── detection/            # YOLO detection wrapper
│   ├── events/               # Line crossing logic
│   ├── ingestion/            # Stream readers, reconnect handlers, chunking
│   ├── segmentation/         # YOLO segmentation wrapper
│   ├── storage/              # CSV persistence
│   ├── tracking/             # ByteTrack logic
│   ├── utils/                # Logging, config loading, generic utilities
│   └── visualization/        # Overlays, HUD rendering, drawing bounding boxes
├── main.py                   # CLI entrypoint (src.main)
├── gradio_app.py             # Gradio web dashboard
├── streamlit_app.py          # Streamlit web dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/oaboelazm/FlowTrack.git
cd FlowTrack
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optional) If you want video chunk conversion to H.264 MP4 for browser compatibility, ensure `ffmpeg` is installed on your system:*
* **Ubuntu/Debian:** `sudo apt install ffmpeg`
* **Mac:** `brew install ffmpeg`

---

## 💻 Usage

### CLI Mode
Run with a webcam (default `0`):
```bash
python -m src.main --source 0
```
Run with an RTSP stream:
```bash
python -m src.main --source "rtsp://user:pass@host:554/stream"
```
*Note: You can pass overrides such as `--weights`, `--conf`, `--show-heatmap`, etc. Check `python -m src.main --help`.*

### Streamlit Dashboard
Launch the interactive Streamlit dashboard for real-time monitoring:
```bash
streamlit run streamlit_app.py
```

### Gradio Dashboard (Recommended for Colab / GPU)
Launch the Gradio app, which includes an optimized runtime loop:
```bash
python gradio_app.py
```

---

## ⚙️ Configuration
The main configuration file is located at `configs/default.yaml`.
You can adjust:
* **`app`**: Display settings, playback mode.
* **`source`**: Input URL, chunking settings, reconnect delays.
* **`model`**: YOLO weights path, confidence threshold (`conf`), NMS IoU (`iou`), image size (`imgsz`).
* **`line_counter`**: Coordinates for the virtual counting line (`x1, y1, x2, y2`) and cooldown logic.
* **`analytics`**: Thresholds for congestion, speed estimation calibration (`meters_per_pixel`), etc.
* **`storage`**: Paths for output CSV files.

---

## ☁️ Cloud Quick Deploy

### Google Colab (GPU)
```bash
!git clone https://github.com/oaboelazm/FlowTrack.git
%cd FlowTrack
!pip install -r requirements.txt
!GRADIO_SHARE=1 python gradio_app.py
```

### Kaggle (GPU)
```bash
!git clone https://github.com/oaboelazm/FlowTrack.git
%cd FlowTrack
!pip install -r requirements.txt
!ffmpeg -version || (apt-get update && apt-get install -y ffmpeg)
!GRADIO_SHARE=1 python gradio_app.py
```

---

## 📚 Documentation
For a deeper dive into the system architecture, training pipelines, and evaluation metrics, please refer to the `docs/` folder:

*   📖 [**Project Documentation**](docs/PROJECT_DOCUMENTATION.md): Detailed explanation of modules, architecture, and runtime modes.
*   🏋️ [**Training Guide**](docs/TRAINING.md): How to train FlowTrack on custom datasets like BDD100K or VisDrone.
*   📊 [**Model Report**](docs/MODEL_REPORT.md): Evaluation metrics and benchmarks for the provided models.

---

## 📝 Outputs & Artifacts
*   **Runtime Analytics:** Stored in `outputs/metrics.csv` and `outputs/crossings.csv`.
*   **Training Artifacts:** Stored in `runs/detect/runs/flowtrack/*`.
*   **Chunked Videos:** Stored in `outputs/chunks/` and `outputs/processed_chunks/`.
