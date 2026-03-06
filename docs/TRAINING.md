# FlowTrack Model Training Guide

This guide details the process for training a custom YOLO model (for detection or segmentation) specifically tailored for the **FlowTrack** traffic analytics pipeline.

## 1) Dataset Choice

For robust traffic monitoring, we recommend datasets that capture diverse urban environments, lighting conditions, and camera angles:

*   **BDD100K**: Excellent for urban traffic diversity, varying weather, and different times of day.
*   **UA-DETRAC**: A large dataset focused specifically on vehicle detection and tracking from traffic cameras.
*   **VisDrone**: Useful for overhead or drone-captured traffic footage. This repository includes "smoke" training profiles based on VisDrone for quick testing.
*   **Custom Data**: The most critical dataset for deployment. Fine-tuning on footage from the *actual cameras* where FlowTrack will be deployed yields the best domain match.

**Target Classes in FlowTrack:**
The primary classes of interest for the pipeline are:
`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`

---

## 2) Prepare BDD100K into YOLO Format

If you plan to use BDD100K, you must convert it from its native JSON format to the YOLO txt format.

**A) Download (Kaggle Mirror Helper):**
```bash
./scripts/training/download_bdd100k_kaggle.sh datasets/raw
```

**B) Verify Directory Structure:**
Ensure your downloaded BDD100K data is organized as follows:
*   `<bdd_root>/images/100k/train`
*   `<bdd_root>/images/100k/val`
*   `<bdd_root>/labels/bdd100k_labels_images_train.json`
*   `<bdd_root>/labels/bdd100k_labels_images_val.json`

**C) Run Conversion Script:**
```bash
python scripts/training/prepare_bdd100k.py --bdd-root /path/to/bdd100k
```

This script will generate:
1.  Images: `datasets/traffic_yolo/images/{train,val}`
2.  Labels: `datasets/traffic_yolo/labels/{train,val}` (YOLO format txt files)
3.  Config: `configs/training/traffic_dataset.yaml` (The data configuration file needed for YOLO training)

---

## 3) Train the Model

Use the provided training script to start the training process. The `--train-config` argument points to a YAML file containing hyperparameters and dataset paths.

```bash
python scripts/training/train_yolo.py --train-config configs/training/train_bdd100k.yaml
```

**Output Checkpoints:**
Training outputs, including checkpoints, logs, and evaluation plots, are saved under `runs/flowtrack/...`.
The best-performing model weights during the training run are usually located at:
*   `runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt`

---

## 4) Evaluate and Export

After training, evaluate the best checkpoint on your validation set to get precise metrics (mAP, Precision, Recall). You can also export the model to ONNX format for potentially faster inference or edge deployment.

```bash
python scripts/training/eval_export.py \
  --weights runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt \
  --data configs/training/traffic_dataset.yaml \
  --export-onnx
```

### Provided Training Profiles (Quick Starts)
The `configs/training/` directory contains several pre-configured YAML files:
*   `train_visdrone_smoke.yaml`: A very short run (1 epoch, 1% data) to quickly verify the training pipeline works without errors.
*   `train_visdrone_smoke5.yaml`: A slightly longer smoke test (5 epochs, 1% data).
*   `train_visdrone_full.yaml`: A configuration intended for a full, comprehensive training run on the VisDrone dataset.
*   `train_bdd100k.yaml`: The configuration for training on the converted BDD100K dataset.

---

## 5) Use the Trained Model in FlowTrack

Once you have a satisfactory `best.pt` file, you need to tell FlowTrack to use it.

**Option A: Register the model (Recommended)**
This script copies your best weights to a standard location (e.g., `models/flowtrack_best.pt`), making it easier to reference.
```bash
python scripts/training/register_best_model.py \
  --best runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt \
  --target models/flowtrack_best.pt
```
Then, update your `configs/default.yaml` to point `model.weights` to `models/flowtrack_best.pt`.

**Option B: Use via CLI argument**
You can temporarily override the default weights when running the CLI:
```bash
python -m src.main --source 0 --weights runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt
```

**Option C: Streamlit UI**
1.  Launch the dashboard: `streamlit run streamlit_app.py`
2.  In the UI settings sidebar, set the `Weights` field to the path of your new `best.pt` file.

---

## 💡 Training Tips

*   **Model Selection:** Start with `yolov8n.pt` (Nano) or `yolo11n.pt` for rapid iteration and prototyping. If accuracy is insufficient and your hardware permits, scale up to `yolov8s.pt` (Small) or `yolov8m.pt` (Medium).
*   **Image Size (`imgsz`):** For traffic monitoring, detecting small, distant vehicles is often crucial. Increase `imgsz` (e.g., from 640 to 960 or 1280) if your GPU memory allows it.
*   **Validation Data:** Always keep 10-20% of the data from *your specific deployment cameras* as a validation set. This provides the most realistic benchmark of how the model will perform in production.
*   **Key Metrics:** While overall mAP is important, pay close attention to class-level Average Precision (AP). Prioritize high AP for `car`, `bus`, `truck`, and `person` to ensure reliable traffic analytics.
