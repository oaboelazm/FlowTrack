# FlowTrack Model Training Guide

This guide covers training a traffic-focused YOLO model for FlowTrack.

## 1) Dataset choice
Recommended:
- BDD100K (strong urban traffic diversity)
- UA-DETRAC (vehicle-heavy scenes)
- Your own camera data (best domain match)
- VisDrone (strong detection + MOT benchmark, used in this repo smoke training)

Target classes in FlowTrack:
- person, bicycle, car, motorcycle, bus, truck

## 2) Prepare BDD100K into YOLO format
If you need an automated download helper (Kaggle mirror):
```bash
./scripts/training/download_bdd100k_kaggle.sh datasets/raw
```

Expected BDD folder layout:
- `<bdd_root>/images/100k/train`
- `<bdd_root>/images/100k/val`
- `<bdd_root>/labels/bdd100k_labels_images_train.json`
- `<bdd_root>/labels/bdd100k_labels_images_val.json`

Run conversion:
```bash
python scripts/training/prepare_bdd100k.py --bdd-root /path/to/bdd100k
```

This generates:
- `datasets/traffic_yolo/images/{train,val}`
- `datasets/traffic_yolo/labels/{train,val}`
- `configs/training/traffic_dataset.yaml`

## 3) Train
```bash
python scripts/training/train_yolo.py --train-config configs/training/train_bdd100k.yaml
```

Output checkpoints are under `runs/flowtrack/...`.
The best checkpoint is usually:
- `runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt`

## 4) Evaluate + export
```bash
python scripts/training/eval_export.py \
  --weights runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt \
  --data configs/training/traffic_dataset.yaml \
  --export-onnx
```

### Quick practical training profile already included
- `configs/training/train_visdrone_smoke.yaml` (1 epoch, fraction 1%)
- `configs/training/train_visdrone_smoke5.yaml` (5 epochs, fraction 1%)
- `configs/training/train_visdrone_full.yaml` (long run)

## 5) Use trained model in FlowTrack runtime
Optional: register best model into the project model path:
```bash
python scripts/training/register_best_model.py \
  --best runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt \
  --target models/flowtrack_best.pt
```

CLI:
```bash
python -m src.main --source 0 --weights runs/flowtrack/yolo_bdd100k_traffic/weights/best.pt
```

Streamlit:
- Launch UI with `streamlit run streamlit_app.py`
- Set `Weights` field to your `best.pt`

## Training tips
- Start with `yolov8n.pt` for speed, then test `yolov8s.pt` for accuracy.
- Increase `imgsz` to 1280 for far-object detection if GPU allows.
- Keep 10-20% validation data from your own cameras for realistic benchmarking.
- Track mAP50-95 and class-level AP; prioritize car/bus/truck AP for traffic analytics reliability.
