# FlowTrack Model Report

This document summarizes the training parameters, performance metrics, and visual artifacts of the currently registered default model (`models/flowtrack_best.pt`).

---

## 1) Training Run Summary

This section details the specific configuration used to train the model.

*   **Run Name:** `visdrone_smoke5`
*   **Framework:** Ultralytics YOLOv8
*   **Base Model (Initialization):** `yolov8n.pt` (Nano variant)
*   **Dataset Configuration:** `VisDrone.yaml` (VisDrone dataset)
*   **Training Profile:**
    *   **Epochs:** 5 (A very short run for verification)
    *   **Data Fraction:** 1% (Using a tiny subset of the full dataset)
    *   **Image Size (`imgsz`):** 640
    *   **Batch Size:** 8
    *   **Hardware:** CPU
*   **Selected Weights Checkpoint:** `runs/detect/runs/flowtrack/visdrone_smoke5/weights/best.pt`
*   **Registered Project Model Path:** `models/flowtrack_best.pt`
*   **ONNX Export Path:** `models/flowtrack_best.onnx`

---

## 2) Final Validation Metrics

The following metrics represent the performance of the best checkpoint on the validation set at the end of the training run.

**Source File:** `runs/detect/runs/flowtrack/visdrone_smoke5/results.csv` (Epoch 5)

*   **Precision (B):** 0.01271
*   **Recall (B):** 0.03768
*   **mAP50 (B):** 0.01122
*   **mAP50-95 (B):** 0.00527

---

## 3) Metric Progress by Epoch

This table shows how the key metrics evolved over the 5 training epochs.

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.00401 | 0.02935 | 0.00374 | 0.00235 |
| 2 | 0.00813 | 0.04138 | 0.00587 | 0.00319 |
| 3 | 0.01069 | 0.04156 | 0.00811 | 0.00414 |
| 4 | 0.01170 | 0.03545 | 0.00993 | 0.00469 |
| 5 | **0.01271** | **0.03768** | **0.01122** | **0.00527** |

---

## 4) Visual Artifacts

The following plots and images provide visual insight into the model's performance during training and on the validation set.

### A) Training Curves
*Graphs showing the evolution of loss and mAP over epochs.*
![Training Results](assets/train_results.png)

### B) Precision-Recall Curve
*Illustrates the tradeoff between precision and recall at different confidence thresholds.*
![PR Curve](assets/pr_curve.png)

### C) Confusion Matrix
*A detailed breakdown of true vs. predicted classifications for each class.*
![Confusion Matrix](assets/confusion_matrix.png)

### D) Validation Predictions
*Sample predictions from the validation set, visually demonstrating the model's current capability.*
![Validation Prediction 0](assets/val_pred_0.jpg)
![Validation Prediction 1](assets/val_pred_1.jpg)

### E) Sample Training Batch
*Visualizing the augmented data and bounding boxes the model learned from during training.*
![Training Batch](assets/train_batch_0.jpg)

---

## 5) Interpretation & Current Status

*   **Functional Baseline:** The current `best.pt` model serves as a valid, end-to-end functional checkpoint for testing the complete FlowTrack pipeline (detection, tracking, analytics, and visualization).
*   **Low Metrics Justification:** The precision and recall metrics reported above are **intentionally low**. This training run (`visdrone_smoke5`) was designed as a "smoke test"—a quick, lightweight process (CPU, 5 epochs, 1% data) simply to verify the codebase and training scripts run without errors.
*   **Not for Production:** This specific checkpoint should *not* be used in a production environment or for accurate real-world traffic analytics.

---

## 6) 🚀 Recommended Accuracy Upgrade Path

To achieve deployment-grade accuracy and robust performance, follow these steps to train a significantly better model:

1.  **Full Dataset Training:**
    *   Transition from the "smoke test" profile to a comprehensive training run. Use the provided `configs/training/train_visdrone_full.yaml` to train on the complete VisDrone dataset for an appropriate number of epochs (e.g., 50-100+).
2.  **Domain-Specific Datasets:**
    *   For urban traffic scenarios, the VisDrone dataset (often captured from drones) might not be ideal. We strongly recommend fine-tuning or training from scratch on the **BDD100K** dataset (`configs/training/train_bdd100k.yaml`), which features diverse, ground-level dashcam footage.
3.  **Incorporate Custom Camera Data:**
    *   The most significant improvement will come from adding specific examples (including hard-negative scenes like strong glare, heavy rain, or unusual angles) captured directly from the actual cameras where FlowTrack will be deployed.
4.  **Prioritize Class AP:**
    *   During evaluation, don't just look at the overall mAP. Ensure the Average Precision (AP) for key traffic classes (`car`, `truck`, `bus`, `person`) meets the specific requirements of your analytics use cases before releasing the model to production.
