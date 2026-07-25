# 🧵 AI Fabric Inspector: Real-Time Defect Detection & Quality Grading

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)

An end-to-end computer vision platform designed for real-time fabric defect detection, classification, and industrial quality grading using modified **YOLOv8n + CBAM (Convolutional Block Attention Module)** architecture with **Soft Retinex** image enhancement.

---

## ✨ Key Features

- 🔍 **CBAM Attention Integration:** Detects low-contrast, fine-grained fabric flaws (slubs, broken stitches, oil stains).
- ☀️ **Soft Retinex Enhancement:** Preprocessing pipeline that neutralizes uneven illumination and shadow noise.
- 🎛️ **Live AI Calibration:** Real-time adjustable confidence and NMS IoU threshold sliders with defect sensitivity tips.
- 📊 **Multi-View Inspection Dashboard:**
  - **Visual Grid:** Simultaneous view of all uploaded fabric samples with neural bounding box overlays.
  - **Single Sample Inspection:** Detailed side-by-side inspection with downloadable high-res outputs.
  - **Quality Audit Trail:** Auto-generates Grade A/B/C defect logs and downloadable CSV reports.
  - **Research Benchmarks:** Interactive comparative charts evaluating Recall, mAP@50, and Latency.

---

## 📁 Project Directory Structure

```text
AI-Fabric-Inspector/
├── app.py                      # Main Streamlit Dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── yolov8n.pt                  # Default YOLOv8 pretrained weights
└── weight/                     # Fine-tuned model checkpoints
    ├── yolov8n_cbam_softretinex.pt   # Proposed model (CBAM + Soft Retinex)
    ├── yolov8n_baseline.pt           # Baseline YOLOv8 model
    └── yolov11n_baseline.pt          # Baseline YOLO11 model
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/AI-Fabric-Inspector.git](https://github.com/YOUR_USERNAME/AI-Fabric-Inspector.git)
cd AI-Fabric-Inspector
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📊 Experimental Performance Summary

| Model Architecture | Recall (Sensitivity) | mAP@50 | Latency (FPS) | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8n + CBAM (Proposed - Soft Retinex)** | **0.4233** | **0.4461** | **~115 FPS** | **3.15 M** |
| YOLOv8n (Baseline) | 0.3693 | 0.4666 | ~140 FPS | 3.01 M |
| YOLO11n (Baseline) | 0.3172 | 0.4097 | ~135 FPS | 2.58 M |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.