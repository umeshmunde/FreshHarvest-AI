---
title: FreshHarvest AI
emoji: 🍎
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FreshHarvest AI – Multi-Model Fruit Freshness and Quality Assessment System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An advanced, end-to-end computer vision and deep learning system for automated fruit classification, freshness determination, explainability heatmapping (Grad-CAM), and agricultural quality control.

---

## 📌 Project Overview

**FreshHarvest AI** is an improved, production-ready final-year engineering system built upon the foundational research of [erickyegon/FreshHarvest](https://github.com/erickyegon/FreshHarvest). It expands upon the original binary classification concept into a comprehensive **Multi-Model, Explainable AI (XAI) Platform** designed for agricultural supply chains, food processing facilities, and retail quality inspection.

### Key Capabilities

1. **Dual-Task AI Inference**: Simultaneously identifies **Fruit Variety** (Apple, Banana, Orange, Mango, Strawberry, Tomato, Lemon) and classifies **Freshness State** (Commercial Fresh vs. Spoiled/Rotten).
2. **Tri-Model Comparative Ensemble**: Leverages three distinct CNN architectures:
   - **EfficientNet-B0** (Compound scaling for optimal parameter efficiency)
   - **ResNet-50** (Deep residual connections preventing vanishing gradients)
   - **MobileNet-V2** (Inverted residuals with linear bottlenecks for high-speed edge inference)
3. **Multi-Model Consensus & Agreement**: Runs predictions across all three backbones simultaneously, comparing confidence scores, latencies, and calculating majority agreement (e.g., `3/3 Agree (Fresh)`).
4. **Explainable AI (Grad-CAM)**: Generates gradient-weighted class activation heatmaps overlaid onto the original fruit imagery, highlighting surface spots, fungal decay, and textural anomalies that guided the model's decision.
5. **High-Throughput Batch Screening**: Upload multiple fruit images or a `.zip` file for automated batch inspection, freshness ratio analytics, and instant CSV export.
6. **Persistent SQLite Audit History**: Complete database history storing scan dates, filenames, predictions, confidence scores, latencies, and Grad-CAM paths with search, filtering, and CSV download.
7. **Interactive Admin Analytics Dashboard**: Live KPI tiles, Chart.js visualizations (Fresh vs Spoiled donut, fruit variety breakdown, multi-model radar/bar benchmark), and interactive confusion matrices.
8. **1-Click Test Drive**: Embedded interactive samples for instant out-of-the-box evaluation without searching for external image files.

---

## 🏗️ Technical Architecture

```
FreshHarvest AI
├── app.py                     # Flask web server, route controllers, and REST APIs
├── config.py                  # Global configurations, hyperparameter registry, and paths
├── database.py                # SQLite ORM, schema migration, audit history, and analytics queries
├── data.py                    # Multi-format dataset loader (F_*/S_* and Kaggle fresh*/rotten*)
├── models.py                  # PyTorch model factory (EfficientNet-B0, ResNet-50, MobileNet-V2)
├── gradcam.py                 # Grad-CAM explainability engine with OpenCV jet colormaps
├── inference.py               # Inference pipelines (Single, Multi-Model Consensus, Batch)
├── train.py                   # Transfer learning training engine with early stopping & class balancing
├── evaluate.py                # Test set evaluation (Accuracy, Precision, Recall, F1, Confusion Matrices)
├── sample_data_setup.py       # Sample dataset generator and static UI demonstration assets
├── requirements.txt           # Python dependency specifications
├── run.bat                    # Windows 1-click application launcher
├── train.bat                  # Windows 1-click model training launcher
├── evaluate.bat               # Windows 1-click model evaluation launcher
├── static/
│   ├── css/style.css          # Modern dashboard styling (glassmorphism, status badges)
│   ├── js/main.js             # Client-side notifications and UI helpers
│   ├── js/admin.js            # Chart.js dashboards and confusion matrix interactive tabs
│   ├── uploads/               # Uploaded fruit imagery
│   ├── results/               # Generated Grad-CAM heatmaps & confusion matrix plots
│   └── samples/               # Embedded fruit test samples
└── templates/
    ├── base.html              # Responsive layout, navbar, and footer
    ├── index.html             # Landing page with 1-click test drive
    ├── upload.html            # Single image inspection with live preview & Grad-CAM toggle
    ├── result.html            # Visual inspection report & Grad-CAM side-by-side viewer
    ├── compare.html           # Live multi-model comparison & benchmark leaderboard
    ├── batch.html             # Batch screening table, KPIs, and CSV export
    ├── history.html           # Database audit log with search, filters, and CSV export
    └── admin.html             # Admin analytics, Chart.js graphs, and confusion matrices
```

---

## 🧠 Deep Learning Models

| Model Architecture | Parameters | Focus & Strengths | Grad-CAM Target Layer |
| :--- | :---: | :--- | :--- |
| **EfficientNet-B0** | ~4.3M | Compound scaling; balanced depth, width & image resolution | `backbone.features[-1]` |
| **ResNet-50** | ~23.8M | Deep 50-layer representation with identity skip connections | `backbone.layer4[-1]` |
| **MobileNet-V2** | ~2.4M | Inverted residuals & linear bottlenecks; ultra-fast mobile inference | `backbone.features[-1]` |

### Preprocessing & Regularization
- **Input Dimensions**: Resized to $224 \times 224 \times 3$
- **Normalization**: ImageNet standard statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Data Augmentation**: Random horizontal & vertical flips, random rotation ($\pm 15^\circ$), color jittering (brightness, contrast, saturation, hue), affine translation.
- **Class Balancing**: Weighted Cross-Entropy Loss to handle class imbalance across fruit categories.
- **Regularization**: Dropout ($p = 0.3$), Weight Decay ($10^{-4}$), and Learning Rate Plateau reduction with early stopping.

---

## 📊 Dataset Compatibility

### Supported Folder Conventions

The data pipeline in `data.py` automatically detects and processes:

1. **Original FreshHarvest Structure**:
   ```
   data/dataset/
   ├── F_Apple/         <-- Fresh Apple
   ├── S_Apple/         <-- Spoiled Apple
   ├── F_Banana/        <-- Fresh Banana
   ├── S_Banana/        <-- Spoiled Banana
   ├── F_Orange/        <-- Fresh Orange
   └── S_Orange/        <-- Spoiled Orange
   ```
2. **Kaggle Fresh and Rotten Fruit Datasets**:
   ```
   dataset/
   ├── freshapples/     <-- Fresh Apple
   ├── rottenapples/    <-- Spoiled Apple
   ├── freshbanana/     <-- Fresh Banana
   ├── rottenbanana/    <-- Spoiled Banana
   ├── freshoranges/    <-- Fresh Orange
   └── rottenoranges/   <-- Spoiled Orange
   ```
3. **Hierarchical / Subdirectory Formats**:
   - `train/fresh/apple/` and `train/spoiled/apple/`

### Recommended Kaggle Datasets
- **Fresh and Rotten Fruits for Classification** (`sriramr/fruits-fresh-and-rotten-for-classification`):
  ```bash
  kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
  ```
- Configure the dataset directory in `config.py`, through the environment variable `FRESHHARVEST_DATA`, or directly in the **Admin Dashboard** (`/admin`).

---

## 💻 Exact Windows Setup & Execution Guide

### Prerequisites
- Windows 10 or 11
- Python 3.11 or Python 3.13 installed

### Step 1: Clone or Open Workspace
Open PowerShell or Command Prompt in the project folder:
```powershell
cd "c:\Users\umesh\Downloads\anna project"
```

### Step 2: Install Python Dependencies
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
```
*(Or if standard python is in your PATH: `pip install -r requirements.txt`)*

### Step 3: Initialize Demo Samples (Optional / Pre-Configured)
The repository includes a ready-to-use sample dataset for instant verification:
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" sample_data_setup.py
```

### Step 4: Run the Web Application
Double-click `run.bat` or run via command line:
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" app.py
```
Open your web browser and navigate to:
👉 **`http://localhost:5000`**

---

## 🚀 Model Training & Evaluation Commands

### Train a Specific Model
To train MobileNet-V2 for 5 epochs:
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" train.py --model mobilenet_v2 --epochs 5 --batch-size 16
```

### Train All Three Models Sequentially
To train EfficientNet-B0, ResNet-50, and MobileNet-V2:
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" train.py --model all --epochs 5 --batch-size 16
```
*(Or double-click `train.bat`)*

### Evaluate Models on Held-Out Test Set
Computes Accuracy, Precision, Recall, F1-Score, and generates Confusion Matrix heatmaps:
```powershell
& "C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" evaluate.py
```
*(Or double-click `evaluate.bat`)*

---

## 🌐 Web Interface Walkthrough

1. **Home Page (`/`)**:
   - Live system statistics and architecture status.
   - 1-Click interactive test buttons for instant prediction on preloaded fruits.
2. **Inspect Image (`/upload`)**:
   - Drag-and-drop or browse image file with instant client-side preview.
   - Choose between Multi-Model Consensus or individual architectures.
   - Toggle Grad-CAM heatmap generation.
3. **Prediction Result (`/result/<id>`)**:
   - Side-by-side visualization: Original image alongside the Grad-CAM activation overlay.
   - Freshness verdict (`Fresh` in emerald green or `Spoiled` in crimson red).
   - Confidence percentage meter and inference latency.
   - Multi-model breakdown table comparing all 3 backbones.
4. **Model Comparison (`/compare`)**:
   - Compare all three architectures simultaneously on any uploaded image.
   - Real-time agreement consensus and speed vs. accuracy benchmark leaderboard.
5. **Batch Processing (`/batch`)**:
   - Upload dozens of fruit images or a `.zip` file for automated batch inspection.
   - Real-time progress, summary KPIs (total, fresh %, spoiled %, avg confidence).
   - Interactive data table and 1-click **Export to CSV**.
6. **Audit History (`/history`)**:
   - Full history of all historical predictions stored in SQLite.
   - Search by filename, fruit type, or audit notes.
   - Filter by freshness status (`All`, `Fresh`, `Spoiled`) or fruit variety.
   - View detailed report or export complete history to CSV.
7. **Admin Analytics (`/admin`)**:
   - Chart.js interactive charts: Fresh vs Spoiled donut, fruit variety breakdown bar chart, model performance radar/bar chart.
   - Interactive confusion matrix tables with green-gradient shading for all 3 models.
   - High-resolution confusion matrix heatmap plots.
   - Dataset directory configuration and re-evaluation triggers.

---

## 📜 Attribution & License

- **Base Reference**: This project builds upon concepts from [FreshHarvest](https://github.com/erickyegon/FreshHarvest) by Erick Kiprotich Yegon.
- **License**: Released under the [MIT License](https://opensource.org/licenses/MIT).
- All third-party libraries (PyTorch, Flask, OpenCV, Bootstrap, Chart.js) are property of their respective maintainers.

