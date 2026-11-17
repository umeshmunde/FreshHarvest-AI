"""
Generator for a Google Colab-ready, self-contained Jupyter Notebook for FreshHarvest AI.
Automatically configured with authenticated 1-click Kaggle download.
"""

import json
from pathlib import Path

def build_notebook():
    cells = []

    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip().splitlines(keepends=True)
        })

    def add_code(code):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.strip().splitlines(keepends=True)
        })

    # Header with Colab Badge
    add_markdown("""# FreshHarvest AI: Multi-Model Fruit Freshness & Quality Assessment System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Final-Year Academic & Engineering Project**  
*Deep Learning • PyTorch • Transfer Learning • Explainable AI (Grad-CAM) • Multi-Model Consensus*

---

### Project Objectives:
1. **Fruit Type Detection & Freshness Classification**: Simultaneously detect fruit species (Apple, Banana, Orange, Mango, Strawberry, Tomato, Lemon) and freshness condition (Fresh vs. Spoiled).
2. **Tri-Model Comparative Benchmarking**: Implement, train, and compare **EfficientNet-B0**, **ResNet-50**, and **MobileNet-V2**.
3. **Explainable AI (Grad-CAM)**: Generate convolutional feature activation heatmaps to interpret model predictions on decay and surface lesions.
4. **Comprehensive Evaluation**: Generate Accuracy, Precision, Recall, F1-Scores, and Confusion Matrices.
5. **Consensus Ensemble**: Multi-model decision voting for high-reliability agricultural QA.
""")

    # Kaggle Dataset Links & Direct Colab Download Section
    add_markdown("""## 📂 Kaggle Dataset Sources & Colab 1-Click Download

### 🔗 Official Kaggle Dataset Links:
1. **Primary Benchmark Dataset (Apples, Bananas, Oranges - 13,500+ Images)**:  
   👉 **[https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)**  
   *Author*: Sriram Reddy Kalluri | Widely cited standard benchmark in agricultural AI literature.
   
2. **Multi-Produce Dataset (Fruits & Vegetables)**:  
   👉 **[https://www.kaggle.com/datasets/raghavneuron/fresh-and-stale-images-of-fruits-and-vegetables](https://www.kaggle.com/datasets/raghavneuron/fresh-and-stale-images-of-fruits-and-vegetables)**  
   *Author*: Raghav Neuron | Fresh and stale produce including Apples, Bananas, Oranges, and Tomatoes.

---

### ⚡ Automatic Authenticated Dataset Download:
The code cell below is pre-configured with your Kaggle API credentials. In Google Colab, it will automatically download and extract all **13,500+ images** in about 25 seconds.
""")

    # Code cell for Colab dataset download
    add_code("""# ==============================================================================
# GOOGLE COLAB AUTOMATIC KAGGLE DATASET DOWNLOAD (13,500+ Images)
# ==============================================================================
# Primary Dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

import sys
import os
from pathlib import Path

# Detect if running in Google Colab
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    print("Running in Google Colab environment. Authenticating and downloading full Kaggle dataset...")
    
    # Configure Kaggle credentials automatically
    os.environ['KAGGLE_USERNAME'] = "umeshmunde"
    os.environ['KAGGLE_KEY'] = "58d1d4269e91491a48b6aed3a8293d9b"
    
    !pip install -q kaggle
    !mkdir -p ~/.kaggle
    !echo '{"username":"umeshmunde","key":"58d1d4269e91491a48b6aed3a8293d9b"}' > ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    
    # Download and extract the full 13,500+ image dataset
    !kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification --unzip -p ./data/fruits-fresh-and-rotten-for-classification
    
    dataset_path = Path("./data/fruits-fresh-and-rotten-for-classification")
    print(f"Full Kaggle dataset downloaded and ready at: {dataset_path.resolve()}")
else:
    print("Running in local environment. Using project dataset directory.")
    dataset_path = Path("data/dataset")
""")

    # Section 1: Setup & Imports
    add_markdown("## 1. Environment Setup & Dependency Imports")
    add_code("""import os
import sys
import types
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import cv2

# Device selection (CUDA GPU if available on Colab/Local, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Hardware Acceleration Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
""")

    # Section 2: Universal Dataset Loader (Self-Contained)
    add_markdown("""## 2. Universal Dataset Loader & Preprocessing
This self-contained module automatically scans and parses:
1. **Original FreshHarvest convention**: `F_<Fruit>` and `S_<Fruit>` (e.g. `F_Apple`, `S_Banana`)
2. **Kaggle convention**: `freshapples`, `rottenapples`, `freshbanana`, `rottenbanana`, etc. at any folder depth.
""")
    add_code("""def parse_fruit_and_freshness(name):
    clean = name.strip()
    freshness, fruit = None, None

    # Format 1: F_<Fruit> or S_<Fruit>
    if clean.startswith("F_") or clean.startswith("f_"):
        freshness = "Fresh"
        fruit = clean[2:].strip()
    elif clean.startswith("S_") or clean.startswith("s_"):
        freshness = "Spoiled"
        fruit = clean[2:].strip()
    # Format 2: fresh* / rotten* / spoiled* (Kaggle format)
    elif clean.lower().startswith("fresh"):
        freshness = "Fresh"
        fruit = re.sub(r"^fresh[_\-\s]*", "", clean, flags=re.IGNORECASE).strip()
    elif clean.lower().startswith("rotten") or clean.lower().startswith("spoiled"):
        freshness = "Spoiled"
        fruit = re.sub(r"^(rotten|spoiled)[_\-\s]*", "", clean, flags=re.IGNORECASE).strip()

    if fruit:
        fruit_clean = fruit.rstrip("sS").capitalize()
        if fruit_clean.lower() == "appl": fruit_clean = "Apple"
        elif fruit_clean.lower() == "orange": fruit_clean = "Orange"
        elif fruit_clean.lower() == "tomato": fruit_clean = "Tomato"
        elif fruit_clean.lower() == "mango": fruit_clean = "Mango"
    else:
        fruit_clean = "Unknown"

    if freshness is None: freshness = "Unknown"
    return freshness, fruit_clean, f"{freshness}_{fruit_clean}"

def scan_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        candidates = list(Path(".").glob("**/F_Apple")) + list(Path(".").glob("**/freshapples"))
        if candidates:
            dataset_dir = candidates[0].parent

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    records = []
    
    # Recursive search to support nested Kaggle folders (e.g. dataset/train/freshapples)
    if dataset_dir.exists():
        for folder in dataset_dir.rglob("*"):
            if folder.is_dir():
                freshness, fruit, unified_class = parse_fruit_and_freshness(folder.name)
                if freshness != "Unknown" and fruit != "Unknown":
                    for f in folder.glob("*"):
                        if f.is_file() and f.suffix.lower() in valid_exts:
                            records.append({
                                'image_path': str(f),
                                'fruit_type': fruit,
                                'freshness': freshness,
                                'class_name': unified_class
                            })

    # Starter images fallback if empty
    if not records:
        print("Creating starter images for demonstration...")
        dataset_dir = Path("./data/dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        demo_fruits = ["Apple", "Banana", "Orange", "Strawberry", "Tomato"]
        for d_fruit in demo_fruits:
            f_dir = dataset_dir / f"F_{d_fruit}"
            s_dir = dataset_dir / f"S_{d_fruit}"
            f_dir.mkdir(parents=True, exist_ok=True)
            s_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                img_f = Image.new('RGB', (224, 224), color=(34, 139, 34))
                img_f.save(f_dir / f"{d_fruit}_f_{i}.jpg")
                records.append({'image_path': str(f_dir / f"{d_fruit}_f_{i}.jpg"), 'fruit_type': d_fruit, 'freshness': 'Fresh', 'class_name': f"Fresh_{d_fruit}"})
                img_s = Image.new('RGB', (224, 224), color=(92, 64, 51))
                img_s.save(s_dir / f"{d_fruit}_s_{i}.jpg")
                records.append({'image_path': str(s_dir / f"{d_fruit}_s_{i}.jpg"), 'fruit_type': d_fruit, 'freshness': 'Spoiled', 'class_name': f"Spoiled_{d_fruit}"})

    df = pd.DataFrame(records)
    classes = sorted(df['class_name'].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    df['label_idx'] = df['class_name'].map(c2i)
    return df, classes

df, classes = scan_dataset(dataset_path)
print(f"Total dataset images found: {len(df)}")
print(f"Number of classes: {len(classes)} -> {classes}")
""")

    # Section 3: EDA Visualizations
    add_markdown("## 3. Exploratory Data Analysis (EDA) Visualizations")
    add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class Distribution
df['class_name'].value_counts().plot(kind='bar', ax=axes[0], color='#10b981', edgecolor='black')
axes[0].set_title("Class Frequency Distribution", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Class Name")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Fresh vs Spoiled Ratio
fresh_counts = df['freshness'].value_counts()
axes[1].pie(fresh_counts, labels=fresh_counts.index, autopct='%1.1f%%',
            colors=['#10b981', '#ef4444'], startangle=90, explode=(0.04, 0.04))
axes[1].set_title("Fresh vs. Spoiled Ratio", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
""")

    # Section 4: Data Augmentation & Loaders
    add_markdown("""## 4. Data Augmentation & PyTorch DataLoaders
Transforms implement ImageNet standard normalization:
$$\\mu = [0.485, 0.456, 0.406], \\quad \\sigma = [0.229, 0.224, 0.225]$$
Along with random flips, rotation (+/- 15 degrees), and color jittering.
""")
    add_code("""IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class FruitDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, int(row['label_idx']), row['fruit_type'], row['freshness']

# Stratified Split (70% Train, 15% Val, 15% Test)
train_df, rem_df = train_test_split(df, train_size=0.70, random_state=42, stratify=df['label_idx'])
val_df, test_df = train_test_split(rem_df, train_size=0.50, random_state=42, stratify=rem_df['label_idx'])

train_loader = DataLoader(FruitDataset(train_df, train_transforms), batch_size=16, shuffle=True)
val_loader = DataLoader(FruitDataset(val_df, eval_transforms), batch_size=16, shuffle=False)
test_loader = DataLoader(FruitDataset(test_df, eval_transforms), batch_size=16, shuffle=False)

# Compute class balancing weights
counts = train_df['label_idx'].value_counts().to_dict()
weights = [len(train_df) / (len(classes) * counts.get(i, 1)) for i in range(len(classes))]
class_weights = torch.FloatTensor(weights).to(DEVICE)

print(f"Training Samples:   {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")
print(f"Testing Samples:    {len(test_df)}")
""")

    # Section 5: Model Factory
    add_markdown("""## 5. Model Factory: EfficientNet-B0, ResNet-50, MobileNet-V2
We construct transfer learning models using ImageNet weights and customized classification heads.
""")
    add_code("""def build_model(arch, num_classes, dropout_rate=0.3):
    arch = arch.lower()
    if arch == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights='DEFAULT')
        except Exception:
            model = models.efficientnet_b0(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    elif arch == "resnet50":
        try:
            model = models.resnet50(weights='DEFAULT')
        except Exception:
            model = models.resnet50(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    elif arch == "mobilenet_v2":
        try:
            model = models.mobilenet_v2(weights='DEFAULT')
        except Exception:
            model = models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    return model.to(DEVICE)

# Parameter Summary
model_eff = build_model("efficientnet_b0", len(classes))
model_res = build_model("resnet50", len(classes))
model_mob = build_model("mobilenet_v2", len(classes))

param_summary = pd.DataFrame([
    {"Architecture": "EfficientNet-B0", "Total Parameters": f"{sum(p.numel() for p in model_eff.parameters()):,}"},
    {"Architecture": "ResNet-50", "Total Parameters": f"{sum(p.numel() for p in model_res.parameters()):,}"},
    {"Architecture": "MobileNet-V2", "Total Parameters": f"{sum(p.numel() for p in model_mob.parameters()):,}"}
])
param_summary
""")

    # Section 6: Training & Evaluation Functions
    add_markdown("## 6. Training & Evaluation Engine")
    add_code("""def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls, _, _ in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    return running_loss / total, (correct / total) * 100

def evaluate_model(model, loader):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, lbls, _, _ in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outs = model(imgs)
            loss = criterion(outs, lbls)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(lbls.cpu().numpy().tolist())
            total += lbls.size(0)

    acc = accuracy_score(all_targets, all_preds) * 100
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(all_targets, all_preds)
    return {
        "loss": running_loss / total, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1, "confusion_matrix": cm,
        "predictions": all_preds, "targets": all_targets
    }

print("Training routines compiled successfully.")
""")

    # Section 7: Multi-Model Training Demo & Test Set Evaluation
    add_markdown("## 7. Comparative Test Set Evaluation & Benchmark Leaderboard")
    add_code("""# Quick demonstration training for 1 epoch on each model
trained_models = {"EfficientNet-B0": model_eff, "ResNet-50": model_res, "MobileNet-V2": model_mob}
benchmark_metrics = []

criterion = nn.CrossEntropyLoss(weight=class_weights)

for name, model in trained_models.items():
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    print(f"Training {name} for 1 epoch demonstration...")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    print(f"  --> {name}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Evaluate on held-out test set
    metrics = evaluate_model(model, test_loader)
    benchmark_metrics.append({
        "Model": name,
        "Accuracy (%)": round(metrics["accuracy"], 2),
        "Precision (%)": round(metrics["precision"], 2),
        "Recall (%)": round(metrics["recall"], 2),
        "F1-Score (%)": round(metrics["f1_score"], 2),
        "Test Loss": round(metrics["loss"], 4),
        "cm": metrics["confusion_matrix"]
    })

results_df = pd.DataFrame(benchmark_metrics).drop(columns=['cm'])
display(results_df)

# Grouped Bar Plot
fig, ax = plt.subplots(figsize=(10, 5))
results_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]].plot(
    kind="bar", ax=ax, colormap="viridis", edgecolor="black")
plt.title("Comparative Performance Across Architectures on Test Set", fontsize=14, pad=10)
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
""")

    # Section 8: Confusion Matrix Heatmaps
    add_markdown("## 8. Confusion Matrix Heatmaps for All Three Models")
    add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, res in enumerate(benchmark_metrics):
    ax = axes[i]
    cm = res["cm"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False,
                xticklabels=[c.replace('_', '\\n') for c in classes[:len(cm)]],
                yticklabels=classes[:len(cm)])
    ax.set_title(f"Confusion Matrix: {res['Model']}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Multi-Model Confusion Matrices", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""")

    # Section 9: Grad-CAM Explainable AI
    add_markdown("""## 9. Explainable AI: Grad-CAM Feature Activation Heatmaps
Grad-CAM highlights regions of the input fruit image that contributed most heavily to the classification:
$$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial y^c}{\\partial A_{i,j}^k}, \\quad L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$
""")
    add_code("""class GradCAMExplainer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self._f_hook),
            target_layer.register_full_backward_hook(self._b_hook)
        ]
    def _f_hook(self, m, i, o): self.activations = o.detach()
    def _b_hook(self, m, gi, go): self.gradients = go[0].detach()
    def remove(self):
        for h in self.hooks: h.remove()
    def generate(self, tensor_img, target_class=None):
        self.model.zero_grad()
        logits = self.model(tensor_img)
        if target_class is None: target_class = torch.argmax(logits, dim=1).item()
        logits[0, target_class].backward(retain_graph=True)
        w = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(w * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        c_min, c_max = np.min(cam), np.max(cam)
        return (cam - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else np.zeros_like(cam)

# ==============================================================================
# Interactive Image Upload for Grad-CAM Analysis
# Supports Google Colab file upload and local image paths
# ==============================================================================
import io

uploaded_img = None
if 'google.colab' in sys.modules:
    try:
        from google.colab import files
        print("📁 Click the button below to upload your own fruit image:")
        uploaded = files.upload()
        if uploaded:
            fname = list(uploaded.keys())[0]
            uploaded_img = Image.open(io.BytesIO(uploaded[fname])).convert('RGB')
            print(f"✅ Loaded uploaded image: {fname}")
    except Exception as e:
        print(f"Upload helper: {e}")

# Fallback to local sample or dataset if not in Colab or no file uploaded
if uploaded_img is None:
    custom_path = "static/samples/sample_fresh_apple.jpg"
    if os.path.exists(custom_path):
        uploaded_img = Image.open(custom_path).convert('RGB')
        print(f"ℹ️ Testing with sample image: {custom_path}")
    elif 'df' in locals() and len(df) > 0:
        uploaded_img = Image.open(df.iloc[0]['image_path']).convert('RGB')
        print("ℹ️ Testing with dataset image.")
    else:
        uploaded_img = Image.new('RGB', (224, 224), color=(180, 50, 50))

sample_pil = uploaded_img
input_tensor = eval_transforms(sample_pil).unsqueeze(0).to(DEVICE)

cam_targets = {
    "EfficientNet-B0": (model_eff, model_eff.features[-1]),
    "ResNet-50": (model_res, model_res.layer4[-1]),
    "MobileNet-V2": (model_mob, model_mob.features[-1])
}

# Run inference to get predictions and confidences for each model
model_preds = {}
for mname, (m, _) in cam_targets.items():
    m.eval()
    with torch.no_grad():
        out = m(input_tensor)
        probs = F.softmax(out, dim=1)
        conf, p_idx = torch.max(probs, dim=1)
        model_preds[mname] = (classes[p_idx.item()], conf.item() * 100)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(sample_pil)
axes[0].set_title("Input Image", fontsize=11, fontweight='bold')
axes[0].axis('off')

for i, (mname, (m, layer)) in enumerate(cam_targets.items()):
    p_class, p_conf = model_preds[mname]
    explainer = GradCAMExplainer(m, layer)
    cam = explainer.generate(input_tensor)
    explainer.remove()
    
    # Resize and blend with OpenCV
    orig_np = np.array(sample_pil)
    cam_resized = cv2.resize(cam, (orig_np.shape[1], orig_np.shape[0]))
    cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig_np, 0.6, cam_rgb, 0.4, 0)
    
    axes[i + 1].imshow(blended)
    axes[i + 1].set_title(f"{mname}\\n{p_class} ({p_conf:.1f}%)", fontsize=10, fontweight='bold')
    axes[i + 1].axis('off')

plt.suptitle("Explainable AI (Grad-CAM) Visual Attentions", fontsize=15, y=0.98)
plt.tight_layout()
plt.show()
""")

    # Section 10: Multi-Model Consensus & Inference
    add_markdown("## 10. Multi-Model Consensus & Live Inference")
    add_code("""# Run consensus inference across all three architectures
print("=" * 65)
print("MULTI-MODEL CONSENSUS INFERENCE REPORT")
print("=" * 65)

votes_freshness = []
votes_fruit = []

for mname, model in trained_models.items():
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_tensor)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        top_idx = int(np.argmax(probs))
        pred_class = classes[top_idx]
        conf = round(float(probs[top_idx]) * 100, 2)
        latency = round((time.time() - t0) * 1000, 2)
        
        freshness, fruit, _ = parse_fruit_and_freshness(pred_class)
        votes_freshness.append(freshness)
        votes_fruit.append(fruit)
        print(f"[{mname:<15}] -> Predicted: {freshness:<8} | Fruit: {fruit:<10} | Conf: {conf:>6.2f}% | Latency: {latency:>5.1f} ms")

consensus_fresh = max(set(votes_freshness), key=votes_freshness.count)
consensus_fruit = max(set(votes_fruit), key=votes_fruit.count)
agreement_ratio = f"{votes_freshness.count(consensus_fresh)}/3 Agree"

print("-" * 65)
print(f"Consensus Quality Verdict: {consensus_fresh.upper()} ({agreement_ratio})")
print(f"Consensus Fruit Species:   {consensus_fruit}")
print("=" * 65)
""")

    # Section 11: Viva Summary
    add_markdown("""## 11. Conclusion & Viva Defense Summary (Exam Quick Reference)

### Technical Highlights:
1. **Multi-Model Triad**: Evaluated compound scaling (**EfficientNet-B0**), deep residual skips (**ResNet-50**), and inverted bottlenecks (**MobileNet-V2**).
2. **Explainability**: Integrated **Grad-CAM** transforms the "black box" neural network into an interpretable system for agricultural inspectors.
3. **Dual Identification**: Combines fruit variety recognition and binary freshness assessment in a unified pipeline.
4. **Kaggle & FreshHarvest Compatibility**: Interoperable across Kaggle benchmarks and the original `F_`/`S_` folder standard.
""")

    notebook_content = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "FreshHarvest_AI_Project_Notebook.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = Path("FreshHarvest_AI_Project_Notebook.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"Notebook generated for Google Colab at: {output_path.resolve()}")

if __name__ == "__main__":
    build_notebook()
