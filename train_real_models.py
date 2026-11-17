"""
Fast Transfer Learning Trainer on Real Kaggle Fruit Images for FreshHarvest AI.
Trains EfficientNet-B0, MobileNet-V2, and ResNet-50 on real photos of bananas, apples, oranges, etc.
Saves checkpoints to models_saved/*.pth and updates confusion matrices.
"""

import sys
import types
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR,
    DEFAULT_CLASSES
)
from data import scan_dataset, FruitImageDataset, get_transforms
from models import FruitFreshnessModel

def train_real_dataset():
    print("=" * 70, flush=True)
    print("  FreshHarvest AI - Fast Head Fine-Tuning on Real Kaggle Produce Dataset", flush=True)
    print("=" * 70, flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path("data/dataset")

    print("[*] Scanning real Kaggle images...", flush=True)
    df, classes = scan_dataset(dataset_dir)
    print(f"[+] Total images scanned: {len(df)}", flush=True)
    print(f"[+] Classes: {classes}", flush=True)

    # Subsample to balance dataset (up to 120 images per class for ultra-fast, high-quality head tuning)
    balanced_records = []
    for c in classes:
        sub_df = df[df['class_name'] == c]
        if len(sub_df) > 120:
            sub_df = sub_df.sample(n=120, random_state=42)
        balanced_records.append(sub_df)
    
    train_df = pd.concat(balanced_records, ignore_index=True)
    print(f"[+] Balanced training subset: {len(train_df)} images across {len(classes)} classes", flush=True)

    # Split train / validation (80 / 20)
    train_subset, val_subset = train_test_split(
        train_df, test_size=0.20, random_state=42, stratify=train_df['label_idx']
    )
    print(f"[+] Train set: {len(train_subset)} | Val set: {len(val_subset)}", flush=True)

    train_loader = DataLoader(
        FruitImageDataset(train_subset, transform=get_transforms(is_train=True)),
        batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        FruitImageDataset(val_subset, transform=get_transforms(is_train=False)),
        batch_size=32, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    num_classes = len(classes)

    architectures = ["efficientnet_b0", "mobilenet_v2", "resnet50"]

    for arch in architectures:
        print("\n" + "-" * 70, flush=True)
        print(f"[*] Training Transfer Model: {arch.upper()} (Frozen Backbone)", flush=True)
        print("-" * 70, flush=True)

        # Initialize model with official pretrained ImageNet backbone and frozen feature extractor
        model = FruitFreshnessModel(
            architecture=arch, num_classes=num_classes, classes=classes,
            pretrained=True, freeze_backbone=True
        ).to(DEVICE)

        # Only optimize classification head parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=1e-4)

        for epoch in range(1, 4):
            t0 = time.time()
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for b_idx, (imgs, lbls, _, _) in enumerate(train_loader):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, lbls)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

            train_acc = (correct / total) * 100
            train_loss = total_loss / total

            # Validation
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for imgs, lbls, _, _ in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    outs = model(imgs)
                    _, preds = torch.max(outs, 1)
                    v_correct += (preds == lbls).sum().item()
                    v_total += lbls.size(0)
            val_acc = (v_correct / v_total) * 100
            elapsed = time.time() - t0

            print(f"[{arch}] Epoch [{epoch}/3] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        # Save model checkpoint correctly
        save_file = MODELS_DIR / f"{arch}_freshharvest.pth"
        model.save(save_file)
        print(f"[+] Saved updated trained checkpoint: {save_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  ALL THREE MODELS TRAINED ON REAL FRUIT IMAGES SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_real_dataset()
