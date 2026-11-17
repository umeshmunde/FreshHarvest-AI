"""
Training script for FreshHarvest AI.
Implements transfer learning, class-balanced loss, data augmentation,
validation tracking with early stopping, checkpointing, and test set evaluation.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
import argparse
import time
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE, DEFAULT_TRAIN_CONFIG, SUPPORTED_ARCHITECTURES, MODELS_DIR,
    DATA_DIR, METRICS_PATH
)
from data import prepare_dataloaders
from models import FruitFreshnessModel
from database import save_model_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def evaluate_loader(model, dataloader, criterion, device):
    """Run evaluation on validation or test DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc, all_preds, all_targets

def train_model(architecture="efficientnet_b0", dataset_path=DATA_DIR,
                epochs=10, batch_size=16, lr=0.0003, weight_decay=1e-4,
                patience=3, save_dir=MODELS_DIR):
    """
    Train a model with transfer learning on the fruit dataset.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f" Starting Training: {architecture.upper()} on {DEVICE}")
    print(f" Dataset Path: {dataset_path}")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"========================================================\n")

    # Load data
    data_bundle = prepare_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=0
    )
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    classes = data_bundle["classes"]
    class_weights = data_bundle["class_weights"].to(DEVICE)

    print(f"Detected {len(classes)} classes: {classes}")
    print(f"Dataset split: Train={data_bundle['stats']['train_count']}, Val={data_bundle['stats']['val_count']}, Test={data_bundle['stats']['test_count']}")

    # Build model
    model = FruitFreshnessModel(
        architecture=architecture,
        num_classes=len(classes),
        classes=classes,
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.3
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    save_filename = SUPPORTED_ARCHITECTURES[architecture]["default_weight_file"]
    save_filepath = save_dir / save_filename

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc * 100, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc * 100, 2))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            model.save(save_filepath)
            print(f"  --> Checkpoint saved to {save_filepath}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience: {patience}).")
                break

    total_time = round(time.time() - start_training_time, 1)
    print(f"\nTraining completed in {total_time}s. Best Val Acc: {best_val_acc*100:.2f}%")

    # Load best model for test evaluation
    best_model = FruitFreshnessModel.load(save_filepath, device=DEVICE)
    from evaluate import evaluate_model_on_loader
    eval_metrics = evaluate_model_on_loader(best_model, test_loader, classes, DEVICE)

    # Save metrics to DB
    display_name = SUPPORTED_ARCHITECTURES[architecture]["display_name"]
    save_model_metrics(
        model_key=architecture,
        model_name=display_name,
        accuracy=eval_metrics["accuracy"],
        precision=eval_metrics["precision"],
        recall=eval_metrics["recall"],
        f1_score=eval_metrics["f1_score"],
        test_loss=eval_metrics["test_loss"],
        confusion_matrix=eval_metrics["confusion_matrix"],
        class_report=eval_metrics["class_report"]
    )

    print(f"\nTest Evaluation for {display_name}:")
    print(f"  Accuracy:  {eval_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {eval_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {eval_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {eval_metrics['f1_score']*100:.2f}%")

    return eval_metrics

def main():
    parser = argparse.ArgumentParser(description="FreshHarvest AI Training Script")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "mobilenet_v2", "all"],
                        help="Model architecture to train")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR),
                        help="Path to fruit dataset directory (F_*/S_* or fresh*/rotten*)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default=str(MODELS_DIR), help="Output directory")

    args = parser.parse_args()

    if args.model == "all":
        for arch in ["efficientnet_b0", "resnet50", "mobilenet_v2"]:
            train_model(
                architecture=arch,
                dataset_path=args.data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                save_dir=args.save_dir
            )
    else:
        train_model(
            architecture=args.model,
            dataset_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
