"""
Evaluation module for FreshHarvest AI.
Computes Accuracy, Precision, Recall, F1-score, Confusion Matrices,
and per-class performance for all three architectures.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config import (
    DEVICE, SUPPORTED_ARCHITECTURES, MODELS_DIR, DATA_DIR,
    METRICS_PATH, RESULTS_DIR
)
from data import prepare_dataloaders
from models import FruitFreshnessModel
from database import save_model_metrics, get_all_model_metrics

def evaluate_model_on_loader(model, dataloader, classes, device=DEVICE):
    """
    Run evaluation on a PyTorch DataLoader and return metrics dictionary.
    """
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    total = 0

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)

    test_loss = running_loss / max(total, 1)
    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    cm = confusion_matrix(all_targets, all_preds).tolist()
    cr = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "test_loss": round(test_loss, 4),
        "confusion_matrix": cm,
        "class_report": cr,
        "classes": classes
    }

def plot_and_save_confusion_matrix(cm, classes, model_name, output_path):
    """Render and save a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(f"Confusion Matrix – {model_name}", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()

def evaluate_all_models(dataset_path=DATA_DIR):
    """
    Evaluate all three models on the test set and update database and JSON summaries.
    """
    print("\n--- Evaluating Models on Test Set ---")
    data_bundle = prepare_dataloaders(dataset_path=dataset_path, num_workers=0)
    test_loader = data_bundle["test_loader"]
    classes = data_bundle["classes"]

    summary = {}

    for arch_key, arch_info in SUPPORTED_ARCHITECTURES.items():
        weight_file = MODELS_DIR / arch_info["default_weight_file"]
        display_name = arch_info["display_name"]

        if weight_file.exists():
            model = FruitFreshnessModel.load(weight_file, device=DEVICE)
        else:
            model = FruitFreshnessModel(architecture=arch_key, num_classes=len(classes), classes=classes, pretrained=True)

        metrics = evaluate_model_on_loader(model, test_loader, classes, DEVICE)
        summary[arch_key] = {
            "model_name": display_name,
            "metrics": metrics
        }

        # Plot confusion matrix
        cm_path = RESULTS_DIR / f"cm_{arch_key}.png"
        plot_and_save_confusion_matrix(metrics["confusion_matrix"], classes, display_name, cm_path)

        # Save to database
        save_model_metrics(
            model_key=arch_key,
            model_name=display_name,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            test_loss=metrics["test_loss"],
            confusion_matrix=metrics["confusion_matrix"],
            class_report=metrics["class_report"]
        )

    # Save summary JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics saved to {METRICS_PATH} and SQLite database.")
    return summary

def main():
    parser = argparse.ArgumentParser(description="FreshHarvest AI Model Evaluator")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR), help="Dataset directory")
    args = parser.parse_args()
    evaluate_all_models(args.data_path)

if __name__ == "__main__":
    main()
