"""
Inference engine module for FreshHarvest AI.
Provides single-model and multi-model prediction, Grad-CAM heatmap visualization,
consensus calculation, and batch processing.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (
    DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEFAULT_CLASSES,
    SUPPORTED_ARCHITECTURES, MODELS_DIR, RESULTS_DIR, UPLOAD_DIR
)
from models import FruitFreshnessModel
from gradcam import generate_and_save_gradcam
from database import save_prediction

# Memory cache for loaded models
_LOADED_MODELS = {}

def get_inference_transform():
    """Standard normalization transform for inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_or_load_model(architecture="efficientnet_b0", num_classes=14):
    """
    Retrieve cached model instance or load/initialize on demand.
    Checks `models_saved/` for trained weights. If absent, initializes
    pretrained transfer model.
    """
    arch = architecture.lower()
    if arch not in SUPPORTED_ARCHITECTURES:
        arch = "efficientnet_b0"

    if arch in _LOADED_MODELS:
        return _LOADED_MODELS[arch]

    weight_filename = SUPPORTED_ARCHITECTURES[arch]["default_weight_file"]
    weight_path = MODELS_DIR / weight_filename

    if weight_path.exists():
        try:
            model = FruitFreshnessModel.load(weight_path, device=DEVICE)
            print(f"[Inference] Loaded trained checkpoint for {arch} from {weight_path}")
        except Exception as e:
            print(f"[Inference] Could not load checkpoint ({e}). Initializing backbone.")
            model = FruitFreshnessModel(architecture=arch, num_classes=num_classes, pretrained=True)
    else:
        # Initialize pretrained backbone
        model = FruitFreshnessModel(architecture=arch, num_classes=num_classes, pretrained=True)
        print(f"[Inference] Initialized transfer model {arch}")

    model.eval()
    _LOADED_MODELS[arch] = model
    return model

def parse_class_name(class_name):
    """
    Split a class name into freshness and fruit type.
    Example: 'Fresh_Apple' -> ('Fresh', 'Apple')
    """
    if "_" in class_name:
        parts = class_name.split("_", 1)
        freshness = parts[0].capitalize()
        fruit_type = parts[1].capitalize()
    elif class_name.lower() in ["fresh", "spoiled", "rotten"]:
        freshness = "Fresh" if class_name.lower() == "fresh" else "Spoiled"
        fruit_type = "Fruit"
    else:
        freshness = "Fresh"
        fruit_type = class_name.capitalize()

    if freshness.lower() == "rotten":
        freshness = "Spoiled"

    return freshness, fruit_type

def predict_single(image_path_or_pil, model_name="efficientnet_b0",
                   generate_gradcam=True, save_to_db=True, notes=None):
    """
    Run inference on a single fruit image using the specified architecture.

    Args:
        image_path_or_pil: Filepath or PIL.Image
        model_name (str): Architecture key
        generate_gradcam (bool): Whether to generate explainability heatmap
        save_to_db (bool): Save result to SQLite
        notes (str): Optional audit note

    Returns:
        dict: Complete inference results, probabilities, and Grad-CAM paths
    """
    # Load and validate image
    if isinstance(image_path_or_pil, (str, Path)):
        img_path = Path(image_path_or_pil)
        image = Image.open(str(img_path)).convert("RGB")
        filename = img_path.name
        rel_orig_path = f"uploads/{filename}"
    else:
        image = image_path_or_pil.convert("RGB")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        save_dest = UPLOAD_DIR / filename
        image.save(str(save_dest))
        rel_orig_path = f"uploads/{filename}"

    # Preprocess
    transform = get_inference_transform()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Load model
    model = get_or_load_model(model_name)
    classes = getattr(model, "classes", DEFAULT_CLASSES)

    # Measure latency
    start_time = time.time()
    logits = model(input_tensor)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx] if top_idx < len(classes) else f"Class_{top_idx}"
    top_confidence = float(probs[top_idx])

    # Extract freshness and fruit type
    freshness, fruit_type = parse_class_name(top_class)

    # Calculate overall freshness probability
    fresh_prob = sum(float(probs[i]) for i, c in enumerate(classes) if c.startswith("Fresh"))
    spoiled_prob = sum(float(probs[i]) for i, c in enumerate(classes) if c.startswith("Spoiled") or c.startswith("Rotten"))

    # Normalize if both are zero
    total_fs = fresh_prob + spoiled_prob
    if total_fs > 0:
        fresh_prob /= total_fs
        spoiled_prob /= total_fs
    else:
        fresh_prob = top_confidence if freshness == "Fresh" else 1.0 - top_confidence
        spoiled_prob = 1.0 - fresh_prob

    # Class breakdown
    top_classes = []
    sorted_indices = np.argsort(probs)[::-1][:5]
    for idx in sorted_indices:
        cname = classes[idx] if idx < len(classes) else f"Class_{idx}"
        f_val, f_type = parse_class_name(cname)
        top_classes.append({
            "class_name": cname,
            "fruit_type": f_type,
            "freshness": f_val,
            "probability": round(float(probs[idx]) * 100, 2)
        })

    # Grad-CAM generation
    rel_heatmap_path = None
    if generate_gradcam:
        try:
            cam_filename = f"cam_{model_name}_{Path(filename).stem}.jpg"
            cam_full_path = RESULTS_DIR / cam_filename
            generate_and_save_gradcam(
                model=model,
                input_tensor=input_tensor,
                original_image=image,
                output_path=cam_full_path,
                target_class=top_idx
            )
            rel_heatmap_path = f"results/{cam_filename}"
        except Exception as e:
            print(f"[Grad-CAM] Error generating heatmap: {e}")

    result = {
        "filename": filename,
        "original_image_path": rel_orig_path,
        "heatmap_path": rel_heatmap_path,
        "model_key": model_name,
        "model_display_name": SUPPORTED_ARCHITECTURES.get(model_name, {}).get("display_name", model_name),
        "fruit_type": fruit_type,
        "freshness": freshness,
        "confidence": round(top_confidence * 100, 2),
        "raw_confidence": top_confidence,
        "fresh_probability": round(fresh_prob * 100, 2),
        "spoiled_probability": round(spoiled_prob * 100, 2),
        "latency_ms": latency_ms,
        "top_classes": top_classes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to history database
    if save_to_db:
        record_id = save_prediction(
            filename=filename,
            original_image_path=rel_orig_path,
            heatmap_path=rel_heatmap_path,
            model_name=result["model_display_name"],
            fruit_type=fruit_type,
            freshness=freshness,
            confidence=top_confidence,
            latency_ms=latency_ms,
            notes=notes
        )
        result["record_id"] = record_id

    return result

def predict_multi_model(image_path_or_pil, generate_gradcam=True, save_to_db=True, notes=None):
    """
    Run predictions using ALL THREE models (EfficientNetB0, ResNet50, MobileNetV2),
    compare their confidence scores, latencies, and compute consensus.

    Args:
        image_path_or_pil: Input image path or PIL Image
        generate_gradcam (bool): Whether to generate Grad-CAM for models
        save_to_db (bool): Save consolidated record to SQLite

    Returns:
        dict: Multi-model comparison report and consensus assessment
    """
    results = {}
    architectures = ["efficientnet_b0", "resnet50", "mobilenet_v2"]

    for arch in architectures:
        pred = predict_single(
            image_path_or_pil=image_path_or_pil,
            model_name=arch,
            generate_gradcam=generate_gradcam,
            save_to_db=False  # Save consolidated later
        )
        results[arch] = pred

    # Consensus calculation
    freshness_votes = [results[a]["freshness"] for a in architectures]
    fruit_votes = [results[a]["fruit_type"] for a in architectures]

    fresh_votes_count = freshness_votes.count("Fresh")
    spoiled_votes_count = freshness_votes.count("Spoiled")

    consensus_freshness = "Fresh" if fresh_votes_count > spoiled_votes_count else "Spoiled"
    agreement_ratio = f"{max(fresh_votes_count, spoiled_votes_count)}/3 Agree"

    # Fruit type consensus (most common)
    fruit_consensus = max(set(fruit_votes), key=fruit_votes.count)

    # Average confidence & latency
    avg_conf = np.mean([results[a]["confidence"] for a in architectures])
    avg_lat = np.mean([results[a]["latency_ms"] for a in architectures])

    # Find highest confidence model
    best_model_arch = max(architectures, key=lambda a: results[a]["confidence"])
    best_model = results[best_model_arch]

    consensus_summary = {
        "consensus_freshness": consensus_freshness,
        "consensus_fruit_type": fruit_consensus,
        "agreement_ratio": agreement_ratio,
        "agreement_percentage": round((max(fresh_votes_count, spoiled_votes_count) / 3.0) * 100, 1),
        "best_model": results[best_model_arch]["model_display_name"],
        "best_model_key": best_model_arch,
        "average_confidence": round(float(avg_conf), 2),
        "average_latency_ms": round(float(avg_lat), 2)
    }

    # Use best model's visual for main record
    record_id = None
    if save_to_db:
        record_id = save_prediction(
            filename=best_model["filename"],
            original_image_path=best_model["original_image_path"],
            heatmap_path=best_model["heatmap_path"],
            model_name="Multi-Model Ensemble",
            fruit_type=fruit_consensus,
            freshness=consensus_freshness,
            confidence=float(avg_conf / 100.0),
            latency_ms=avg_lat,
            all_models_summary={
                "agreement": agreement_ratio,
                "models": {a: {"pred": results[a]["freshness"], "conf": results[a]["confidence"], "lat": results[a]["latency_ms"]} for a in architectures}
            },
            notes=notes
        )

    return {
        "models": results,
        "consensus": consensus_summary,
        "primary_result": best_model,
        "record_id": record_id
    }

def predict_batch(image_paths, model_name="efficientnet_b0"):
    """
    Execute batch prediction across multiple image files.

    Args:
        image_paths (list): List of image file paths
        model_name (str): Selected model architecture

    Returns:
        dict: Batch summary, items list, and counts
    """
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    items = []
    fresh_count = 0
    spoiled_count = 0
    total_conf = 0.0

    for path in image_paths:
        try:
            pred = predict_single(
                image_path_or_pil=path,
                model_name=model_name,
                generate_gradcam=False,  # Skip heatmaps during bulk for speed
                save_to_db=True,
                notes=f"Batch ID: {batch_id}"
            )
            if pred["freshness"] == "Fresh":
                fresh_count += 1
            else:
                spoiled_count += 1
            total_conf += pred["confidence"]

            items.append({
                "filename": pred["filename"],
                "fruit_type": pred["fruit_type"],
                "freshness": pred["freshness"],
                "confidence": pred["confidence"],
                "latency_ms": pred["latency_ms"],
                "image_path": pred["original_image_path"]
            })
        except Exception as e:
            items.append({
                "filename": Path(path).name,
                "fruit_type": "Unknown",
                "freshness": "Error",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "error": str(e)
            })

    total = len(items)
    avg_confidence = round(total_conf / total, 2) if total > 0 else 0.0

    return {
        "batch_id": batch_id,
        "total": total,
        "fresh_count": fresh_count,
        "spoiled_count": spoiled_count,
        "fresh_percentage": round((fresh_count / total * 100), 1) if total > 0 else 0,
        "spoiled_percentage": round((spoiled_count / total * 100), 1) if total > 0 else 0,
        "average_confidence": avg_confidence,
        "model_name": SUPPORTED_ARCHITECTURES.get(model_name, {}).get("display_name", model_name),
        "items": items
    }
