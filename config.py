"""
Configuration module for FreshHarvest AI.
Extended and upgraded from the base FreshHarvest reference by erickyegon.
Provides paths, model hyperparameters, dataset mappings, and application settings.
"""

import os
from pathlib import Path
import torch

# Fix for OpenMP multiple runtime error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
SAMPLES_DIR = STATIC_DIR / "samples"
MODELS_DIR = BASE_DIR / "models_saved"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = Path(os.environ.get("FRESHHARVEST_DATA", BASE_DIR / "data" / "dataset"))

# Database path
DB_PATH = BASE_DIR / "database.db"
METRICS_PATH = BASE_DIR / "metrics_summary.json"

# Ensure all runtime folders exist
for folder in [STATIC_DIR, UPLOAD_DIR, RESULTS_DIR, SAMPLES_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Device configuration (CUDA GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing parameters
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported architectures
SUPPORTED_ARCHITECTURES = {
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "description": "Lightweight compound-scaled CNN, high parameter efficiency",
        "default_weight_file": "efficientnet_b0_freshharvest.pth",
        "gradcam_layer": "features"
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "description": "Deep 50-layer residual network with skip connections",
        "default_weight_file": "resnet50_freshharvest.pth",
        "gradcam_layer": "layer4"
    },
    "mobilenet_v2": {
        "display_name": "MobileNet-V2",
        "description": "Ultra-fast inverted residual model optimized for mobile/edge",
        "default_weight_file": "mobilenet_v2_freshharvest.pth",
        "gradcam_layer": "features"
    }
}

# Standard fruit types and classes
FRUIT_TYPES = ["Apple", "Banana", "Orange", "Mango", "Strawberry", "Tomato", "Lemon"]

# Standard 14-class mapping for multi-fruit quality classification
DEFAULT_CLASSES = [
    "Fresh_Apple", "Spoiled_Apple",
    "Fresh_Banana", "Spoiled_Banana",
    "Fresh_Orange", "Spoiled_Orange",
    "Fresh_Mango", "Spoiled_Mango",
    "Fresh_Strawberry", "Spoiled_Strawberry",
    "Fresh_Tomato", "Spoiled_Tomato",
    "Fresh_Lemon", "Spoiled_Lemon"
]

# Training defaults
DEFAULT_TRAIN_CONFIG = {
    "batch_size": 16,
    "num_workers": 0,  # 0 is most reliable on Windows
    "learning_rate": 0.0003,
    "weight_decay": 1e-4,
    "epochs": 10,
    "early_stopping_patience": 3,
    "dropout_rate": 0.3,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15
}

# Allowed file formats for web upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'zip'}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB max upload size

