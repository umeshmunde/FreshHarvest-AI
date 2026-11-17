"""
Models module for FreshHarvest AI.
Provides unified model creation, transfer learning backbones, and checkpoint management
for EfficientNet-B0, ResNet-50, and MobileNet-V2.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, DEFAULT_CLASSES, SUPPORTED_ARCHITECTURES, MODELS_DIR

def create_model(architecture="efficientnet_b0", num_classes=14, pretrained=True,
                 freeze_backbone=False, dropout_rate=0.3):
    """
    Build transfer learning model with customized classification head.

    Args:
        architecture (str): One of 'efficientnet_b0', 'resnet50', 'mobilenet_v2'
        num_classes (int): Number of target classification outputs
        pretrained (bool): Whether to use ImageNet pretrained backbone weights
        freeze_backbone (bool): Freeze feature extractor layers during initial training
        dropout_rate (float): Dropout probability for regularization

    Returns:
        nn.Module: PyTorch neural network model
    """
    arch = architecture.lower()
    weights_param = "DEFAULT" if pretrained else None

    if arch == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=weights_param)
        except Exception:
            # Fallback if download blocked or offline
            model = models.efficientnet_b0(weights=None)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, num_classes)
        )

    elif arch == "resnet50":
        try:
            model = models.resnet50(weights=weights_param)
        except Exception:
            model = models.resnet50(weights=None)

        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, num_classes)
        )

    elif arch == "mobilenet_v2":
        try:
            model = models.mobilenet_v2(weights=weights_param)
        except Exception:
            model = models.mobilenet_v2(weights=None)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, num_classes)
        )

    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'. Choose from {list(SUPPORTED_ARCHITECTURES.keys())}")

    return model.to(DEVICE)

class FruitFreshnessModel(nn.Module):
    """
    Unified Fruit Freshness and Quality Assessment Model wrapper.
    Encapsulates architecture, forward pass, saving, and loading checkpoints.
    """

    def __init__(self, architecture="efficientnet_b0", num_classes=14,
                 classes=None, pretrained=True, freeze_backbone=False, dropout_rate=0.3):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.classes = classes if classes is not None else list(DEFAULT_CLASSES[:num_classes])
        self.backbone = create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the final convolutional layer targeted for Grad-CAM."""
        if self.architecture == "efficientnet_b0":
            return self.backbone.features[-1]
        elif self.architecture == "resnet50":
            return self.backbone.layer4[-1]
        elif self.architecture == "mobilenet_v2":
            return self.backbone.features[-1]
        else:
            raise ValueError(f"Unknown target layer for {self.architecture}")

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save(self, filepath):
        """Save complete model weights and metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "state_dict": self.state_dict()
        }, str(filepath))

    @classmethod
    def load(cls, filepath, device=DEVICE):
        """Load a saved FruitFreshnessModel from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        checkpoint = torch.load(str(filepath), map_location=device)
        architecture = checkpoint.get("architecture", "efficientnet_b0")
        num_classes = checkpoint.get("num_classes", len(checkpoint.get("classes", DEFAULT_CLASSES)))
        classes = checkpoint.get("classes", DEFAULT_CLASSES[:num_classes])

        model = cls(
            architecture=architecture,
            num_classes=num_classes,
            classes=classes,
            pretrained=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
