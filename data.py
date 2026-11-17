"""
Data loading and preprocessing module for FreshHarvest AI.
Preserves strict compatibility with erickyegon/FreshHarvest (F_ and S_ folder naming)
while also seamlessly supporting Kaggle fresh/rotten fruit datasets.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEFAULT_TRAIN_CONFIG,
    FRUIT_TYPES, DEFAULT_CLASSES
)

def parse_fruit_and_freshness(name):
    """
    Parse a directory name into (freshness, fruit_type, unified_class).
    Supports:
      1. Original FreshHarvest convention: F_Banana -> ('Fresh', 'Banana', 'Fresh_Banana')
                                            S_Banana -> ('Spoiled', 'Banana', 'Spoiled_Banana')
      2. Kaggle dataset convention: freshapples -> ('Fresh', 'Apple', 'Fresh_Apple')
                                    rottenbananas -> ('Spoiled', 'Banana', 'Spoiled_Banana')
      3. Underscore/Hyphen format: fresh_orange -> ('Fresh', 'Orange', 'Fresh_Orange')
    """
    clean = name.strip()

    freshness = None
    fruit = None

    # Format 1: F_<Fruit> or S_<Fruit>
    if clean.startswith("F_") or clean.startswith("f_"):
        freshness = "Fresh"
        fruit = clean[2:].strip()
    elif clean.startswith("S_") or clean.startswith("s_"):
        freshness = "Spoiled"
        fruit = clean[2:].strip()
    # Format 2: fresh* / rotten* / spoiled*
    elif clean.lower().startswith("fresh"):
        freshness = "Fresh"
        fruit = re.sub(r"^fresh[_\-\s]*", "", clean, flags=re.IGNORECASE).strip()
    elif clean.lower().startswith("rotten") or clean.lower().startswith("spoiled"):
        freshness = "Spoiled"
        fruit = re.sub(r"^(rotten|spoiled)[_\-\s]*", "", clean, flags=re.IGNORECASE).strip()

    if fruit:
        # Normalize fruit names (e.g. apples -> Apple, bananas -> Banana)
        fruit_clean = fruit.rstrip("sS").capitalize()
        # Common corrections
        if fruit_clean.lower() == "appl":
            fruit_clean = "Apple"
        elif fruit_clean.lower() == "orange":
            fruit_clean = "Orange"
        elif fruit_clean.lower() == "tomato":
            fruit_clean = "Tomato"
        elif fruit_clean.lower() == "mango":
            fruit_clean = "Mango"
    else:
        fruit_clean = "Unknown"

    if freshness is None:
        freshness = "Unknown"

    unified_class = f"{freshness}_{fruit_clean}"
    return freshness, fruit_clean, unified_class

def get_transforms(is_train=True):
    """Get torchvision image transformation pipeline."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class FruitImageDataset(Dataset):
    """PyTorch Dataset for fruit freshness and type classification."""

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label_idx'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images: create neutral gray canvas
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label, row['fruit_type'], row['freshness']

def scan_dataset(dataset_path):
    """
    Scan a dataset directory recursively and return a DataFrame.
    Gracefully identifies F_*, S_*, fresh*, and rotten* folders.

    Args:
        dataset_path (str or Path): Path to the dataset root folder

    Returns:
        tuple: (pd.DataFrame, list_of_classes)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    records = []

    # Recursive search to support nested Kaggle folders (e.g. data/dataset/train/freshapples)
    candidate_folders = [dataset_path] + [p for p in dataset_path.rglob("*") if p.is_dir()]

    for folder in candidate_folders:
        freshness, fruit, unified_class = parse_fruit_and_freshness(folder.name)
        if freshness == "Unknown" or fruit == "Unknown":
            # Check if parent folder contains fresh/spoiled info (e.g., train/fresh/banana)
            parent_freshness, _, _ = parse_fruit_and_freshness(folder.parent.name)
            if parent_freshness != "Unknown":
                freshness = parent_freshness
                fruit = folder.name.rstrip("sS").capitalize()
                unified_class = f"{freshness}_{fruit}"
            else:
                continue

        for file_path in folder.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                records.append({
                    'image_path': str(file_path),
                    'folder_name': folder.name,
                    'fruit_type': fruit,
                    'freshness': freshness,
                    'class_name': unified_class
                })

    if not records:
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    classes = sorted(df['class_name'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df['label_idx'] = df['class_name'].map(class_to_idx)

    return df, classes

def calculate_class_weights(df, num_classes):
    """Calculate balanced class weights to handle dataset class imbalance."""
    counts = df['label_idx'].value_counts().to_dict()
    total = len(df)
    weights = []
    for i in range(num_classes):
        cnt = counts.get(i, 1)
        w = total / (num_classes * cnt)
        weights.append(w)
    return torch.FloatTensor(weights)

def prepare_dataloaders(dataset_path, batch_size=16, num_workers=0,
                        train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Scan dataset, split into train/val/test splits, and return DataLoaders.

    Returns:
        dict: {
            'train_loader': DataLoader,
            'val_loader': DataLoader,
            'test_loader': DataLoader,
            'classes': list,
            'class_weights': torch.Tensor,
            'stats': dict
        }
    """
    df, classes = scan_dataset(dataset_path)
    if len(df) == 0:
        raise ValueError(f"No valid fruit images found in '{dataset_path}' with F_*/S_* or fresh*/rotten* structure.")

    stratify_col = df['label_idx'] if df['label_idx'].nunique() > 1 else None

    # First split: train vs remaining
    train_df, rem_df = train_test_split(
        df,
        train_size=train_split,
        random_state=42,
        stratify=stratify_col
    )

    # Second split: val vs test
    rem_stratify = rem_df['label_idx'] if rem_df['label_idx'].nunique() > 1 else None
    val_ratio = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        rem_df,
        train_size=val_ratio,
        random_state=42,
        stratify=rem_stratify
    )

    train_ds = FruitImageDataset(train_df, transform=get_transforms(is_train=True))
    val_ds = FruitImageDataset(val_df, transform=get_transforms(is_train=False))
    test_ds = FruitImageDataset(test_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_weights = calculate_class_weights(train_df, len(classes))

    stats = {
        "total_images": len(df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "num_classes": len(classes),
        "classes": classes,
        "class_distribution": df['class_name'].value_counts().to_dict(),
        "freshness_distribution": df['freshness'].value_counts().to_dict()
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_weights": class_weights,
        "stats": stats,
        "test_df": test_df
    }
