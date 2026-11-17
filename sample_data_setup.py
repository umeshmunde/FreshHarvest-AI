"""
Dataset setup and sample image generation script for FreshHarvest AI.
Provides:
1. Setup instructions for standard Kaggle Fresh & Rotten Fruits datasets.
2. Generates initial demo sample images in `static/samples/` and `data/dataset/`
   adhering to the `F_<Fruit>` and `S_<Fruit>` folder structure for instant out-of-the-box testing.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from config import DATA_DIR, SAMPLES_DIR, DEFAULT_CLASSES, FRUIT_TYPES

# Color profiles for realistic fruit generation
FRUIT_COLORS = {
    "Apple": {"fresh": (34, 139, 34), "fresh_alt": (220, 20, 60), "spoiled": (92, 64, 51)},
    "Banana": {"fresh": (255, 225, 53), "fresh_alt": (230, 200, 30), "spoiled": (60, 40, 20)},
    "Orange": {"fresh": (255, 140, 0), "fresh_alt": (255, 165, 0), "spoiled": (70, 60, 45)},
    "Mango": {"fresh": (255, 185, 15), "fresh_alt": (255, 120, 30), "spoiled": (75, 50, 30)},
    "Strawberry": {"fresh": (220, 20, 60), "fresh_alt": (255, 40, 80), "spoiled": (65, 45, 40)},
    "Tomato": {"fresh": (235, 40, 40), "fresh_alt": (255, 60, 50), "spoiled": (80, 50, 40)},
    "Lemon": {"fresh": (255, 245, 50), "fresh_alt": (240, 230, 40), "spoiled": (70, 65, 35)}
}

def create_synthetic_fruit_image(fruit_type, is_fresh=True, size=(224, 224), seed=None):
    """
    Synthesize an image with realistic fruit geometry, texture, and decay patterns
    for testing dataset pipeline and Grad-CAM activation.
    """
    if seed is not None:
        np.random.seed(seed)

    w, h = size
    # Background: clean gradient
    bg = np.ones((h, w, 3), dtype=np.uint8) * np.random.randint(235, 248)

    # Base canvas
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    colors = FRUIT_COLORS.get(fruit_type, FRUIT_COLORS["Apple"])
    base_color = colors["fresh"] if is_fresh else colors["spoiled"]
    alt_color = colors["fresh_alt"] if is_fresh else (50, 35, 25)

    # Fruit geometry
    cx, cy = w // 2, h // 2
    rx = int(w * np.random.uniform(0.30, 0.38))
    ry = int(h * np.random.uniform(0.32, 0.40))

    if fruit_type == "Banana":
        # Crescent banana shape
        rx, ry = int(w * 0.42), int(h * 0.22)
    elif fruit_type == "Strawberry":
        # Conical / strawberry shape
        rx, ry = int(w * 0.32), int(h * 0.36)

    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=base_color, outline=(40, 40, 40))

    # Convert to NumPy for texture & decay spots
    arr = np.array(img).astype(np.float32)

    # Lighting highlight
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(((x_coords - (cx - rx * 0.3)) / rx)**2 + ((y_coords - (cy - ry * 0.3)) / ry)**2)
    highlight_mask = (dist_from_center < 0.6).astype(np.float32)
    arr += highlight_mask[:, :, None] * 35.0

    if not is_fresh:
        # Add decay / fungal spots and browning lesions
        num_spots = np.random.randint(5, 12)
        for _ in range(num_spots):
            sx = int(cx + np.random.uniform(-rx * 0.7, rx * 0.7))
            sy = int(cy + np.random.uniform(-ry * 0.7, ry * 0.7))
            sr = np.random.randint(8, 26)
            spot_dist = np.sqrt((x_coords - sx)**2 + (y_coords - sy)**2)
            spot_mask = np.clip(1.0 - (spot_dist / sr), 0, 1)

            # Dark necrotic decay color
            arr[:, :, 0] = arr[:, :, 0] * (1.0 - spot_mask * 0.8) + (40 * spot_mask * 0.8)
            arr[:, :, 1] = arr[:, :, 1] * (1.0 - spot_mask * 0.8) + (25 * spot_mask * 0.8)
            arr[:, :, 2] = arr[:, :, 2] * (1.0 - spot_mask * 0.8) + (15 * spot_mask * 0.8)

    # Add gentle noise/texture
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Stem / leaf for fresh fruit
    final_img = Image.fromarray(arr)
    draw_final = ImageDraw.Draw(final_img)
    if is_fresh:
        stem_color = (101, 67, 33)
        draw_final.line([cx, cy - ry, cx + 5, cy - ry - 14], fill=stem_color, width=4)
        leaf_color = (46, 139, 87)
        draw_final.polygon([(cx + 5, cy - ry - 10), (cx + 20, cy - ry - 16), (cx + 12, cy - ry - 6)], fill=leaf_color)

    return final_img

def setup_demo_environment():
    """
    Generate structured dataset and web demo samples:
    Folders created:
    data/dataset/F_<Fruit>
    data/dataset/S_<Fruit>
    """
    print("\n--- Generating Demo Dataset & Static Web Samples ---")
    data_dir = Path(DATA_DIR)
    samples_dir = Path(SAMPLES_DIR)

    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    fruits = ["Apple", "Banana", "Orange", "Mango", "Strawberry", "Tomato", "Lemon"]

    sample_manifest = []

    for fruit in fruits:
        # 1. Dataset folders for training/testing
        fresh_folder = data_dir / f"F_{fruit}"
        spoiled_folder = data_dir / f"S_{fruit}"
        fresh_folder.mkdir(parents=True, exist_ok=True)
        spoiled_folder.mkdir(parents=True, exist_ok=True)

        # Generate 15 images per class for immediate training verification
        for i in range(15):
            img_fresh = create_synthetic_fruit_image(fruit, is_fresh=True, seed=1000 + i)
            img_fresh.save(fresh_folder / f"{fruit}_fresh_{i+1:03d}.jpg", quality=90)

            img_spoiled = create_synthetic_fruit_image(fruit, is_fresh=False, seed=2000 + i)
            img_spoiled.save(spoiled_folder / f"{fruit}_spoiled_{i+1:03d}.jpg", quality=90)

        # 2. Static Web UI Samples (for 1-click test buttons in Flask app)
        sample_fresh_path = samples_dir / f"sample_fresh_{fruit.lower()}.jpg"
        sample_spoiled_path = samples_dir / f"sample_spoiled_{fruit.lower()}.jpg"

        create_synthetic_fruit_image(fruit, is_fresh=True, seed=42).save(sample_fresh_path, quality=92)
        create_synthetic_fruit_image(fruit, is_fresh=False, seed=84).save(sample_spoiled_path, quality=92)

        sample_manifest.append({
            "fruit": fruit,
            "fresh_image": f"samples/sample_fresh_{fruit.lower()}.jpg",
            "spoiled_image": f"samples/sample_spoiled_{fruit.lower()}.jpg"
        })

    print(f"Generated demo dataset at {data_dir} with {len(fruits)*2} classes.")
    print(f"Generated UI interactive samples in {samples_dir}.")
    return sample_manifest

if __name__ == "__main__":
    setup_demo_environment()

