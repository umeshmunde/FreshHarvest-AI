"""
Automatic Kaggle Dataset Downloader for FreshHarvest AI.
Downloads and unzips the full 13,500+ fruits dataset into data/dataset.
Dataset: sriramr/fruits-fresh-and-rotten-for-classification
"""

import os
import sys
from pathlib import Path

# Set user-provided Kaggle credentials
os.environ['KAGGLE_USERNAME'] = "umeshmunde"
os.environ['KAGGLE_KEY'] = "58d1d4269e91491a48b6aed3a8293d9b"

def download_dataset():
    print("=" * 70)
    print("  FreshHarvest AI - Full Kaggle Dataset Downloader (13,500+ Images)")
    print("=" * 70)
    print("Dataset: sriramr/fruits-fresh-and-rotten-for-classification")
    print("Target : ./data/dataset")
    print("-" * 70)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[!] Kaggle package not installed. Installing kaggle...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
        print("[+] Kaggle authentication successful (User: umeshmunde)")
    except Exception as e:
        print(f"[!] Authentication failed: {e}")
        return

    dest_dir = Path(__file__).resolve().parent / "data" / "dataset"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading dataset archive (approx ~2.1 GB)... This may take 2-5 minutes depending on your internet connection.")
    try:
        api.dataset_download_files(
            "sriramr/fruits-fresh-and-rotten-for-classification",
            path=str(dest_dir),
            unzip=True,
            quiet=False
        )
        print("\n[+] Extraction complete!")
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return

    # Verify count
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_imgs = sum(1 for p in dest_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts)
    print("-" * 70)
    print(f"[SUCCESS] Dataset successfully updated! Total images found: {total_imgs}")
    print("=" * 70)

if __name__ == "__main__":
    download_dataset()

