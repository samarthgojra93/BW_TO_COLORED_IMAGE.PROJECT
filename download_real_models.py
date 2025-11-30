#!/usr/bin/env python3
"""
Download real AI model files for proper colorization
"""

import os
import urllib.request
import zipfile
import tempfile

def download_real_models():
    print("Downloading real AI model files for proper colorization...")
    print("This will give you much better results than demo mode!")
    print()
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Try to download from alternative sources
    model_urls = {
        "pts_in_hull.npy": "https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy",
        "colorization_deploy_v2.prototxt": "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"
    }
    
    print("Attempting to download model files...")
    
    for filename, url in model_urls.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"✓ {filename} already exists and looks valid")
            continue
            
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    # The large caffemodel file needs manual download
    caffemodel_path = os.path.join(models_dir, "colorization_release_v2.caffemodel")
    if not os.path.exists(caffemodel_path) or os.path.getsize(caffemodel_path) < 1000000:
        print()
        print("IMPORTANT: The main model file (colorization_release_v2.caffemodel) is ~300MB")
        print("Please download it manually from:")
        print("https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing")
        print()
        print("Steps:")
        print("1. Go to the Google Drive link above")
        print("2. Download 'colorization_release_v2.caffemodel'")
        print("3. Place it in the 'models' folder")
        print("4. Run the app again for real AI colorization!")
    
    print()
    print("After downloading all files, your app will use real AI instead of demo mode!")

if __name__ == "__main__":
    download_real_models()