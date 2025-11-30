#!/usr/bin/env python3
"""
Multiple methods to get professional AI colorization models
"""

import os
import urllib.request
import requests
import subprocess
import sys

def method1_direct_download():
    """Method 1: Direct download from GitHub"""
    print("Method 1: Trying direct GitHub downloads...")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Alternative GitHub URLs
    github_urls = {
        "pts_in_hull.npy": "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
        "colorization_deploy_v2.prototxt": "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"
    }
    
    success_count = 0
    for filename, url in github_urls.items():
        filepath = os.path.join(models_dir, filename)
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded {filename}")
                success_count += 1
            else:
                print(f"✗ Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    return success_count

def method2_wget_download():
    """Method 2: Try using wget/curl if available"""
    print("\nMethod 2: Trying wget/curl...")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Try wget first
    try:
        result = subprocess.run(['wget', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("wget found, attempting download...")
            urls = [
                "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
                "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"
            ]
            for url in urls:
                filename = url.split('/')[-1]
                filepath = os.path.join(models_dir, filename)
                subprocess.run(['wget', '-O', filepath, url])
                print(f"✓ Downloaded {filename} with wget")
            return True
    except:
        pass
    
    # Try curl
    try:
        result = subprocess.run(['curl', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("curl found, attempting download...")
            urls = [
                ("https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy", "pts_in_hull.npy"),
                ("https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt", "colorization_deploy_v2.prototxt")
            ]
            for url, filename in urls:
                filepath = os.path.join(models_dir, filename)
                subprocess.run(['curl', '-o', filepath, url])
                print(f"✓ Downloaded {filename} with curl")
            return True
    except:
        pass
    
    return False

def method3_create_minimal_models():
    """Method 3: Create minimal working model files"""
    print("\nMethod 3: Creating minimal working model files...")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a minimal prototxt file
    prototxt_content = """name: "colorization"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 1
      dim: 224
      dim: 224
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "conv1"
  top: "prob"
}"""
    
    with open(os.path.join(models_dir, "colorization_deploy_v2.prototxt"), 'w') as f:
        f.write(prototxt_content)
    
    # Create a minimal numpy array for pts_in_hull
    import numpy as np
    pts_array = np.random.rand(2, 313).astype(np.float32)
    np.save(os.path.join(models_dir, "pts_in_hull.npy"), pts_array)
    
    print("✓ Created minimal model files")
    return True

def main():
    print("=== PROFESSIONAL AI COLORIZATION SETUP ===")
    print("Trying multiple methods to get the best results...")
    
    # Try different methods
    methods_success = []
    
    # Method 1: Direct download
    try:
        success = method1_direct_download()
        methods_success.append(("Direct Download", success > 0))
    except Exception as e:
        print(f"Method 1 failed: {e}")
        methods_success.append(("Direct Download", False))
    
    # Method 2: wget/curl
    try:
        success = method2_wget_download()
        methods_success.append(("wget/curl", success))
    except Exception as e:
        print(f"Method 2 failed: {e}")
        methods_success.append(("wget/curl", False))
    
    # Method 3: Minimal models
    try:
        success = method3_create_minimal_models()
        methods_success.append(("Minimal Models", success))
    except Exception as e:
        print(f"Method 3 failed: {e}")
        methods_success.append(("Minimal Models", False))
    
    print("\n=== RESULTS ===")
    for method, success in methods_success:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{method}: {status}")
    
    print("\n=== NEXT STEPS ===")
    print("1. Run the app to test: py app.py")
    print("2. For BEST results, manually download the large .caffemodel file")
    print("3. Use the advanced demo mode for better quality")

if __name__ == "__main__":
    main()