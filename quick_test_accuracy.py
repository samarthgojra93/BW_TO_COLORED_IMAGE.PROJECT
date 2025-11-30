#!/usr/bin/env python3
"""Quick accuracy test - simplified version"""

import numpy as np
import cv2
import os
import time

def quick_test():
    """Quick test of all models"""
    print("=" * 70)
    print("QUICK MODEL ACCURACY TEST")
    print("=" * 70)
    
    # Find test image
    test_images = ["shashwat.jpg", "test_grayscale.jpg"]
    img_path = None
    for img in test_images:
        if os.path.exists(img):
            # Verify it's a valid image
            test_img = cv2.imread(img)
            if test_img is not None:
                img_path = img
                break
    
    if not img_path:
        print("No test image found. Please provide an image path.")
        return
    
    print(f"\nTesting with: {img_path}\n")
    
    # Test 1: Image loading
    print("[1] Image Loading...")
    img = cv2.imread(img_path)
    if img is None:
        print("✗ Failed to load image")
        return
    print(f"✓ Loaded: {img.shape}")
    
    # Test 2: Demo colorization
    print("\n[2] Demo Colorization...")
    try:
        start = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        lab[:,:,1] *= 1.3
        lab[:,:,2] *= 1.2
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elapsed = time.time() - start
        print(f"✓ Success ({elapsed:.3f}s) - Output: {result.shape}")
        print(f"  Color variance: {np.var(result):.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Professional Advanced
    print("\n[3] Professional Advanced...")
    try:
        start = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Simplified version
        lab_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(lab_colorized, cv2.COLOR_BGR2LAB)
        lab[:,:,1] *= 1.4
        lab[:,:,2] *= 1.3
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elapsed = time.time() - start
        print(f"✓ Success ({elapsed:.3f}s) - Output: {result.shape}")
        print(f"  Color variance: {np.var(result):.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: AI Model
    print("\n[4] AI Colorization Model...")
    proto_file = "models/colorization_deploy_v2.prototxt"
    model_file = "models/colorization_release_v2.caffemodel"
    hull_pts = "models/pts_in_hull.npy"
    
    if all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
        try:
            from cv2 import dnn
            start = time.time()
            net = dnn.readNetFromCaffe(proto_file, model_file)
            kernel = np.load(hull_pts)
            print("  ✓ Model loaded")
            
            scaled = img.astype("float32") / 255.0
            lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab_img, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = kernel.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            net.setInput(cv2.dnn.blobFromImage(L))
            ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
            
            L_orig = cv2.split(lab_img)[0]
            colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_channel), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            elapsed = time.time() - start
            print(f"✓ Success ({elapsed:.3f}s) - Output: {colorized.shape}")
            print(f"  Color variance: {np.var(colorized):.2f}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("✗ Model files not found")
    
    # Test 5: GradCAM
    print("\n[5] GradCAM Explainability...")
    try:
        from explainability import apply_gradcam
        from cv2 import dnn
        
        if all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
            start = time.time()
            net = dnn.readNetFromCaffe(proto_file, model_file)
            kernel = np.load(hull_pts)
            heatmap, overlay = apply_gradcam(net, kernel, img)
            elapsed = time.time() - start
            if heatmap is not None:
                print(f"✓ Success ({elapsed:.3f}s)")
            else:
                print("✗ Returned None")
        else:
            print("✗ Model files not found")
    except ImportError:
        print("✗ explainability module not available")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    quick_test()

