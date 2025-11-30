#!/usr/bin/env python3
"""
Comprehensive Model Accuracy Test
Tests all colorization models and measures performance
"""

import numpy as np
import cv2
import os
import time
import sys

def create_test_image():
    """Create a simple test image if none exists"""
    # Create a grayscale test pattern
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        img[:, i] = i
    cv2.imwrite("test_pattern.jpg", img)
    return "test_pattern.jpg"

def test_model(name, func, *args):
    """Test a model function"""
    try:
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        
        if result is not None:
            if isinstance(result, np.ndarray):
                return {
                    "success": True,
                    "time": elapsed,
                    "shape": result.shape,
                    "variance": float(np.var(result)),
                    "mean": float(np.mean(result))
                }
            else:
                return {"success": True, "time": elapsed, "result": str(result)}
        return {"success": False, "error": "Returned None"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def demo_colorize(img):
    """Demo colorization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,1] = np.clip(lab[:,:,1] * 1.3, 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2] * 1.2, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def professional_advanced(img):
    """Professional advanced"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lab_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab_colorized, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,1] = np.clip(lab[:,:,1] * 1.4, 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2] * 1.3, 0, 255)
    result1 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    hsv_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv_colorized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = np.clip(hsv[:,:,0] * 0.8 + 30, 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.5, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)
    result2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    final = cv2.addWeighted(result1, 0.4, result2, 0.3, 0)
    return final

def ai_colorize(img):
    """AI colorization"""
    from cv2 import dnn
    
    proto_file = "models/colorization_deploy_v2.prototxt"
    model_file = "models/colorization_release_v2.caffemodel"
    hull_pts = "models/pts_in_hull.npy"
    
    if not all([os.path.exists(f) for f in [proto_file, model_file, hull_pts]]):
        raise FileNotFoundError("Model files not found")
    
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)
    
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
    return (255 * colorized).astype("uint8")

def main():
    print("=" * 70)
    print("MODEL ACCURACY TEST SUITE")
    print("=" * 70)
    print()
    
    # Get or create test image
    test_images = ["shashwat.jpg", "test_grayscale.jpg"]
    img_path = None
    img = None
    
    for path in test_images:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img_path = path
                break
    
    if img is None:
        print("Creating test image...")
        img_path = create_test_image()
        img = cv2.imread(img_path)
    
    if img is None:
        print("Error: Could not load or create test image")
        return
    
    print(f"Test Image: {img_path}")
    print(f"Image Shape: {img.shape}")
    print()
    
    results = {}
    
    # Test 1: Demo Colorization
    print("[1] Testing Demo Colorization...")
    results["demo"] = test_model("Demo", demo_colorize, img)
    if results["demo"]["success"]:
        print(f"  ✓ Success - Time: {results['demo']['time']:.3f}s")
        print(f"    Output Shape: {results['demo']['shape']}")
        print(f"    Color Variance: {results['demo']['variance']:.2f}")
    else:
        print(f"  ✗ Failed: {results['demo'].get('error', 'Unknown error')}")
    
    # Test 2: Professional Advanced
    print("\n[2] Testing Professional Advanced...")
    results["professional"] = test_model("Professional", professional_advanced, img)
    if results["professional"]["success"]:
        print(f"  ✓ Success - Time: {results['professional']['time']:.3f}s")
        print(f"    Output Shape: {results['professional']['shape']}")
        print(f"    Color Variance: {results['professional']['variance']:.2f}")
    else:
        print(f"  ✗ Failed: {results['professional'].get('error', 'Unknown error')}")
    
    # Test 3: AI Colorization
    print("\n[3] Testing AI Colorization...")
    try:
        results["ai"] = test_model("AI", ai_colorize, img)
        if results["ai"]["success"]:
            print(f"  ✓ Success - Time: {results['ai']['time']:.3f}s")
            print(f"    Output Shape: {results['ai']['shape']}")
            print(f"    Color Variance: {results['ai']['variance']:.2f}")
        else:
            print(f"  ✗ Failed: {results['ai'].get('error', 'Unknown error')}")
    except Exception as e:
        print(f"  ✗ Model files not available: {e}")
        results["ai"] = {"success": False, "error": str(e)}
    
    # Test 4: GradCAM
    print("\n[4] Testing GradCAM Explainability...")
    try:
        from explainability import apply_gradcam
        from cv2 import dnn
        
        proto_file = "models/colorization_deploy_v2.prototxt"
        model_file = "models/colorization_release_v2.caffemodel"
        hull_pts = "models/pts_in_hull.npy"
        
        if all([os.path.exists(f) for f in [proto_file, model_file, hull_pts]]):
            net = dnn.readNetFromCaffe(proto_file, model_file)
            kernel = np.load(hull_pts)
            
            start = time.time()
            heatmap, overlay = apply_gradcam(net, kernel, img)
            elapsed = time.time() - start
            
            if heatmap is not None and overlay is not None:
                print(f"  ✓ Success - Time: {elapsed:.3f}s")
                print(f"    Heatmap Shape: {heatmap.shape}")
                print(f"    Overlay Shape: {overlay.shape}")
                results["gradcam"] = {"success": True, "time": elapsed}
            else:
                print("  ✗ Returned None")
                results["gradcam"] = {"success": False}
        else:
            print("  ✗ Model files not found")
            results["gradcam"] = {"success": False, "error": "Model files not found"}
    except ImportError:
        print("  ✗ explainability module not available")
        results["gradcam"] = {"success": False, "error": "Module not available"}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["gradcam"] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("success", False))
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\nPerformance Metrics:")
    for name, result in results.items():
        if result.get("success"):
            print(f"  {name.capitalize()}: {result.get('time', 0):.3f}s")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

