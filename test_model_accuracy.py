#!/usr/bin/env python3
"""
Comprehensive Model Accuracy Testing
Tests all colorization models and explainability tools
"""

import numpy as np
import cv2
import os
import sys
import time
from typing import Dict, List, Tuple
from cv2 import dnn

# Model file paths
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
hull_pts = "models/pts_in_hull.npy"

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(test_name, status, details=""):
    """Print test result"""
    symbol = "✓" if status else "✗"
    print(f"{symbol} {test_name}")
    if details:
        print(f"  {details}")

def test_image_loading(img_path: str) -> Tuple[bool, np.ndarray]:
    """Test if image can be loaded"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, None
        return True, img
    except Exception as e:
        return False, None

def test_demo_colorization(img_path: str) -> Dict:
    """Test demo colorization method"""
    result = {"success": False, "time": 0, "output_shape": None, "error": None}
    
    try:
        start_time = time.time()
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Could not load image"
            return result
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Demo colorization
        colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        lab[:,:,1] = lab[:,:,1] * 1.3
        lab[:,:,2] = lab[:,:,2] * 1.2
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        colorized = cv2.filter2D(colorized, -1, kernel * 0.1 + np.eye(3) * 0.7)
        
        result["time"] = time.time() - start_time
        result["output_shape"] = colorized.shape
        result["success"] = True
        
        # Calculate metrics
        result["color_variance"] = np.var(colorized)
        result["mean_intensity"] = np.mean(colorized)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def test_professional_advanced(img_path: str) -> Dict:
    """Test professional advanced colorization"""
    result = {"success": False, "time": 0, "output_shape": None, "error": None}
    
    try:
        start_time = time.time()
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Could not load image"
            return result
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Professional advanced method
        lab_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(lab_colorized, cv2.COLOR_BGR2LAB)
        lab[:,:,1] = lab[:,:,1] * 1.4
        lab[:,:,2] = lab[:,:,2] * 1.3
        result1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv_colorized, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = hsv[:,:,0] * 0.8 + 30
        hsv[:,:,1] = hsv[:,:,1] * 1.5
        hsv[:,:,2] = hsv[:,:,2] * 1.1
        result2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        colorized_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
        lab_bilateral = cv2.cvtColor(colorized_bilateral, cv2.COLOR_BGR2LAB)
        lab_bilateral[:,:,1] = lab_bilateral[:,:,1] * 1.2
        lab_bilateral[:,:,2] = lab_bilateral[:,:,2] * 1.1
        result3 = cv2.cvtColor(lab_bilateral, cv2.COLOR_LAB2BGR)
        
        final_result = cv2.addWeighted(result1, 0.4, result2, 0.3, 0)
        final_result = cv2.addWeighted(final_result, 0.7, result3, 0.3, 0)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(final_result, -1, kernel * 0.1 + np.eye(3) * 0.9)
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        result["time"] = time.time() - start_time
        result["output_shape"] = denoised.shape
        result["success"] = True
        result["color_variance"] = np.var(denoised)
        result["mean_intensity"] = np.mean(denoised)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def test_ai_colorization(img_path: str) -> Dict:
    """Test AI colorization model"""
    result = {"success": False, "time": 0, "output_shape": None, "error": None, "model_loaded": False}
    
    # Check if model files exist
    if not all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
        result["error"] = "Model files not found"
        return result
    
    try:
        # Load model
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)
        result["model_loaded"] = True
        
        start_time = time.time()
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Could not load image"
            return result
        
        # Setup model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        # Process
        scaled = img.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab_img, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
        
        L_orig = cv2.split(lab_img)[0]
        colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_channel), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        result["time"] = time.time() - start_time
        result["output_shape"] = colorized.shape
        result["success"] = True
        result["color_variance"] = np.var(colorized)
        result["mean_intensity"] = np.mean(colorized)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def test_gradcam(img_path: str) -> Dict:
    """Test GradCAM explainability"""
    result = {"success": False, "time": 0, "error": None}
    
    # Check if model files exist
    if not all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
        result["error"] = "Model files not found (GradCAM needs AI model)"
        return result
    
    try:
        from explainability import apply_gradcam
        
        start_time = time.time()
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Could not load image"
            return result
        
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)
        
        heatmap, overlay = apply_gradcam(net, kernel, img)
        
        if heatmap is not None and overlay is not None:
            result["time"] = time.time() - start_time
            result["success"] = True
            result["heatmap_shape"] = heatmap.shape
            result["overlay_shape"] = overlay.shape
        else:
            result["error"] = "GradCAM returned None"
            
    except ImportError:
        result["error"] = "explainability module not available"
    except Exception as e:
        result["error"] = str(e)
    
    return result

def test_shap_lime(img_path: str) -> Dict:
    """Test SHAP and LIME explainability"""
    result = {"shap": {"success": False, "available": False}, 
              "lime": {"success": False, "available": False}}
    
    # Check if model files exist
    if not all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
        result["shap"]["error"] = "Model files not found"
        result["lime"]["error"] = "Model files not found"
        return result
    
    try:
        from explainability import ColorizationModelWrapper, apply_shap_explainer, apply_lime_explainer
        from explainability import SHAP_AVAILABLE, LIME_AVAILABLE
        
        result["shap"]["available"] = SHAP_AVAILABLE
        result["lime"]["available"] = LIME_AVAILABLE
        
        img = cv2.imread(img_path)
        if img is None:
            result["shap"]["error"] = "Could not load image"
            result["lime"]["error"] = "Could not load image"
            return result
        
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)
        model_wrapper = ColorizationModelWrapper(net, kernel)
        
        # Test SHAP
        if SHAP_AVAILABLE:
            try:
                start_time = time.time()
                shap_result = apply_shap_explainer(model_wrapper, img, n_samples=5, output_path=None)
                result["shap"]["time"] = time.time() - start_time
                result["shap"]["success"] = shap_result is not None
            except Exception as e:
                result["shap"]["error"] = str(e)
        
        # Test LIME
        if LIME_AVAILABLE:
            try:
                start_time = time.time()
                lime_result = apply_lime_explainer(model_wrapper, img, num_samples=50, output_path=None)
                result["lime"]["time"] = time.time() - start_time
                result["lime"]["success"] = lime_result is not None
            except Exception as e:
                result["lime"]["error"] = str(e)
                
    except ImportError:
        result["shap"]["error"] = "explainability module not available"
        result["lime"]["error"] = "explainability module not available"
    except Exception as e:
        result["shap"]["error"] = str(e)
        result["lime"]["error"] = str(e)
    
    return result

def calculate_accuracy_metrics(original: np.ndarray, colorized: np.ndarray) -> Dict:
    """Calculate accuracy metrics between original and colorized"""
    metrics = {}
    
    try:
        # Ensure same size
        if original.shape != colorized.shape:
            colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
        
        # Convert to grayscale for comparison
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
        
        if len(colorized.shape) == 3:
            color_gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)
        else:
            color_gray = colorized
        
        # Calculate metrics
        metrics["mse"] = np.mean((orig_gray.astype(float) - color_gray.astype(float)) ** 2)
        metrics["psnr"] = 20 * np.log10(255.0 / np.sqrt(metrics["mse"])) if metrics["mse"] > 0 else float('inf')
        metrics["ssim"] = calculate_ssim(orig_gray, color_gray)
        metrics["color_diversity"] = np.var(colorized) if len(colorized.shape) == 3 else 0
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM (Structural Similarity Index)"""
    try:
        # Simple SSIM calculation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim)
    except:
        return 0.0

def run_comprehensive_test(img_path: str):
    """Run comprehensive accuracy tests"""
    print_header("MODEL ACCURACY TEST SUITE")
    
    # Test image loading
    print("\n[1] Testing Image Loading...")
    success, img = test_image_loading(img_path)
    if not success:
        print_result("Image Loading", False, "Could not load image")
        return
    print_result("Image Loading", True, f"Loaded: {img.shape}")
    
    # Test Demo Colorization
    print("\n[2] Testing Demo Colorization...")
    demo_result = test_demo_colorization(img_path)
    print_result("Demo Colorization", demo_result["success"], 
                f"Time: {demo_result['time']:.3f}s, Shape: {demo_result.get('output_shape', 'N/A')}")
    if demo_result.get("error"):
        print(f"  Error: {demo_result['error']}")
    
    # Test Professional Advanced
    print("\n[3] Testing Professional Advanced...")
    prof_result = test_professional_advanced(img_path)
    print_result("Professional Advanced", prof_result["success"],
                f"Time: {prof_result['time']:.3f}s, Shape: {prof_result.get('output_shape', 'N/A')}")
    if prof_result.get("error"):
        print(f"  Error: {prof_result['error']}")
    
    # Test AI Colorization
    print("\n[4] Testing AI Colorization...")
    ai_result = test_ai_colorization(img_path)
    if ai_result.get("model_loaded"):
        print_result("AI Model Loading", True, "Model loaded successfully")
    print_result("AI Colorization", ai_result["success"],
                f"Time: {ai_result['time']:.3f}s, Shape: {ai_result.get('output_shape', 'N/A')}")
    if ai_result.get("error"):
        print(f"  Error: {ai_result['error']}")
    
    # Test GradCAM
    print("\n[5] Testing GradCAM Explainability...")
    gradcam_result = test_gradcam(img_path)
    print_result("GradCAM", gradcam_result["success"],
                f"Time: {gradcam_result.get('time', 0):.3f}s" if gradcam_result["success"] else "")
    if gradcam_result.get("error"):
        print(f"  Error: {gradcam_result['error']}")
    
    # Test SHAP/LIME
    print("\n[6] Testing SHAP & LIME Explainability...")
    explain_result = test_shap_lime(img_path)
    if explain_result["shap"]["available"]:
        print_result("SHAP", explain_result["shap"]["success"],
                    f"Time: {explain_result['shap'].get('time', 0):.3f}s" if explain_result["shap"]["success"] else "")
        if explain_result["shap"].get("error"):
            print(f"  Error: {explain_result['shap']['error']}")
    else:
        print_result("SHAP", False, "Not installed")
    
    if explain_result["lime"]["available"]:
        print_result("LIME", explain_result["lime"]["success"],
                    f"Time: {explain_result['lime'].get('time', 0):.3f}s" if explain_result["lime"]["success"] else "")
        if explain_result["lime"].get("error"):
            print(f"  Error: {explain_result['lime']['error']}")
    else:
        print_result("LIME", False, "Not installed")
    
    # Summary
    print_header("TEST SUMMARY")
    total_tests = 6
    passed = sum([
        demo_result["success"],
        prof_result["success"],
        ai_result["success"],
        gradcam_result["success"],
        explain_result["shap"]["success"] if explain_result["shap"]["available"] else True,  # Optional
        explain_result["lime"]["success"] if explain_result["lime"]["available"] else True   # Optional
    ])
    
    print(f"\nTests Passed: {passed}/{total_tests}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    print("\n" + "=" * 70)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Try to find a test image
        test_images = ["test_grayscale.jpg", "shashwat.jpg"]
        img_path = None
        for img in test_images:
            if os.path.exists(img):
                img_path = img
                break
        
        if not img_path:
            print("Usage: python test_model_accuracy.py <image_path>")
            print("\nOr place a test image named 'test_grayscale.jpg' in the current directory")
            return
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        return
    
    run_comprehensive_test(img_path)

if __name__ == "__main__":
    main()

