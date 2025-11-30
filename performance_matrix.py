#!/usr/bin/env python3
"""
Comprehensive Performance Matrix for Image Colorization Project
Measures processing time, accuracy, quality, and resource usage
"""

import numpy as np
import cv2
import os
import time
import sys
from typing import Dict, List, Tuple
from cv2 import dnn

# Optional psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    else:
        # Fallback: return 0 if psutil not available
        return 0.0

def calculate_image_metrics(img: np.ndarray) -> Dict:
    """Calculate image quality metrics"""
    metrics = {}
    try:
        metrics["mean_intensity"] = float(np.mean(img))
        metrics["std_intensity"] = float(np.std(img))
        metrics["variance"] = float(np.var(img))
        metrics["min_value"] = int(np.min(img))
        metrics["max_value"] = int(np.max(img))
        
        if len(img.shape) == 3:
            # Color metrics
            metrics["color_channels"] = img.shape[2]
            metrics["color_variance"] = float(np.var(img))
            # Calculate color diversity (entropy-like measure)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist / hist.sum()
            metrics["color_diversity"] = float(-np.sum(hist * np.log(hist + 1e-10)))
        else:
            metrics["color_channels"] = 1
            metrics["color_variance"] = 0.0
            metrics["color_diversity"] = 0.0
        
        metrics["resolution"] = f"{img.shape[1]}x{img.shape[0]}"
        metrics["total_pixels"] = int(img.shape[0] * img.shape[1])
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def calculate_similarity(original: np.ndarray, colorized: np.ndarray) -> Dict:
    """Calculate similarity metrics between original and colorized"""
    metrics = {}
    try:
        # Ensure same size
        if original.shape[:2] != colorized.shape[:2]:
            colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
        
        # Convert to grayscale for structure comparison
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
        
        if len(colorized.shape) == 3:
            color_gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)
        else:
            color_gray = colorized
        
        # MSE (Mean Squared Error)
        mse = np.mean((orig_gray.astype(float) - color_gray.astype(float)) ** 2)
        metrics["mse"] = float(mse)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            metrics["psnr"] = float(20 * np.log10(255.0 / np.sqrt(mse)))
        else:
            metrics["psnr"] = float('inf')
        
        # SSIM (Simplified Structural Similarity)
        metrics["ssim"] = calculate_ssim(orig_gray, color_gray)
        
        # Color enhancement metric (how much color was added)
        if len(colorized.shape) == 3:
            color_enhancement = np.var(colorized) - np.var(orig_gray)
            metrics["color_enhancement"] = float(color_enhancement)
        else:
            metrics["color_enhancement"] = 0.0
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images"""
    try:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim = numerator / denominator
        return float(ssim)
    except:
        return 0.0

def test_model_performance(name: str, func, *args) -> Dict:
    """Test model performance with comprehensive metrics"""
    result = {
        "name": name,
        "success": False,
        "processing_time": 0,
        "memory_usage": 0,
        "input_metrics": {},
        "output_metrics": {},
        "similarity_metrics": {},
        "error": None
    }
    
    try:
        # Get input image
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            input_img = args[0]
            result["input_metrics"] = calculate_image_metrics(input_img)
        
        # Measure memory before
        mem_before = get_memory_usage()
        
        # Run model
        start_time = time.time()
        output = func(*args)
        elapsed = time.time() - start_time
        
        # Measure memory after
        mem_after = get_memory_usage()
        
        if output is not None and isinstance(output, np.ndarray):
            result["success"] = True
            result["processing_time"] = elapsed
            result["memory_usage"] = mem_after - mem_before
            result["output_metrics"] = calculate_image_metrics(output)
            
            # Calculate similarity if we have input
            if len(args) > 0 and isinstance(args[0], np.ndarray):
                result["similarity_metrics"] = calculate_similarity(args[0], output)
        else:
            result["error"] = "Output is None or invalid"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result

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

def professional_gradient(img):
    """Professional gradient-based"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    direction_normalized = ((direction + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    magnitude_normalized = np.clip(magnitude * 2, 0, 255).astype(np.uint8)
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    hsv[:,:,0] = direction_normalized
    hsv[:,:,1] = magnitude_normalized
    hsv[:,:,2] = gray
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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

def print_performance_matrix(results: List[Dict]):
    """Print formatted performance matrix"""
    print("\n" + "=" * 100)
    print("PERFORMANCE MATRIX - IMAGE COLORIZATION MODELS")
    print("=" * 100)
    print()
    
    # Header
    print(f"{'Model':<25} {'Status':<10} {'Time (s)':<12} {'Memory (MB)':<15} {'Color Var':<12} {'SSIM':<10} {'PSNR':<10}")
    print("-" * 100)
    
    # Data rows
    for result in results:
        name = result["name"][:24]
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        time_str = f"{result['processing_time']:.4f}" if result["success"] else "N/A"
        mem_str = f"{result['memory_usage']:.2f}" if result["success"] else "N/A"
        
        if result["success"]:
            color_var = f"{result['output_metrics'].get('color_variance', 0):.2f}"
            ssim = f"{result['similarity_metrics'].get('ssim', 0):.3f}"
            psnr = f"{result['similarity_metrics'].get('psnr', 0):.2f}"
        else:
            color_var = "N/A"
            ssim = "N/A"
            psnr = "N/A"
        
        print(f"{name:<25} {status:<10} {time_str:<12} {mem_str:<15} {color_var:<12} {ssim:<10} {psnr:<10}")
    
    print("\n" + "=" * 100)
    print("\nDETAILED METRICS")
    print("=" * 100)
    
    for result in results:
        if result["success"]:
            print(f"\n{result['name']}:")
            print(f"  Processing Time: {result['processing_time']:.4f} seconds")
            print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
            print(f"  Output Resolution: {result['output_metrics'].get('resolution', 'N/A')}")
            print(f"  Color Variance: {result['output_metrics'].get('color_variance', 0):.2f}")
            print(f"  Color Diversity: {result['output_metrics'].get('color_diversity', 0):.3f}")
            print(f"  SSIM: {result['similarity_metrics'].get('ssim', 0):.3f}")
            print(f"  PSNR: {result['similarity_metrics'].get('psnr', 0):.2f} dB")
            print(f"  MSE: {result['similarity_metrics'].get('mse', 0):.2f}")
        else:
            print(f"\n{result['name']}:")
            print(f"  Status: FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")

def generate_performance_report(results: List[Dict], output_file: str = "PERFORMANCE_MATRIX_REPORT.md"):
    """Generate markdown performance report"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 Performance Matrix Report\n\n")
        f.write("## Executive Summary\n\n")
        
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        f.write(f"- **Total Models Tested**: {total}\n")
        f.write(f"- **Models Working**: {passed}\n")
        f.write(f"- **Success Rate**: {(passed/total)*100:.1f}%\n\n")
        
        f.write("## Performance Matrix\n\n")
        f.write("| Model | Status | Time (s) | Memory (MB) | Color Variance | SSIM | PSNR (dB) |\n")
        f.write("|-------|--------|----------|-------------|----------------|------|-----------|\n")
        
        for result in results:
            name = result["name"]
            status = "✅" if result["success"] else "❌"
            time_str = f"{result['processing_time']:.4f}" if result["success"] else "N/A"
            mem_str = f"{result['memory_usage']:.2f}" if result["success"] else "N/A"
            
            if result["success"]:
                color_var = f"{result['output_metrics'].get('color_variance', 0):.2f}"
                ssim = f"{result['similarity_metrics'].get('ssim', 0):.3f}"
                psnr = f"{result['similarity_metrics'].get('psnr', 0):.2f}"
            else:
                color_var = "N/A"
                ssim = "N/A"
                psnr = "N/A"
            
            f.write(f"| {name} | {status} | {time_str} | {mem_str} | {color_var} | {ssim} | {psnr} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for result in results:
            f.write(f"### {result['name']}\n\n")
            if result["success"]:
                f.write(f"- **Status**: ✅ Working\n")
                f.write(f"- **Processing Time**: {result['processing_time']:.4f} seconds\n")
                f.write(f"- **Memory Usage**: {result['memory_usage']:.2f} MB\n")
                f.write(f"- **Output Resolution**: {result['output_metrics'].get('resolution', 'N/A')}\n")
                f.write(f"- **Color Variance**: {result['output_metrics'].get('color_variance', 0):.2f}\n")
                f.write(f"- **Color Diversity**: {result['output_metrics'].get('color_diversity', 0):.3f}\n")
                f.write(f"- **SSIM (Structural Similarity)**: {result['similarity_metrics'].get('ssim', 0):.3f}\n")
                f.write(f"- **PSNR (Peak Signal-to-Noise Ratio)**: {result['similarity_metrics'].get('psnr', 0):.2f} dB\n")
                f.write(f"- **MSE (Mean Squared Error)**: {result['similarity_metrics'].get('mse', 0):.2f}\n")
            else:
                f.write(f"- **Status**: ❌ Failed\n")
                f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("## Performance Rankings\n\n")
        
        # Fastest
        working = [r for r in results if r["success"]]
        if working:
            fastest = min(working, key=lambda x: x["processing_time"])
            f.write(f"### ⚡ Fastest Model\n")
            f.write(f"- **{fastest['name']}**: {fastest['processing_time']:.4f} seconds\n\n")
            
            # Best color diversity
            best_color = max(working, key=lambda x: x["output_metrics"].get("color_diversity", 0))
            f.write(f"### 🎨 Best Color Diversity\n")
            f.write(f"- **{best_color['name']}**: {best_color['output_metrics'].get('color_diversity', 0):.3f}\n\n")
            
            # Best SSIM
            best_ssim = max(working, key=lambda x: x["similarity_metrics"].get("ssim", 0))
            f.write(f"### 📐 Best Structural Similarity (SSIM)\n")
            f.write(f"- **{best_ssim['name']}**: {best_ssim['similarity_metrics'].get('ssim', 0):.3f}\n\n")

def main():
    """Main function"""
    print("=" * 100)
    print("PERFORMANCE MATRIX GENERATOR")
    print("=" * 100)
    print()
    
    # Find test image
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
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            img[:, i] = [i, i, i]
        img_path = "test_pattern.jpg"
        cv2.imwrite(img_path, img)
    
    print(f"Test Image: {img_path}")
    print(f"Image Shape: {img.shape}")
    print()
    
    results = []
    
    # Test all models
    print("Testing models...")
    print()
    
    # 1. Demo Colorization
    print("[1/4] Testing Demo Colorization...")
    results.append(test_model_performance("Demo Colorization", demo_colorize, img))
    
    # 2. Professional Advanced
    print("[2/4] Testing Professional Advanced...")
    results.append(test_model_performance("Professional Advanced", professional_advanced, img))
    
    # 3. Professional Gradient
    print("[3/4] Testing Professional Gradient...")
    results.append(test_model_performance("Professional Gradient", professional_gradient, img))
    
    # 4. AI Colorization
    print("[4/4] Testing AI Colorization...")
    try:
        results.append(test_model_performance("AI Colorization", ai_colorize, img))
    except Exception as e:
        results.append({
            "name": "AI Colorization",
            "success": False,
            "error": str(e)
        })
    
    # Print matrix
    print_performance_matrix(results)
    
    # Generate report
    print("\nGenerating performance report...")
    generate_performance_report(results)
    print("✓ Report saved to: PERFORMANCE_MATRIX_REPORT.md")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    if not PSUTIL_AVAILABLE:
        print("Note: psutil not available - memory metrics will be 0")
        print("Install with: pip install psutil (optional)")
        print()
    main()

