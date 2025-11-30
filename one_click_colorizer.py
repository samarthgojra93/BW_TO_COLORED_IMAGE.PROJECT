#!/usr/bin/env python3
"""
One-Click Image Colorization App
Generates all colorized versions: demo, professional advanced, combined, and gradient-based
Includes xAI integration and explainability models (GradCAM, SHAP, LIME)
"""

import numpy as np
import cv2
from cv2 import dnn
import os
import sys
from typing import Optional, Tuple

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception:
            pass  # Silently fail if .env can't be read

# Load .env at startup
load_env_file()

# Optional xAI integration
try:
    from xai_integration import initialize_xai, enhance_with_xai
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False

# Optional Explainability integration
try:
    from explainability import (
        generate_explanations,
        check_explainability_availability,
        ColorizationModelWrapper
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

# Model file paths
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
hull_pts = "models/pts_in_hull.npy"

def print_status(msg, status="info"):
    """Print status messages with formatting"""
    symbols = {
        "info": "ℹ",
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "processing": "⟳"
    }
    symbol = symbols.get(status, "•")
    print(f"[{symbol}] {msg}")

def validate_model_files():
    """Check if model files exist and are valid"""
    files_to_check = {
        proto_file: (1 * 1024, "prototxt"),
        model_file: (200 * 1024 * 1024, "caffemodel"),
        hull_pts: (1 * 1024, "npy")
    }
    
    all_valid = True
    for filepath, (min_size, file_type) in files_to_check.items():
        if not os.path.exists(filepath):
            all_valid = False
            break
        file_size = os.path.getsize(filepath)
        if file_size < min_size:
            all_valid = False
            break
    
    return all_valid

def demo_colorization(img_path: str) -> Optional[np.ndarray]:
    """Generate demo colorized version"""
    print_status("Generating demo colorized version...", "processing")
    
    img = cv2.imread(img_path)
    if img is None:
        print_status("Could not read image file", "error")
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Create colorized version
    colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply LAB color space enhancement
    lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
    lab[:,:,1] = lab[:,:,1] * 1.3  # Increase a channel
    lab[:,:,2] = lab[:,:,2] * 1.2  # Increase b channel
    colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    colorized = cv2.filter2D(colorized, -1, kernel * 0.1 + np.eye(3) * 0.7)
    
    return colorized

def professional_advanced_demo(img_path: str) -> Optional[np.ndarray]:
    """Generate professional advanced demo version"""
    print_status("Generating professional advanced demo...", "processing")
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Method 1A: LAB Color Space Enhancement
    lab_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab_colorized, cv2.COLOR_BGR2LAB)
    lab[:,:,1] = lab[:,:,1] * 1.4
    lab[:,:,2] = lab[:,:,2] * 1.3
    result1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Method 1B: HSV Enhancement
    hsv_colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv_colorized, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,0] * 0.8 + 30
    hsv[:,:,1] = hsv[:,:,1] * 1.5
    hsv[:,:,2] = hsv[:,:,2] * 1.1
    result2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Method 1C: Edge-Preserving Colorization
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    colorized_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    lab_bilateral = cv2.cvtColor(colorized_bilateral, cv2.COLOR_BGR2LAB)
    lab_bilateral[:,:,1] = lab_bilateral[:,:,1] * 1.2
    lab_bilateral[:,:,2] = lab_bilateral[:,:,2] * 1.1
    result3 = cv2.cvtColor(lab_bilateral, cv2.COLOR_LAB2BGR)
    
    # Combine results
    final_result = cv2.addWeighted(result1, 0.4, result2, 0.3, 0)
    final_result = cv2.addWeighted(final_result, 0.7, result3, 0.3, 0)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(final_result, -1, kernel * 0.1 + np.eye(3) * 0.9)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    return denoised

def professional_gradient_based(img_path: str) -> Optional[np.ndarray]:
    """Generate professional gradient-based version"""
    print_status("Generating professional gradient-based version...", "processing")
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Normalize direction to 0-180 for HSV
    direction_normalized = ((direction + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    magnitude_normalized = np.clip(magnitude * 2, 0, 255).astype(np.uint8)
    
    # Create HSV image (vectorized)
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    hsv[:,:,0] = direction_normalized  # Hue from gradient direction
    hsv[:,:,1] = magnitude_normalized   # Saturation from gradient magnitude
    hsv[:,:,2] = gray                   # Value from original image
    
    # Convert HSV to BGR
    colorized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return colorized

def ai_colorization(img_path: str, net, kernel) -> Optional[np.ndarray]:
    """Generate AI colorized version using the deep learning model"""
    print_status("Generating AI colorized version...", "processing")
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Setup model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # Resize for network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Forward pass
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    # Recombine channels
    L_orig = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

def generate_all_colorizations(img_path: str, use_ai: bool = False, net=None, kernel=None) -> dict:
    """Generate all colorized versions"""
    results = {}
    
    # 1. Demo colorized
    demo_result = demo_colorization(img_path)
    if demo_result is not None:
        cv2.imwrite("demo_colorized_output.jpg", demo_result)
        results['demo_colorized'] = "demo_colorized_output.jpg"
        print_status("Demo colorized saved", "success")
    
    # 2. Professional advanced demo
    advanced_result = professional_advanced_demo(img_path)
    if advanced_result is not None:
        cv2.imwrite("professional_advanced_demo.jpg", advanced_result)
        results['professional_advanced'] = "professional_advanced_demo.jpg"
        print_status("Professional advanced demo saved", "success")
    
    # 3. Professional gradient-based
    gradient_result = professional_gradient_based(img_path)
    if gradient_result is not None:
        cv2.imwrite("professional_gradient-based.jpg", gradient_result)
        results['professional_gradient'] = "professional_gradient-based.jpg"
        print_status("Professional gradient-based saved", "success")
    
    # 4. AI colorized (if model available)
    if use_ai and net is not None and kernel is not None:
        ai_result = ai_colorization(img_path, net, kernel)
        if ai_result is not None:
            cv2.imwrite("ai_colorized_output.jpg", ai_result)
            results['ai_colorized'] = "ai_colorized_output.jpg"
            print_status("AI colorized saved", "success")
    
    # 5. Professional combined (average of all available results)
    available_results = [r for r in [demo_result, advanced_result, gradient_result] if r is not None]
    if len(available_results) > 1:
        print_status("Generating professional combined version...", "processing")
        combined = available_results[0].astype(np.float32)
        for result in available_results[1:]:
            # Resize to match if needed
            if result.shape != combined.shape:
                result = cv2.resize(result, (combined.shape[1], combined.shape[0]))
            combined += result.astype(np.float32)
        combined = combined / len(available_results)
        combined = combined.astype(np.uint8)
        cv2.imwrite("professional_combined.jpg", combined)
        results['professional_combined'] = "professional_combined.jpg"
        print_status("Professional combined saved", "success")
    
    return results

def main():
    """Main function - one-click colorization"""
    print("=" * 60)
    print("ONE-CLICK IMAGE COLORIZER")
    print("=" * 60)
    print()
    
    # Get image path from command line or prompt
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip().strip('"').replace("\\", "/")
    else:
        img_path = input("Enter path to black & white image: ").strip().strip('"').replace("\\", "/")
    
    # Validate image path
    if not os.path.exists(img_path):
        print_status(f"Image not found: {img_path}", "error")
        return
    
    if os.path.isdir(img_path):
        print_status("Path is a directory, not a file", "error")
        return
    
    print_status(f"Processing image: {img_path}", "info")
    print()
    
    # Check for AI model
    model_files_exist = validate_model_files()
    net = None
    kernel = None
    
    if model_files_exist:
        try:
            print_status("Loading AI model...", "processing")
            net = dnn.readNetFromCaffe(proto_file, model_file)
            kernel = np.load(hull_pts)
            print_status("AI model loaded successfully", "success")
        except Exception as e:
            print_status(f"Could not load AI model: {e}", "warning")
            print_status("Continuing with demo modes only", "info")
    
    # Generate all colorizations
    print()
    print("=" * 60)
    print("GENERATING ALL COLORIZED VERSIONS")
    print("=" * 60)
    print()
    
    results = generate_all_colorizations(img_path, use_ai=model_files_exist, net=net, kernel=kernel)
    
    print()
    print("=" * 60)
    print("COLORIZATION COMPLETE")
    print("=" * 60)
    print()
    print_status("Generated files:", "success")
    for name, path in results.items():
        print(f"  • {name}: {path}")
    print()
    
    # xAI integration
    if XAI_AVAILABLE:
        try:
            print_status("Initializing xAI for enhanced suggestions...", "processing")
            xai_client = initialize_xai()
            if xai_client:
                enhance_with_xai(xai_client, img_path)
                print_status("xAI suggestions generated", "success")
        except Exception as e:
            print_status(f"xAI integration skipped: {e}", "warning")
    
    # Explainability models
    if EXPLAINABILITY_AVAILABLE and model_files_exist and net is not None and kernel is not None:
        try:
            avail_tools = check_explainability_availability()
            available_tools = [tool for tool, avail in avail_tools.items() if avail]
            
            if available_tools:
                print()
                print_status("Generating explainability visualizations...", "processing")
                print_status(f"Available tools: {', '.join(available_tools).upper()}", "info")
                
                explanations = generate_explanations(net, kernel, img_path)
                
                if explanations:
                    print_status("Explainability visualizations generated", "success")
                    for name, path in explanations.items():
                        print(f"  • {name}: {path}")
        except Exception as e:
            print_status(f"Explainability skipped: {e}", "warning")
    
    print()
    print("=" * 60)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 60)
    print()
    print_status("Check the generated JPG files in the current directory", "info")

if __name__ == "__main__":
    main()

