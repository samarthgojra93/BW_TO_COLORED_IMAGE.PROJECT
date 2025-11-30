"""
Test script for explainability features
This script tests GradCAM, SHAP, and LIME on a sample image
"""

import cv2
import numpy as np
import os
from cv2 import dnn

# Import explainability module
try:
    from explainability import (
        generate_explanations,
        apply_gradcam,
        check_explainability_availability,
        ColorizationModelWrapper
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Error importing explainability module: {e}")
    EXPLAINABILITY_AVAILABLE = False

# Model file paths
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
hull_pts = "models/pts_in_hull.npy"

def test_explainability():
    """Test explainability features"""
    
    if not EXPLAINABILITY_AVAILABLE:
        print("ERROR: Explainability module not available")
        return False
    
    # Check availability
    print("Checking explainability tool availability...")
    avail = check_explainability_availability()
    for tool, available in avail.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {tool.upper()}: {status}")
    print()
    
    # Check if model files exist
    if not all([os.path.exists(proto_file), os.path.exists(model_file), os.path.exists(hull_pts)]):
        print("ERROR: Model files not found!")
        print("Please ensure model files are in the 'models' directory")
        return False
    
    # Test image path
    test_image = "test_grayscale.jpg"
    if not os.path.exists(test_image):
        print(f"ERROR: Test image '{test_image}' not found")
        print("Please provide a test image path")
        return False
    
    print(f"Loading model and testing with image: {test_image}")
    print()
    
    try:
        # Load model
        print("Loading model...")
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)
        print("✓ Model loaded successfully")
        print()
        
        # Test GradCAM (most reliable)
        print("=" * 50)
        print("Testing GradCAM...")
        print("=" * 50)
        try:
            image = cv2.imread(test_image)
            if image is None:
                print("ERROR: Could not read test image")
                return False
            
            heatmap, overlay = apply_gradcam(net, kernel, image)
            if heatmap is not None and overlay is not None:
                # Save test output
                cv2.imwrite("test_gradcam_heatmap.jpg", heatmap)
                cv2.imwrite("test_gradcam_overlay.jpg", overlay)
                print("✓ GradCAM test passed")
                print("  Saved: test_gradcam_heatmap.jpg")
                print("  Saved: test_gradcam_overlay.jpg")
            else:
                print("✗ GradCAM test failed (returned None)")
        except Exception as e:
            print(f"✗ GradCAM test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
        # Test Model Wrapper
        print("=" * 50)
        print("Testing Model Wrapper...")
        print("=" * 50)
        try:
            wrapper = ColorizationModelWrapper(net, kernel)
            test_img = cv2.imread(test_image)
            if test_img is not None:
                result = wrapper.predict(test_img)
                if result is not None and result.shape == test_img.shape:
                    print("✓ Model Wrapper test passed")
                    cv2.imwrite("test_wrapper_output.jpg", result)
                    print("  Saved: test_wrapper_output.jpg")
                else:
                    print("✗ Model Wrapper test failed (invalid output)")
            else:
                print("✗ Could not read test image")
        except Exception as e:
            print(f"✗ Model Wrapper test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
        # Test full explanation generation
        print("=" * 50)
        print("Testing Full Explanation Generation...")
        print("=" * 50)
        try:
            explanations = generate_explanations(net, kernel, test_image, output_dir="test_explanations")
            if explanations:
                print(f"✓ Generated {len(explanations)} explanation(s)")
                for name, path in explanations.items():
                    print(f"  • {name}: {path}")
            else:
                print("⚠ No explanations generated (some tools may have failed)")
        except Exception as e:
            print(f"✗ Full explanation test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 50)
        print("Test completed!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Explainability Test Script")
    print("=" * 50)
    print()
    success = test_explainability()
    if success:
        print("\n✓ All tests completed (some may have warnings)")
    else:
        print("\n✗ Tests failed")

