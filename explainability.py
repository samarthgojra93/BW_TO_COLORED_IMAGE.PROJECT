"""
Explainability module for image colorization model
Integrates GradCAM, SHAP, and LIME for model interpretability
"""

import numpy as np
import cv2
from cv2 import dnn
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Optional imports for explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ColorizationModelWrapper:
    """Wrapper class to make OpenCV DNN model compatible with explainability tools"""
    
    def __init__(self, net, kernel, target_size=(224, 224)):
        self.net = net
        self.kernel = kernel
        self.target_size = target_size
        
        # Setup model layers
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict colorization for a single image
        Args:
            image: Input image (grayscale or BGR)
        Returns:
            Colorized image in BGR format
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Preprocess
        scaled = gray.astype("float32") / 255.0
        lab_img = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        # Resize for network
        resized = cv2.resize(lab_img, self.target_size)
        L = cv2.split(resized)[0]
        L -= 50
        
        # Forward pass
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_channel = cv2.resize(ab_channel, (gray.shape[1], gray.shape[0]))
        
        # Recombine channels
        L_orig = cv2.split(lab_img)[0]
        colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_channel), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        return colorized
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Predict for a batch of images"""
        return [self.predict(img) for img in images]


def apply_gradcam(net, kernel, image: np.ndarray, layer_name: str = "conv8_313_rh") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply GradCAM-like visualization using activation maps
    Since OpenCV DNN doesn't expose gradients, we use activation maps from intermediate layers
    
    Args:
        net: OpenCV DNN network
        kernel: Color cluster centers
        image: Input grayscale image
        layer_name: Layer to extract activations from
        
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Preprocess image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    scaled = gray.astype("float32") / 255.0
    lab_img = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    
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
    
    # Forward pass to get activations
    try:
        # Set input
        blob = cv2.dnn.blobFromImage(L)
        net.setInput(blob)
        
        # Forward pass
        output = net.forward()
        
        # Get activation map (sum across channels for visualization)
        # Since we can't easily get intermediate activations with OpenCV DNN,
        # we'll use the output ab channel as a proxy
        ab_channel = output[0, :, :, :].transpose((1, 2, 0))
        
        # Create heatmap from ab channel magnitude
        heatmap = np.sqrt(ab_channel[:, :, 0]**2 + ab_channel[:, :, 1]**2)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype("uint8")
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()
        
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        return heatmap, overlay
        
    except Exception as e:
        print(f"Error in GradCAM: {str(e)}")
        # Return empty heatmap on error
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        overlay = image.copy()
        return heatmap, overlay


def apply_shap_explainer(model_wrapper: ColorizationModelWrapper, image: np.ndarray, 
                        n_samples: int = 50, output_path: Optional[str] = None) -> np.ndarray:
    """
    Apply SHAP explainer to colorization model
    
    Args:
        model_wrapper: Wrapped colorization model
        image: Input grayscale image
        n_samples: Number of samples for SHAP
        output_path: Optional path to save explanation
        
    Returns:
        SHAP explanation visualization
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return None
    
    try:
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for faster computation
        small_size = (64, 64)  # Smaller for faster SHAP computation
        gray_small = cv2.resize(gray, small_size)
        
        # Define a prediction function that returns a single value for SHAP
        def predict_fn(images):
            """Predict and return colorization intensity"""
            predictions = []
            for img in images:
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.astype(np.uint8)
                
                # Get colorized result
                colorized = model_wrapper.predict(gray_img)
                # Return average color intensity as a single value
                intensity = np.mean(colorized.astype(float))
                predictions.append(intensity)
            return np.array(predictions)
        
        # Create masker for image
        try:
            masker = shap.maskers.Image("inpaint_telea", gray_small.shape)
        except:
            # Fallback to blur masker
            masker = shap.maskers.Image("blur(128,128)", gray_small.shape)
        
        # Create explainer
        explainer = shap.Explainer(predict_fn, masker)
        
        # Prepare input (convert to RGB for SHAP)
        gray_rgb = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2RGB)
        
        # Explain
        shap_values = explainer(np.expand_dims(gray_rgb, axis=0), max_evals=n_samples)
        
        # Visualize
        if output_path:
            plt.figure(figsize=(10, 5))
            shap.image_plot(shap_values, show=False)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            return shap_values
        else:
            shap.image_plot(shap_values, show=False)
            return shap_values
        
    except Exception as e:
        print(f"Error in SHAP explainer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def apply_lime_explainer(model_wrapper: ColorizationModelWrapper, image: np.ndarray,
                         num_features: int = 100, num_samples: int = 1000,
                         output_path: Optional[str] = None) -> np.ndarray:
    """
    Apply LIME explainer to colorization model
    
    Args:
        model_wrapper: Wrapped colorization model
        image: Input grayscale image
        num_features: Number of features to explain
        num_samples: Number of samples for LIME
        output_path: Optional path to save explanation
        
    Returns:
        LIME explanation visualization
    """
    if not LIME_AVAILABLE:
        print("LIME not available. Install with: pip install lime")
        return None
    
    try:
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for faster computation
        small_size = (64, 64)
        gray_small = cv2.resize(gray, small_size)
        
        # Convert to RGB for LIME (LIME expects RGB)
        gray_rgb = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2RGB)
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Define prediction function for LIME
        def predict_fn(images):
            """Convert RGB to grayscale and predict"""
            predictions = []
            for img in images:
                # Convert RGB to grayscale
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.astype(np.uint8)
                
                # Predict
                pred = model_wrapper.predict(gray_img)
                # Return average color intensity as a single value
                intensity = np.mean(pred.astype(float))
                predictions.append(intensity)
            return np.array(predictions)
        
        # Explain
        explanation = explainer.explain_instance(
            gray_rgb.astype(np.uint8),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get explanation image
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        # Resize back to original size
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, 
                                 (image.shape[1], image.shape[0]))
        
        # Create visualization
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()
        
        # Overlay mask
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(image_bgr, 0.6, mask_colored, 0.4, 0)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
        
        return overlay
        
    except Exception as e:
        print(f"Error in LIME explainer: {str(e)}")
        return None


def generate_explanations(net, kernel, image_path: str, output_dir: str = "explanations") -> dict:
    """
    Generate all explanations (GradCAM, SHAP, LIME) for an image
    
    Args:
        net: OpenCV DNN network
        kernel: Color cluster centers
        image_path: Path to input image
        output_dir: Directory to save explanations
        
    Returns:
        Dictionary with paths to saved explanation images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return {}
    
    results = {}
    
    # Create model wrapper
    model_wrapper = ColorizationModelWrapper(net, kernel)
    
    # 1. GradCAM
    print("Generating GradCAM explanation...")
    try:
        heatmap, overlay = apply_gradcam(net, kernel, image)
        gradcam_path = os.path.join(output_dir, "gradcam_heatmap.jpg")
        overlay_path = os.path.join(output_dir, "gradcam_overlay.jpg")
        cv2.imwrite(gradcam_path, heatmap)
        cv2.imwrite(overlay_path, overlay)
        results['gradcam_heatmap'] = gradcam_path
        results['gradcam_overlay'] = overlay_path
        print(f"GradCAM saved to {overlay_path}")
    except Exception as e:
        print(f"GradCAM error: {str(e)}")
    
    # 2. SHAP
    if SHAP_AVAILABLE:
        print("Generating SHAP explanation...")
        try:
            shap_path = os.path.join(output_dir, "shap_explanation.png")
            apply_shap_explainer(model_wrapper, image, output_path=shap_path)
            results['shap'] = shap_path
            print(f"SHAP explanation saved to {shap_path}")
        except Exception as e:
            print(f"SHAP error: {str(e)}")
    else:
        print("SHAP not available. Install with: pip install shap")
    
    # 3. LIME
    if LIME_AVAILABLE:
        print("Generating LIME explanation...")
        try:
            lime_path = os.path.join(output_dir, "lime_explanation.jpg")
            lime_overlay = apply_lime_explainer(model_wrapper, image, output_path=lime_path)
            results['lime'] = lime_path
            print(f"LIME explanation saved to {lime_path}")
        except Exception as e:
            print(f"LIME error: {str(e)}")
    else:
        print("LIME not available. Install with: pip install lime")
    
    return results


def check_explainability_availability() -> dict:
    """Check which explainability tools are available"""
    return {
        'gradcam': True,  # Always available (custom implementation)
        'shap': SHAP_AVAILABLE,
        'lime': LIME_AVAILABLE
    }

