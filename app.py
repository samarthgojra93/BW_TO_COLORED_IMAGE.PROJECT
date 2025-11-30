import numpy as np
import cv2
from cv2 import dnn
import os

# Optional xAI integration
try:
    from xai_integration import initialize_xai, enhance_with_xai
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False

# Optional Explainability integration (GradCAM, SHAP, LIME)
try:
    from explainability import (
        generate_explanations,
        apply_gradcam,
        apply_shap_explainer,
        apply_lime_explainer,
        check_explainability_availability,
        ColorizationModelWrapper
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

# -------- Model file paths -------- #
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
hull_pts = "models/pts_in_hull.npy"

# -------- Chatbot style -------- #
def chatbot_print(msg):
    print(f"[Chatbot]: {msg}")

def user_input(msg):
    return input(f"[You]: {msg}")

def validate_model_files():
    """Check if model files exist and are valid (not empty/corrupted)"""
    files_to_check = {
        proto_file: (1 * 1024, "prototxt"),  # At least 1 KB
        model_file: (200 * 1024 * 1024, "caffemodel"),  # At least 200 MB
        hull_pts: (1 * 1024, "npy")  # At least 1 KB
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

def demo_mode(img_path):
    """Demo mode that simulates colorization without actual model files"""
    chatbot_print("Running in DEMO MODE (model files not available)")
    chatbot_print("This will simulate the colorization process...")
    chatbot_print("Note: For much better results, download real AI model files!")
    
    # Read the input image
    img = cv2.imread(img_path)
    if img is None:
        chatbot_print("ERROR: Could not read the image file.")
        return
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Simulate colorization by creating a simple color version
    chatbot_print("Simulating colorization process...")
    
    # Create a simple colorized version (this is just for demo)
    # In real mode, this would use the deep learning model
    colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply better color enhancement for demo
    # Convert to LAB color space for better color manipulation
    lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
    
    # Enhance the color channels
    lab[:,:,1] = lab[:,:,1] * 1.3  # Increase a channel (green-red)
    lab[:,:,2] = lab[:,:,2] * 1.2  # Increase b channel (blue-yellow)
    
    # Convert back to BGR
    colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply some sharpening to reduce blur
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    colorized = cv2.filter2D(colorized, -1, kernel * 0.1 + np.eye(3) * 0.7)
    
    # Save output
    output_path = "demo_colorized_output.jpg"
    cv2.imwrite(output_path, colorized)
    
    chatbot_print("Demo colorization completed!")
    chatbot_print(f"Saved demo output as: {output_path}")
    chatbot_print("")
    chatbot_print("FOR MUCH BETTER RESULTS:")
    chatbot_print("1. Download real AI model files from:")
    chatbot_print("   https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing")
    chatbot_print("2. Place them in the 'models' folder")
    chatbot_print("3. Run the app again for real AI colorization!")
    
    return output_path

def main():
    chatbot_print("Hi! I'm your Image Colorization Assistant")
    chatbot_print("I can convert your black & white images into color.")

    # Optional: Initialize xAI if available
    xai_client = None
    if XAI_AVAILABLE:
        use_xai = user_input("Would you like to use xAI for enhanced colorization suggestions? (y/n): ")
        if use_xai.lower() == "y":
            xai_client = initialize_xai()
            if xai_client:
                chatbot_print("xAI enabled! You'll get AI-powered colorization insights.")

    # Check if model files exist and are valid
    model_files_exist = validate_model_files()
    
    if not model_files_exist:
        chatbot_print("WARNING: Model files not found or invalid - running in DEMO MODE")
        chatbot_print("The model files appear to be empty or corrupted.")
        chatbot_print("For full AI colorization, download real model files from:")
        chatbot_print("https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing")
        chatbot_print("")
        chatbot_print("You can also run 'python check_models.py' to check model file status.")
        chatbot_print("")

    # Ask for input image
    img_path = user_input("Please enter the path of your black & white image: ")

    # Fix path (allow forward/backslashes)
    img_path = img_path.strip().strip('"').replace("\\", "/")

    if not os.path.exists(img_path):
        chatbot_print("ERROR: File not found. Please check your path.")
        chatbot_print(f"You entered: {img_path}")
        chatbot_print("Make sure to:")
        chatbot_print("1. Use the full path to the image file")
        chatbot_print("2. Include the file extension (.jpg, .png, etc.)")
        chatbot_print("3. Use quotes if path contains spaces")
        chatbot_print("Example: test_grayscale.jpg")
        return
    
    # Check if it's a directory instead of a file
    if os.path.isdir(img_path):
        chatbot_print("ERROR: You provided a directory path, not a file path.")
        chatbot_print(f"You entered: {img_path}")
        chatbot_print("Please provide the full path to an image file.")
        return

    # Run in demo mode if model files don't exist
    if not model_files_exist:
        demo_mode(img_path)
        return

    # Full mode with actual model files
    try:
        chatbot_print("Loading pre-trained colorization model...")
        chatbot_print("This may take a moment...")
        net = dnn.readNetFromCaffe(proto_file, model_file)
        kernel = np.load(hull_pts)
        chatbot_print("Model loaded successfully!")
    except Exception as e:
        chatbot_print(f"ERROR loading model: {str(e)}")
        chatbot_print("")
        chatbot_print("The model files may be corrupted or incomplete.")
        chatbot_print("Expected file sizes:")
        chatbot_print("  - colorization_release_v2.caffemodel: ~200-300 MB")
        chatbot_print("  - colorization_deploy_v2.prototxt: ~1-10 KB")
        chatbot_print("  - pts_in_hull.npy: ~1-100 KB")
        chatbot_print("")
        chatbot_print("Falling back to demo mode...")
        chatbot_print("")
        demo_mode(img_path)
        return

    # Read and preprocess image
    chatbot_print("Reading input image...")
    img = cv2.imread(img_path)
    
    if img is None:
        chatbot_print("ERROR: Could not read the image file.")
        chatbot_print("Please ensure the file is a valid image format (jpg, png, bmp, etc.)")
        return
    
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Add cluster centers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Resize for network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    try:
        chatbot_print("Running colorization model...")
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

        # Recombine channels
        L = cv2.split(lab_img)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save output
        output_path = "colorized_output.jpg"
        success = cv2.imwrite(output_path, colorized)
        
        if not success:
            chatbot_print("ERROR: Failed to save the colorized image.")
            return
            
    except Exception as e:
        chatbot_print(f"ERROR during colorization: {str(e)}")
        return

    chatbot_print("Done! Your image has been colorized successfully.")
    chatbot_print(f"Saved output as: {output_path}")

    # Optional: Get xAI suggestions if enabled
    if xai_client:
        enhance_with_xai(xai_client, img_path)

    # Optional: Generate explainability visualizations (only if model is loaded)
    if EXPLAINABILITY_AVAILABLE and model_files_exist:
        avail_tools = check_explainability_availability()
        available_tools_list = [tool for tool, avail in avail_tools.items() if avail]
        
        if available_tools_list:
            chatbot_print("")
            chatbot_print("Explainability tools available:")
            for tool in available_tools_list:
                chatbot_print(f"  - {tool.upper()}")
            
            use_explain = user_input("Would you like to generate model explanations? (y/n): ")
            if use_explain.lower() == "y":
                chatbot_print("")
                chatbot_print("Generating explanations...")
                chatbot_print("This may take a few moments (especially SHAP and LIME)...")
                chatbot_print("")
                
                try:
                    explanations = generate_explanations(net, kernel, img_path)
                    
                    if explanations:
                        chatbot_print("")
                        chatbot_print("✓ Explanations generated successfully!")
                        chatbot_print("")
                        chatbot_print("Saved explanations:")
                        for name, path in explanations.items():
                            chatbot_print(f"  • {name}: {path}")
                        chatbot_print("")
                        chatbot_print("These visualizations show which parts of the image")
                        chatbot_print("the model focuses on when making colorization decisions.")
                    else:
                        chatbot_print("")
                        chatbot_print("⚠ No explanations were generated.")
                        chatbot_print("Check the error messages above for details.")
                        
                except Exception as e:
                    chatbot_print(f"")
                    chatbot_print(f"⚠ Error generating explanations: {str(e)}")
                    chatbot_print("GradCAM should still work even if SHAP/LIME fail.")
        else:
            chatbot_print("")
            chatbot_print("Note: Explainability tools not fully available.")
            chatbot_print("Install dependencies with: pip install shap lime matplotlib")
            chatbot_print("(GradCAM works without additional dependencies)")
    elif EXPLAINABILITY_AVAILABLE and not model_files_exist:
        chatbot_print("")
        chatbot_print("Note: Explainability requires the AI model to be loaded.")
        chatbot_print("Explanations are only available in full AI mode (not demo mode).")
    elif not EXPLAINABILITY_AVAILABLE:
        chatbot_print("")
        chatbot_print("Note: Explainability module not available.")
        chatbot_print("To enable GradCAM, SHAP, and LIME explanations:")
        chatbot_print("  pip install shap lime matplotlib")

    # Optional: show images side by side
    show = user_input("Do you want to preview the result? (y/n): ")
    if show.lower() == "y":
        img_resized = cv2.resize(img, (400, 400))
        colorized_resized = cv2.resize(colorized, (400, 400))
        result = cv2.hconcat([img_resized, colorized_resized])
        cv2.imshow("Grayscale -> Colour", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        chatbot_print("Okay, preview skipped")

if __name__ == "__main__":
    main()