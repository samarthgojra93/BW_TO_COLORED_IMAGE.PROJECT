# Black-and-White-to-Colour-Image

<img width="1280" height="720" alt="SQL_Thumbnail (35)" src="https://github.com/user-attachments/assets/5765d777-122c-4416-8011-58daf62cf079" />

Bring your old black-and-white photos to life using Artificial Intelligence (AI) and Deep Learning! 🧠🎨

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shashwatnarwal9/black-and-white-project.git
   cd black-and-white-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up xAI (for enhanced AI suggestions)**
   - The xai-sdk package will be installed automatically with requirements.txt
   - Get your xAI API key from https://x.ai
   - Set it as an environment variable:
     ```bash
     # Windows (PowerShell)
     $env:XAI_API_KEY="your-api-key-here"
     
     # Linux/Mac
     export XAI_API_KEY="your-api-key-here"
     ```
   - Or pass it directly when using xAI features

4. **Download model files**
   - Download the required model files from: https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing
   - Place the following files in the `models/` directory:
     - `colorization_deploy_v2.prototxt`
     - `colorization_release_v2.caffemodel`
     - `pts_in_hull.npy`

5. **Run the application**
   ```bash
   python app.py
   ```

### Usage
1. Run the script
2. Enter the path to your black and white image when prompted
3. The application will process the image and save the colorized result as `colorized_output.jpg`
4. Optionally preview the before/after comparison

## 🤖 xAI Integration

This project now includes optional xAI (Grok API) integration for enhanced AI-powered colorization suggestions:

- **AI-Powered Insights**: Get intelligent suggestions for colorization based on image analysis
- **Enhanced Workflow**: Use xAI to provide context-aware color recommendations
- **Optional Feature**: xAI integration is completely optional and can be enabled during runtime

To use xAI features:
1. Install the xai package (included in requirements.txt)
2. Set your XAI_API_KEY environment variable
3. Enable xAI when prompted during app execution

See `xai_integration.py` for detailed implementation and usage examples.

## 🔍 Model Explainability (GradCAM, SHAP, LIME)

This project includes comprehensive explainability tools to understand how the AI model makes colorization decisions:

### Available Tools

1. **GradCAM** (Gradient-weighted Class Activation Mapping)
   - Always available - no additional dependencies
   - Shows which regions the model focuses on for colorization
   - Generates heatmaps and overlay visualizations

2. **SHAP** (SHapley Additive exPlanations)
   - Explains model predictions using game theory
   - Shows feature importance for colorization decisions
   - Requires: `pip install shap`

3. **LIME** (Local Interpretable Model-agnostic Explanations)
   - Explains individual predictions by approximating the model locally
   - Highlights important regions for colorization
   - Requires: `pip install lime`

### Usage

1. Run the main application:
   ```bash
   python app.py
   ```

2. After colorization, you'll be prompted to generate explanations:
   ```
   [Chatbot]: Explainability tools available:
     - GRADCAM
     - SHAP
     - LIME
   [You]: Would you like to generate model explanations? (y/n): y
   ```

3. Explanations will be saved in the `explanations/` directory:
   - `gradcam_heatmap.jpg` - Raw heatmap visualization
   - `gradcam_overlay.jpg` - Heatmap overlaid on original image
   - `shap_explanation.png` - SHAP feature importance visualization
   - `lime_explanation.jpg` - LIME region importance visualization

### Testing Explainability

Test the explainability features:
```bash
python test_explainability.py
```

This will test all available explainability tools on a sample image.

### Installation

Install explainability dependencies:
```bash
pip install shap lime matplotlib
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

**Note**: GradCAM works immediately without additional dependencies. SHAP and LIME require their respective packages to be installed.

See `EXPLAINABILITY_GUIDE.md` for detailed documentation.

## 🎨 Professional Features

This project includes multiple colorization methods:

- **Basic Demo Mode**: Simple color enhancement
- **Professional Mode**: Advanced algorithms with multiple techniques
- **Real AI Mode**: Deep learning colorization (when model files are downloaded)

### Professional Colorization
```bash
python professional_colorization.py
```

This uses advanced algorithms including:
- LAB color space enhancement
- HSV color enhancement
- Edge-preserving bilateral filtering
- Gradient-based colorization
- Combined results for best quality

## 📁 Project Structure

```
black-and-white-project/
├── app.py                           # Main application
├── professional_colorization.py     # Advanced colorization
├── explainability.py                # Explainability module (GradCAM, SHAP, LIME)
├── test_explainability.py           # Test script for explainability
├── requirements.txt                 # Dependencies
├── models/                          # AI model files
├── explanations/                    # Generated explainability visualizations
├── professional_*.jpg              # Generated results
├── EXPLAINABILITY_GUIDE.md          # Explainability documentation
└── README.md                        # This file
```

## 🎯 Results

The application generates multiple high-quality colorized versions:
- `professional_advanced_demo.jpg` - Advanced multi-algorithm result
- `professional_gradient-based.jpg` - Gradient-based colorization
- `professional_combined.jpg` - Combined result (recommended)

## 💡 Ever wondered if AI could restore and colorize vintage photos automatically?

In this project, I built an AI-Powered Image Colorization System using Python, OpenCV, and Deep Learning — a complete end-to-end Computer Vision project that transforms grayscale images into realistic, vibrant color photos.

### What you'll learn:
- Use OpenCV's DNN module for real-world AI applications
- Work with LAB color models in Python
- Load and apply pre-trained deep learning models
- Build an interactive chatbot-style terminal project
- Create an AI that intelligently colorizes historical images

## 📌 Technologies Used:
Python 🐍 | OpenCV | NumPy | Deep Learning | Caffe Model | xAI (Grok API) | GradCAM | SHAP | LIME

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download model files (optional, for AI mode)
4. Run: `python app.py` or `python professional_colorization.py`
5. Enjoy professional-quality colorized images!

---

**Created by Shashwat Narwal** - Bringing old photos back to life with AI! 🎨✨