# One-Click Image Colorizer

A comprehensive one-click solution that generates all colorized versions of your black and white photos, including xAI integration and explainability models.

## 🚀 Quick Start

### Windows
```bash
run_one_click.bat "path/to/your/image.jpg"
```

### Linux/Mac
```bash
chmod +x run_one_click.sh
./run_one_click.sh "path/to/your/image.jpg"
```

### Python Direct
```bash
python one_click_colorizer.py "path/to/your/image.jpg"
```

## 📦 Installation

### Automatic Installation
Run the installer:
```bash
python install_all_packages.py
```

### Manual Installation
```bash
pip install -r requirements.txt
```

## 🎨 Generated Outputs

The app generates **ALL** of these colorized versions in one click:

1. **demo_colorized_output.jpg** - Basic demo colorization
2. **professional_advanced_demo.jpg** - Advanced multi-algorithm colorization
3. **professional_combined.jpg** - Combined result from all methods
4. **professional_gradient-based.jpg** - Gradient-based colorization
5. **ai_colorized_output.jpg** - AI-powered colorization (if model files available)

## ✨ Features

### ✅ All Colorization Methods
- Demo colorization
- Professional advanced demo
- Professional gradient-based
- Professional combined
- AI colorization (if model available)

### ✅ xAI Integration
- AI-powered colorization suggestions
- Enhanced workflow with context-aware recommendations

### ✅ Explainability Models
- **GradCAM** - Always available, shows model attention
- **SHAP** - Feature importance explanations (if installed)
- **LIME** - Local interpretability (if installed)

## 📋 Requirements

### Core Packages
- `opencv-python>=4.5.0`
- `numpy>=1.19.0`
- `matplotlib>=3.5.0`

### Optional Packages
- `xai-sdk>=0.1.0` - For xAI integration
- `shap>=0.41.0` - For SHAP explanations
- `lime>=0.2.0.1` - For LIME explanations

## 🎯 Usage Examples

### Example 1: Basic Usage
```bash
python one_click_colorizer.py test_grayscale.jpg
```

### Example 2: Interactive Mode
```bash
python one_click_colorizer.py
# Then enter image path when prompted
```

### Example 3: Windows Batch File
```bash
run_one_click.bat "C:\Users\YourName\Pictures\old_photo.jpg"
```

## 📁 Output Files

All files are saved in the current directory:

- `demo_colorized_output.jpg`
- `professional_advanced_demo.jpg`
- `professional_combined.jpg`
- `professional_gradient-based.jpg`
- `ai_colorized_output.jpg` (if AI model available)
- `explanations/` folder (if explainability enabled)
  - `gradcam_heatmap.jpg`
  - `gradcam_overlay.jpg`
  - `shap_explanation.png` (if SHAP installed)
  - `lime_explanation.jpg` (if LIME installed)

## 🔧 Configuration

### xAI Setup
Set your xAI API key:
```bash
# Windows PowerShell
$env:XAI_API_KEY="your-api-key-here"

# Linux/Mac
export XAI_API_KEY="your-api-key-here"
```

### AI Model Files (Optional)
For best results, download AI model files:
1. Download from: https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing
2. Place in `models/` directory:
   - `colorization_deploy_v2.prototxt`
   - `colorization_release_v2.caffemodel`
   - `pts_in_hull.npy`

## 🎨 What Each Method Does

1. **Demo Colorized**: Basic LAB color space enhancement
2. **Professional Advanced**: Multi-algorithm approach combining LAB, HSV, and bilateral filtering
3. **Professional Gradient-Based**: Uses image gradients to determine colors
4. **Professional Combined**: Average of all available methods for best result
5. **AI Colorized**: Deep learning model colorization (requires model files)

## 💡 Tips

- **Best Quality**: Use AI model files for the best results
- **Fast Processing**: Demo modes work without model files
- **Multiple Versions**: Compare all outputs to choose your favorite
- **Explainability**: Understand how the AI makes decisions with GradCAM/SHAP/LIME

## 🐛 Troubleshooting

### "Package not found"
```bash
python install_all_packages.py
```

### "Image not found"
- Use full path to image
- Check file extension (.jpg, .png, etc.)
- Use quotes if path has spaces

### "Model files not found"
- App will still work in demo mode
- Download model files for AI colorization

## 📝 Notes

- All outputs are saved as JPG files
- Processing time depends on image size
- AI colorization requires model files (~300MB download)
- Explainability tools may take 1-5 minutes for large images

## 🎉 Enjoy!

Your one-click colorization solution is ready! Just run the script with your image path and get all colorized versions instantly.

