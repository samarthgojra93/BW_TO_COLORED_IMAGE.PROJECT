# Quick Start: Explainability Features

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `shap` - For SHAP explanations
- `lime` - For LIME explanations  
- `matplotlib` - For visualization
- All other project dependencies

### 2. Run the Application
```bash
python app.py
```

### 3. Use Explainability

1. **Colorize an image** - Follow the normal workflow
2. **When prompted**, choose to generate explanations:
   ```
   [Chatbot]: Explainability tools available:
     - GRADCAM
     - SHAP
     - LIME
   [You]: Would you like to generate model explanations? (y/n): y
   ```

3. **Wait for results** - Explanations will be saved in `explanations/` directory

## 📊 What You Get

### GradCAM (Always Available)
- `gradcam_heatmap.jpg` - Heatmap showing model attention
- `gradcam_overlay.jpg` - Heatmap overlaid on original image

### SHAP (If Installed)
- `shap_explanation.png` - Feature importance visualization

### LIME (If Installed)
- `lime_explanation.jpg` - Region importance visualization

## 🧪 Test the Features

Run the test script:
```bash
python test_explainability.py
```

This will:
- Check which tools are available
- Test GradCAM
- Test Model Wrapper
- Generate all explanations

## ⚠️ Important Notes

1. **GradCAM works immediately** - No extra dependencies needed
2. **SHAP and LIME are optional** - Install with `pip install shap lime`
3. **Model files required** - Explainability only works in full AI mode (not demo mode)
4. **Processing time** - SHAP and LIME can take 1-5 minutes depending on image size

## 🔧 Troubleshooting

### "SHAP not available"
```bash
pip install shap
```

### "LIME not available"
```bash
pip install lime
```

### "Model files not found"
- Download model files from the Google Drive link in README
- Place them in the `models/` directory

### Explanations take too long
- SHAP and LIME are computationally intensive
- They automatically resize images to 32x32 for faster processing
- GradCAM is much faster and always works

## 📖 More Information

See `EXPLAINABILITY_GUIDE.md` for detailed documentation.

