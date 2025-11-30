# Explainability Integration Guide

This project now includes **GradCAM**, **SHAP**, and **LIME** explainability tools to help you understand how the colorization model makes decisions.

## Features

### 1. GradCAM (Gradient-weighted Class Activation Mapping)
- **Always available** - Custom implementation that works with OpenCV DNN
- Shows which regions of the image the model focuses on for colorization
- Generates heatmaps and overlay visualizations

### 2. SHAP (SHapley Additive exPlanations)
- Explains model predictions by showing feature importance
- Uses game theory to attribute predictions to input features
- Requires: `pip install shap`

### 3. LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions by approximating the model locally
- Highlights important regions for colorization decisions
- Requires: `pip install lime`

## Installation

Install the required dependencies:

```bash
pip install shap lime matplotlib
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
   ```bash
   python app.py
   ```

2. After colorization completes, you'll be asked if you want to generate explanations:
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

## Understanding the Visualizations

- **GradCAM**: Red/yellow regions indicate areas where the model is most active for colorization
- **SHAP**: Shows how different image regions contribute to the colorization output
- **LIME**: Highlights the most important regions that influence the colorization decision

## Technical Details

The explainability module (`explainability.py`) includes:
- `ColorizationModelWrapper`: Wraps OpenCV DNN model for compatibility with explainability tools
- `apply_gradcam()`: Generates GradCAM-style heatmaps
- `apply_shap_explainer()`: Applies SHAP explanation
- `apply_lime_explainer()`: Applies LIME explanation
- `generate_explanations()`: Convenience function to generate all explanations at once

## Notes

- GradCAM works immediately (no additional dependencies)
- SHAP and LIME require their respective packages to be installed
- Explanations may take a few moments to generate, especially for larger images
- The tools automatically resize images for faster computation while maintaining accuracy

