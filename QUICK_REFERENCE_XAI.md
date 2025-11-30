# Quick Reference: XAI & Explainability

## 🎯 Quick Answers

### What is XAI?
- **XAI (x.AI)** = Company providing Grok API
- **Purpose**: Provides AI-powered **text suggestions** for colorization
- **NOT**: Image generation (XAI doesn't create images, only gives advice)

### Why XAI is Used?
- ✅ Expert colorization recommendations
- ✅ Context-aware color suggestions
- ✅ Enhanced user experience
- ⚠️ **Optional** - project works without it

### Heatmap Formula (Blue/Red Regions)

**Red Regions** (High Activity):
```
heatmap = √(a² + b²)  [from LAB color space]
normalized = (heatmap - min) / (max - min)
Red = high(normalized) → 170-255 range
```

**Blue Regions** (Low Activity):
```
Blue = low(normalized) → 0-85 range
```

**Formula Steps**:
1. Extract AB channels from model output
2. Calculate magnitude: `√(a² + b²)`
3. Normalize to 0-255
4. Apply JET colormap (blue=low, red=high)

### Accuracy Metrics

| Model | SSIM | PSNR (dB) | MSE | Status |
|-------|------|-----------|-----|--------|
| Professional Gradient | **1.000** | **39.02** | **8.15** | ✅ Best |
| Demo Colorization | 0.995 | 29.61 | 71.21 | ✅ Good |
| Professional Advanced | 0.871 | 14.91 | 2100.32 | ✅ Fast |

**Overall**: 75% success rate (3/4 models working)

### GradCAM, SHAP, LIME Usage

**GradCAM**:
- ✅ Always available
- Shows model attention regions
- Formula: `heatmap = √(a² + b²)`
- Output: Heatmaps (red=active, blue=inactive)

**SHAP**:
- Requires: `pip install shap`
- Uses game theory
- Shows feature importance
- Output: Feature contribution maps

**LIME**:
- Requires: `pip install lime`
- Local explanations
- Shows important regions
- Output: Region masks

### Architecture Design

```
User Input (Grayscale Image)
    ↓
Preprocessing (LAB, resize 224×224)
    ↓
Deep Learning Model (Caffe DNN)
    ├─ Input: L channel
    ├─ Output: AB channels
    └─ 313 color clusters
    ↓
Post-processing (Recombine L+AB)
    ↓
Colorized Image
    ↓
[Optional] XAI Suggestions (text)
    ↓
[Optional] Explainability (GradCAM/SHAP/LIME)
    ↓
Final Output + Visualizations
```

**Key Components**:
- **Models**: Caffe DNN (OpenCV)
- **Color Space**: LAB (Lightness + AB channels)
- **Network**: 224×224 input → 313 color bins → AB output
- **Integration**: XAI (HTTP API), Explainability (Python libraries)

### XAI Images Location

**Note**: XAI doesn't generate images, only provides text suggestions.

**Actual Generated Images**:
- `colorized_output.jpg` - Main colorized result
- `explanations/gradcam_heatmap.jpg` - GradCAM visualization
- `explanations/gradcam_overlay.jpg` - GradCAM overlay
- `explanations/shap_explanation.png` - SHAP visualization
- `explanations/lime_explanation.jpg` - LIME visualization

---

## 📊 Visual Summary

### Heatmap Color Meaning:
- 🔴 **Red/Yellow** = High model activity (important for colorization)
- 🟢 **Green** = Medium activity
- 🔵 **Blue** = Low activity (less important)

### Model Performance:
- 🥇 **Professional Gradient**: Best quality (SSIM=1.0, PSNR=39dB)
- 🥈 **Demo Colorization**: Best color diversity
- 🥉 **Professional Advanced**: Fastest processing

---

**For detailed information, see**: `XAI_AND_EXPLAINABILITY_COMPREHENSIVE_GUIDE.md`

