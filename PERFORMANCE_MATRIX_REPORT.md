# 📊 Performance Matrix Report

## Executive Summary

- **Total Models Tested**: 4
- **Models Working**: 3
- **Success Rate**: 75.0%

## Performance Matrix

| Model | Status | Time (s) | Memory (MB) | Color Variance | SSIM | PSNR (dB) |
|-------|--------|----------|-------------|----------------|------|-----------|
| Demo Colorization | ✅ | 0.3691 | 0.00 | 6596.25 | 0.995 | 29.61 |
| Professional Advanced | ✅ | 0.0162 | 0.00 | 2969.99 | 0.871 | 14.91 |
| Professional Gradient | ✅ | 0.0095 | 0.00 | 5262.49 | 1.000 | 39.02 |
| AI Colorization | ❌ | N/A | N/A | N/A | N/A | N/A |

## Detailed Metrics

### Demo Colorization

- **Status**: ✅ Working
- **Processing Time**: 0.3691 seconds
- **Memory Usage**: 0.00 MB
- **Output Resolution**: 256x256
- **Color Variance**: 6596.25
- **Color Diversity**: 2.802
- **SSIM (Structural Similarity)**: 0.995
- **PSNR (Peak Signal-to-Noise Ratio)**: 29.61 dB
- **MSE (Mean Squared Error)**: 71.21

### Professional Advanced

- **Status**: ✅ Working
- **Processing Time**: 0.0162 seconds
- **Memory Usage**: 0.00 MB
- **Output Resolution**: 256x256
- **Color Variance**: 2969.99
- **Color Diversity**: 2.484
- **SSIM (Structural Similarity)**: 0.871
- **PSNR (Peak Signal-to-Noise Ratio)**: 14.91 dB
- **MSE (Mean Squared Error)**: 2100.32

### Professional Gradient

- **Status**: ✅ Working
- **Processing Time**: 0.0095 seconds
- **Memory Usage**: 0.00 MB
- **Output Resolution**: 256x256
- **Color Variance**: 5262.49
- **Color Diversity**: 2.564
- **SSIM (Structural Similarity)**: 1.000
- **PSNR (Peak Signal-to-Noise Ratio)**: 39.02 dB
- **MSE (Mean Squared Error)**: 8.15

### AI Colorization

- **Status**: ❌ Failed
- **Error**: OpenCV(4.12.0) D:\a\opencv-python\opencv-python\opencv\modules\dnn\src\caffe\caffe_io.cpp:1176: error: (-2:Unspecified error) FAILED: ReadProtoFromBinaryFile(param_file, param). Failed to parse NetParameter file: models/colorization_release_v2.caffemodel in function 'cv::dnn::ReadNetParamsFromBinaryFileOrDie'


## Performance Rankings

### ⚡ Fastest Model
- **Professional Gradient**: 0.0095 seconds

### 🎨 Best Color Diversity
- **Demo Colorization**: 2.802

### 📐 Best Structural Similarity (SSIM)
- **Professional Gradient**: 1.000

