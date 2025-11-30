# Model Files Required

This application requires the following model files to be downloaded and placed in this directory:

1. **colorization_deploy_v2.prototxt** - Network architecture file
2. **colorization_release_v2.caffemodel** - Pre-trained model weights
3. **pts_in_hull.npy** - Cluster centers for colorization

## Download Instructions

Download the model files from: https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing

Place all three files in this `models/` directory.

## File Structure After Download

```
models/
├── colorization_deploy_v2.prototxt
├── colorization_release_v2.caffemodel
├── pts_in_hull.npy
└── README.md (this file)
```