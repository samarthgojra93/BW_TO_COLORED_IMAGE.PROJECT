# How to Add Images to the Colorization App

## Quick Start
1. Copy your grayscale/black & white image to this folder
2. Run: `py app.py`
3. Enter the image filename when prompted

## Example Usage

### Images in Project Folder:
```
test_grayscale.jpg          ← Use this for testing
my_photo.jpg                ← Add your images here
old_picture.png            ← Or here
```

### Images Elsewhere:
```
C:\Users\asus\Desktop\photo.jpg
C:\Users\asus\Pictures\image.png
"C:\Users\asus\OneDrive\Pictures\Screenshots\photo.jpg"
```

## Supported Formats
- .jpg / .jpeg
- .png
- .bmp
- .tiff

## Tips
- Use quotes around paths with spaces
- Include the file extension
- Grayscale images work best
- High resolution gives better results

## Easy Commands
```bash
# Run with test image
py run_with_test.py

# Run interactively
py app.py

# Run with batch file
run_app.bat
```