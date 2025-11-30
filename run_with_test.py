#!/usr/bin/env python3
"""
Run the colorization app with the test image automatically
"""

import subprocess
import sys
import os

def run_app_with_test_image():
    print("Running Black-and-White-to-Colour-Image with test image...")
    
    # Check if test image exists
    if not os.path.exists("test_grayscale.jpg"):
        print("ERROR: test_grayscale.jpg not found!")
        print("Please make sure the test image exists in the current directory.")
        return
    
    # Run the app with test image as input
    try:
        result = subprocess.run([sys.executable, "app.py"], 
                              input="test_grayscale.jpg\n", 
                              text=True, 
                              capture_output=False)
        print(f"\nApplication finished with exit code: {result.returncode}")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    run_app_with_test_image()