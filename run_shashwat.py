#!/usr/bin/env python3
"""
Run the colorization app with shashwat.jpg automatically
"""

import subprocess
import sys
import os

def run_app_with_shashwat():
    print("Running Black-and-White-to-Colour-Image with shashwat.jpg...")
    
    # Check if shashwat.jpg exists
    if not os.path.exists("shashwat.jpg"):
        print("ERROR: shashwat.jpg not found!")
        print("Please make sure shashwat.jpg exists in the current directory.")
        return
    
    # Run the app with shashwat.jpg as input
    try:
        result = subprocess.run([sys.executable, "app.py"], 
                              input="shashwat.jpg\n", 
                              text=True, 
                              capture_output=False)
        print(f"\nApplication finished with exit code: {result.returncode}")
        print("Check demo_colorized_output.jpg for the result!")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    run_app_with_shashwat()