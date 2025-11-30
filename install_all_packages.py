#!/usr/bin/env python3
"""
Install all required packages for the colorization project
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=" * 60)
    print("INSTALLING ALL REQUIRED PACKAGES")
    print("=" * 60)
    print()
    
    # Core packages
    packages = [
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "matplotlib>=3.5.0",
    ]
    
    # Optional packages
    optional_packages = [
        "xai-sdk>=0.1.0",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
    ]
    
    print("Installing core packages...")
    for package in packages:
        print(f"  Installing {package}...", end=" ")
        if install_package(package):
            print("✓")
        else:
            print("✗")
    
    print()
    print("Installing optional packages...")
    for package in optional_packages:
        print(f"  Installing {package}...", end=" ")
        if install_package(package):
            print("✓")
        else:
            print("✗ (optional - can continue without)")
    
    print()
    print("=" * 60)
    print("INSTALLATION COMPLETE")
    print("=" * 60)
    print()
    print("You can now run: python one_click_colorizer.py <image_path>")

if __name__ == "__main__":
    main()

