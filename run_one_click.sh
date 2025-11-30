#!/bin/bash

echo "========================================"
echo "ONE-CLICK IMAGE COLORIZER"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Install packages if needed
echo "Installing required packages..."
python3 install_all_packages.py

echo ""
echo "========================================"
echo "Starting Colorization..."
echo "========================================"
echo ""

# Run the one-click colorizer
if [ -z "$1" ]; then
    python3 one_click_colorizer.py
else
    python3 one_click_colorizer.py "$1"
fi

