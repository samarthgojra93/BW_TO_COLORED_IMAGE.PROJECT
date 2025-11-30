#!/usr/bin/env python3
"""Check installed packages"""

packages = {
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'xai-sdk': 'xai',
    'shap': 'shap',
    'lime': 'lime'
}

print("=" * 60)
print("PACKAGE INSTALLATION STATUS")
print("=" * 60)
print()

installed = []
missing = []

for package_name, import_name in packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'installed')
        print(f"✓ {package_name}: {version}")
        installed.append(package_name)
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        missing.append(package_name)
    except Exception as e:
        print(f"? {package_name}: Error checking ({e})")

print()
print("=" * 60)
print(f"Summary: {len(installed)}/{len(packages)} packages installed")
print("=" * 60)

if missing:
    print()
    print("To install missing packages, run:")
    print(f"  python -m pip install {' '.join(missing)}")
else:
    print()
    print("✓ All packages are installed!")

