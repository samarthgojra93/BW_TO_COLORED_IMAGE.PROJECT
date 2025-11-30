"""
Check and verify model files for the colorization project
"""
import os
import sys

def format_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def check_models():
    """Check if model files exist and are valid"""
    print("=" * 60)
    print("MODEL FILES STATUS CHECK")
    print("=" * 60)
    print()
    
    models_dir = "models"
    required_files = {
        "colorization_deploy_v2.prototxt": (1 * 1024, 10 * 1024),  # 1KB - 10KB
        "colorization_release_v2.caffemodel": (200 * 1024 * 1024, 400 * 1024 * 1024),  # 200MB - 400MB
        "pts_in_hull.npy": (1 * 1024, 100 * 1024)  # 1KB - 100KB
    }
    
    all_valid = True
    
    for filename, (min_size, max_size) in required_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"[X] MISSING: {filename}")
            all_valid = False
        else:
            size = os.path.getsize(filepath)
            size_str = format_size(size)
            
            if size == 0:
                print(f"[X] EMPTY:   {filename} ({size_str})")
                all_valid = False
            elif size < min_size:
                print(f"[!] TOO SMALL: {filename} ({size_str}) - Expected at least {format_size(min_size)}")
                all_valid = False
            elif size > max_size:
                print(f"[!] TOO LARGE: {filename} ({size_str}) - Expected at most {format_size(max_size)}")
                all_valid = False
            else:
                print(f"[OK] VALID:   {filename} ({size_str})")
    
    print()
    print("=" * 60)
    
    if all_valid:
        print("[OK] ALL MODEL FILES ARE VALID!")
        print("You can now use real AI colorization mode.")
        return True
    else:
        print("[X] SOME MODEL FILES ARE MISSING OR INVALID")
        print()
        print("TO FIX THIS:")
        print("1. Go to: https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing")
        print("2. Download all 3 model files")
        print("3. Place them in the 'models' folder")
        print("4. Run this script again to verify")
        print()
        print("Expected file sizes:")
        print("  - colorization_deploy_v2.prototxt: ~1-10 KB")
        print("  - colorization_release_v2.caffemodel: ~200-300 MB (LARGE FILE)")
        print("  - pts_in_hull.npy: ~1-100 KB")
        return False

if __name__ == "__main__":
    check_models()

