#!/usr/bin/env python3
"""
Setup API Keys for the Colorization Project
"""

import os
import sys

def setup_xai_key():
    """Setup xAI API key"""
    print("=" * 60)
    print("xAI API KEY SETUP")
    print("=" * 60)
    print()
    print("To use xAI features, you need an API key from https://x.ai")
    print()
    
    current_key = os.environ.get('XAI_API_KEY', '')
    if current_key:
        print(f"Current XAI_API_KEY is set: {current_key[:10]}...")
        change = input("Do you want to change it? (y/n): ").lower()
        if change != 'y':
            print("Keeping existing key.")
            return current_key
        print()
    
    api_key = input("Enter your xAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("xAI API key setup skipped.")
        return None
    
    # Set for current session
    os.environ['XAI_API_KEY'] = api_key
    
    # Save to .env file for future use
    try:
        env_file = '.env'
        env_content = []
        
        # Read existing .env if it exists
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_content = f.readlines()
        
        # Remove old XAI_API_KEY if exists
        env_content = [line for line in env_content if not line.startswith('XAI_API_KEY=')]
        
        # Add new key
        env_content.append(f'XAI_API_KEY={api_key}\n')
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(env_content)
        
        print("✓ xAI API key saved to .env file")
        print("  Note: For permanent setup, also set it as environment variable:")
        print("  Windows PowerShell: $env:XAI_API_KEY='your-key-here'")
        print("  Windows CMD: set XAI_API_KEY=your-key-here")
        print("  Linux/Mac: export XAI_API_KEY='your-key-here'")
        
    except Exception as e:
        print(f"Could not save to .env file: {e}")
        print("Please set environment variable manually:")
        print("  Windows PowerShell: $env:XAI_API_KEY='your-key-here'")
        print("  Windows CMD: set XAI_API_KEY=your-key-here")
        print("  Linux/Mac: export XAI_API_KEY='your-key-here'")
    
    return api_key

def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"Could not load .env file: {e}")

def main():
    print()
    print("=" * 60)
    print("API KEYS SETUP")
    print("=" * 60)
    print()
    
    # Load existing .env if available
    load_env_file()
    
    # Setup xAI key
    xai_key = setup_xai_key()
    
    print()
    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print()
    
    if xai_key:
        print("✓ xAI API key configured")
        print("  xAI features will be available in the colorization app")
    else:
        print("ℹ xAI API key not set")
        print("  xAI features will be skipped (app will still work)")
    
    print()
    print("You can now run the colorization app!")
    print("  python one_click_colorizer.py <image_path>")

if __name__ == "__main__":
    main()

