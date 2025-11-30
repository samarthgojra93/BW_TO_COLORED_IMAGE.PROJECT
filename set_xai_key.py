#!/usr/bin/env python3
"""
Set xAI API Key
"""

import os
import sys

def set_xai_key(api_key):
    """Set xAI API key"""
    if not api_key or api_key.strip() == "":
        print("Error: API key cannot be empty")
        return False
    
    api_key = api_key.strip().strip('"').strip("'")
    
    # Set for current session
    os.environ['XAI_API_KEY'] = api_key
    print(f"✓ xAI API key set for current session")
    
    # Save to .env file
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
        
        print(f"✓ xAI API key saved to .env file")
        return True
        
    except Exception as e:
        print(f"Could not save to .env file: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your xAI API key: ").strip()
    
    if set_xai_key(api_key):
        print()
        print("=" * 60)
        print("xAI API KEY CONFIGURED")
        print("=" * 60)
        print()
        print("For permanent setup, also run this in PowerShell:")
        print(f'  $env:XAI_API_KEY="{api_key}"')
        print()
        print("Or add to your system environment variables.")
        print()
        print("xAI features are now enabled!")
    else:
        print("Failed to set API key")

if __name__ == "__main__":
    main()

