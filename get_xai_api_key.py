#!/usr/bin/env python3
"""
xAI API Key Setup Guide and Helper
This script helps you get and configure your xAI API key
"""

import os
import sys
import webbrowser
from urllib.parse import quote

def print_header():
    """Print header"""
    print("=" * 70)
    print("xAI API KEY SETUP GUIDE")
    print("=" * 70)
    print()

def open_xai_website():
    """Open xAI website in browser"""
    url = "https://x.ai"
    print(f"Opening xAI website: {url}")
    try:
        webbrowser.open(url)
        print("✓ Browser opened")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Please manually visit: {url}")

def guide_get_api_key():
    """Guide user through getting API key"""
    print("STEP 1: GET YOUR xAI API KEY")
    print("-" * 70)
    print()
    print("To get your xAI API key:")
    print()
    print("1. Visit: https://x.ai")
    print("2. Sign up or log in to your account")
    print("3. Navigate to API settings or Developer section")
    print("4. Create a new API key")
    print("5. Copy your API key (it will look like: xai-xxxxxxxxxxxxx)")
    print()
    print("Note: You may need to:")
    print("  - Sign up for xAI/Grok if you don't have an account")
    print("  - Subscribe to a plan that includes API access")
    print("  - Check the API documentation at https://docs.x.ai")
    print()
    
    open_browser = input("Would you like to open x.ai in your browser? (y/n): ").lower()
    if open_browser == 'y':
        open_xai_website()
        print()
    
    print("Press Enter when you have your API key ready...")
    input()

def set_api_key_interactive():
    """Interactively set the API key"""
    print()
    print("STEP 2: ENTER YOUR API KEY")
    print("-" * 70)
    print()
    
    api_key = input("Enter your xAI API key: ").strip()
    
    if not api_key:
        print("No API key entered. Exiting.")
        return False
    
    # Remove quotes if user added them
    api_key = api_key.strip('"').strip("'")
    
    # Basic validation
    if len(api_key) < 10:
        print("Warning: API key seems too short. Are you sure it's correct?")
        confirm = input("Continue anyway? (y/n): ").lower()
        if confirm != 'y':
            return False
    
    return set_api_key(api_key)

def set_api_key(api_key):
    """Set the API key"""
    print()
    print("STEP 3: SAVING API KEY")
    print("-" * 70)
    print()
    
    # Set for current session
    os.environ['XAI_API_KEY'] = api_key
    print("✓ API key set for current session")
    
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
        
        print(f"✓ API key saved to .env file")
        print(f"  Location: {os.path.abspath(env_file)}")
        return True
        
    except Exception as e:
        print(f"✗ Could not save to .env file: {e}")
        print()
        print("You can manually create a .env file with:")
        print(f"  XAI_API_KEY={api_key}")
        return False

def verify_api_key():
    """Verify the API key is set"""
    print()
    print("STEP 4: VERIFICATION")
    print("-" * 70)
    print()
    
    api_key = os.getenv('XAI_API_KEY')
    
    if api_key:
        # Show partial key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ API key is set: {masked_key}")
        print()
        print("To verify it works, you can test it:")
        print("  python test_xai_connection.py")
        return True
    else:
        print("✗ API key is not set")
        return False

def show_usage_instructions():
    """Show how to use the API key"""
    print()
    print("=" * 70)
    print("USAGE INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Your API key is now configured! Here's how to use it:")
    print()
    print("1. The API key is saved in .env file and will be loaded automatically")
    print()
    print("2. For current PowerShell session, you can also set:")
    print('   $env:XAI_API_KEY="your_key_here"')
    print()
    print("3. Run the colorization app:")
    print("   python one_click_colorizer.py path/to/image.jpg")
    print()
    print("4. xAI features will work automatically!")
    print()
    print("=" * 70)

def test_xai_connection():
    """Test xAI connection"""
    print()
    print("Would you like to test the connection? (y/n): ", end="")
    test = input().lower()
    
    if test == 'y':
        print()
        print("Testing xAI connection...")
        try:
            from xai_integration import initialize_xai
            client = initialize_xai()
            if client:
                print("✓ xAI connection successful!")
                return True
            else:
                print("✗ xAI connection failed. Check your API key.")
                return False
        except Exception as e:
            print(f"✗ Error testing connection: {e}")
            print("Make sure xai-sdk is installed: pip install xai-sdk")
            return False
    return None

def main():
    """Main function"""
    print_header()
    
    # Check if API key already exists
    existing_key = os.getenv('XAI_API_KEY')
    if existing_key:
        masked = existing_key[:8] + "..." + existing_key[-4:] if len(existing_key) > 12 else "***"
        print(f"Found existing API key: {masked}")
        use_existing = input("Use existing key or set a new one? (use/new): ").lower()
        if use_existing == 'use':
            print("Using existing API key.")
            verify_api_key()
            show_usage_instructions()
            return
    
    # Guide through getting API key
    guide_get_api_key()
    
    # Set API key
    if set_api_key_interactive():
        # Verify
        if verify_api_key():
            # Show usage
            show_usage_instructions()
            # Optional test
            test_xai_connection()
    else:
        print()
        print("API key setup cancelled or failed.")
        print("You can run this script again later.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)

