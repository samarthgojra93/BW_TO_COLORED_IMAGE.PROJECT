#!/usr/bin/env python3
"""
Test xAI API Connection
"""

import os

# Load .env file if it exists
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
        except Exception:
            pass

load_env_file()

def test_connection():
    """Test xAI connection"""
    print("=" * 60)
    print("TESTING xAI CONNECTION")
    print("=" * 60)
    print()
    
    # Check API key
    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        print("✗ XAI_API_KEY not found")
        print()
        print("To set your API key, run:")
        print("  python get_xai_api_key.py")
        print()
        print("Or set it manually:")
        print("  python set_xai_key.py \"your_api_key_here\"")
        return False
    
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"✓ API Key found: {masked}")
    print()
    
    # Check xai-sdk
    try:
        from xai_integration import initialize_xai
        print("✓ xAI integration module loaded")
    except ImportError as e:
        print(f"✗ xAI integration module not available: {e}")
        print()
        print("Install with: pip install xai-sdk")
        return False
    
    # Test connection
    print()
    print("Testing connection to xAI...")
    try:
        client = initialize_xai()
        if client:
            print("✓ Connection successful!")
            print()
            print("xAI is ready to use in your colorization app!")
            return True
        else:
            print("✗ Connection failed")
            print("Check your API key and make sure it's valid")
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        print()
        print("Possible issues:")
        print("  1. Invalid API key")
        print("  2. API key doesn't have proper permissions")
        print("  3. Network connection issue")
        print("  4. xAI service temporarily unavailable")
        return False

if __name__ == "__main__":
    success = test_connection()
    print()
    if success:
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        print()
        print("Run 'python get_xai_api_key.py' to set up your API key")

