#!/usr/bin/env python3
"""
Test xAI API using the actual API format
"""

import os
import requests
import json

# Load .env file
def load_env_file():
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

def test_xai_api():
    """Test xAI API using direct HTTP request"""
    print("=" * 60)
    print("TESTING xAI API (Direct HTTP)")
    print("=" * 60)
    print()
    
    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        print("✗ XAI_API_KEY not found")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print()
    
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for image colorization."
            },
            {
                "role": "user",
                "content": "Hello! Can you help with image colorization suggestions?"
            }
        ],
        "model": "grok-4-latest",
        "stream": False,
        "temperature": 0
    }
    
    print("Sending request to xAI API...")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API request successful!")
            print()
            print("Response:")
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content', '')
                print(content)
            else:
                print(json.dumps(result, indent=2))
            return True
        else:
            print(f"✗ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("✗ requests library not installed")
        print("Install with: pip install requests")
        exit(1)
    
    success = test_xai_api()
    print()
    if success:
        print("=" * 60)
        print("✓ API TEST SUCCESSFUL")
        print("=" * 60)
        print()
        print("Your xAI API is working correctly!")
        print("You can now use xAI features in the colorization app.")
    else:
        print("=" * 60)
        print("✗ API TEST FAILED")
        print("=" * 60)

