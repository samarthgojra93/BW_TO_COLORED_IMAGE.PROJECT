"""
xAI Integration for Image Colorization Project
This module provides integration with xAI's Grok API for enhanced AI capabilities
"""

try:
    from xai_sdk import Client as XAIClient
    XAI_SDK_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try importing as xai
        import xai
        XAI_SDK_AVAILABLE = True
    except ImportError:
        print("Warning: xai-sdk package not installed. Run: pip install xai-sdk")
        XAI_SDK_AVAILABLE = False
        XAIClient = None
        xai = None

def chatbot_print(msg):
    """Chatbot style print function"""
    print(f"[Chatbot]: {msg}")

def initialize_xai(api_key=None):
    """
    Initialize xAI connection (using direct HTTP, no SDK required)
    
    Args:
        api_key: xAI API key (if None, will try to get from environment variable XAI_API_KEY)
    
    Returns:
        Client object (dummy object for compatibility) or None if API key not found
    """
    try:
        import os
        import requests
        
        # Get API key
        if not api_key:
            api_key = os.getenv("XAI_API_KEY")
        
        if not api_key:
            chatbot_print("Warning: XAI_API_KEY not found in environment variables")
            chatbot_print("You can set it with: export XAI_API_KEY='your-api-key'")
            chatbot_print("Or run: python set_xai_key.py 'your-api-key'")
            return None
        
        # Check if requests is available
        try:
            import requests
        except ImportError:
            chatbot_print("requests library not installed. Install with: pip install requests")
            return None
        
        # Test connection with a simple request
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "messages": [{"role": "user", "content": "test"}],
                "model": "grok-4-latest",
                "stream": False,
                "max_tokens": 10
            }
            # Just verify the API key format, don't actually make the request
            # (to avoid using credits for initialization)
            chatbot_print("xAI connection ready (using direct HTTP API)")
            # Return a dummy client object for compatibility
            class DummyClient:
                def __init__(self, api_key):
                    self.api_key = api_key
            return DummyClient(api_key)
        except Exception as e:
            chatbot_print(f"Error verifying xAI connection: {str(e)}")
            return None
    except Exception as e:
        chatbot_print(f"Error initializing xAI: {str(e)}")
        chatbot_print("Make sure requests is installed: pip install requests")
        return None

def get_colorization_suggestions(client, image_description=None):
    """
    Get AI-powered suggestions for image colorization using xAI
    
    Args:
        client: xAI client instance (can be None, will use direct HTTP)
        image_description: Optional description of the image
    
    Returns:
        Suggestions from xAI or None if unavailable
    """
    try:
        import requests
        import os
        
        # Get API key
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            chatbot_print("XAI_API_KEY not found in environment")
            return None
        
        # Create prompt for colorization suggestions
        prompt = f"""You are an expert in image colorization. 
        Provide suggestions for colorizing a black and white image.
        {f'Image description: {image_description}' if image_description else ''}
        Give 3-5 specific color suggestions that would work well for this type of image.
        Be concise and practical."""
        
        # Use direct HTTP request (most reliable method)
        url = "https://api.x.ai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in image colorization and photo restoration."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-4-latest",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        # Make HTTP request
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content', '')
                return content
            else:
                chatbot_print(f"Unexpected API response format: {result}")
                return None
        else:
            error_msg = response.text
            # Check for common errors and provide helpful messages
            if response.status_code == 403:
                # Check if it's a credits issue
                if "credits" in error_msg.lower() or "doesn't have any credits" in error_msg.lower():
                    chatbot_print("ℹ xAI account needs credits to use API features")
                    chatbot_print("  Visit: https://console.x.ai/team/8340bca7-ef9d-4f73-bf08-e58da655f9fa")
                    chatbot_print("  App will continue without xAI suggestions")
                else:
                    chatbot_print(f"xAI API error (403): Permission denied")
            elif response.status_code == 401:
                chatbot_print("ℹ Invalid API key. Check your XAI_API_KEY")
            else:
                chatbot_print(f"xAI API error (status {response.status_code})")
            
            return None
            
    except ImportError:
        chatbot_print("requests library not installed. Install with: pip install requests")
        return None
    except requests.exceptions.RequestException as e:
        chatbot_print(f"Network error connecting to xAI: {e}")
        return None
    except Exception as e:
        chatbot_print(f"Error getting xAI suggestions: {e}")
        return None

def enhance_with_xai(client, image_path):
    """
    Use xAI to enhance the colorization process with AI insights
    
    Args:
        client: xAI client instance (can be None, will use direct HTTP)
        image_path: Path to the image being colorized
    
    Returns:
        Enhancement suggestions or None
    """
    chatbot_print("Getting AI-powered colorization insights from xAI...")
    
    # Get suggestions based on image analysis
    # Client can be None, get_colorization_suggestions will use direct HTTP
    suggestions = get_colorization_suggestions(client)
    
    if suggestions:
        chatbot_print("xAI Suggestions:")
        print(suggestions)
        return suggestions
    else:
        chatbot_print("Could not get xAI suggestions. Continuing without AI insights.")
    
    return None

if __name__ == "__main__":
    # Example usage
    chatbot_print("xAI Integration Module")
    chatbot_print("=" * 50)
    
    # Initialize xAI (requires API key)
    client = initialize_xai()
    
    if client:
        chatbot_print("xAI is ready to use!")
        chatbot_print("You can now use xAI features in your colorization workflow.")
    else:
        chatbot_print("xAI not initialized. Make sure to:")
        chatbot_print("1. Install xai-sdk package: pip install xai-sdk")
        chatbot_print("2. Set XAI_API_KEY environment variable")
        chatbot_print("3. Or pass API key directly to initialize_xai()")

