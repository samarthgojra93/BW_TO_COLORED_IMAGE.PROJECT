# 🔑 How to Get Your xAI API Key

## Step-by-Step Guide

### Step 1: Visit xAI Website
1. Go to **https://x.ai**
2. Click on **Sign Up** or **Log In**

### Step 2: Create/Login to Account
- If you don't have an account, create one
- If you have an account, log in

### Step 3: Navigate to API Settings
1. Look for **API** or **Developer** section in your account
2. This might be under:
   - Settings → API
   - Developer Portal
   - API Keys
   - Account → API Access

### Step 4: Create API Key
1. Click **Create New API Key** or **Generate API Key**
2. Give it a name (e.g., "Colorization Project")
3. Copy the API key immediately (you may not see it again!)

### Step 5: Set Up in Project
Run this command with your API key:
```bash
python get_xai_api_key.py
```

Or directly:
```bash
python set_xai_key.py "your_api_key_here"
```

## 📝 Important Notes

### API Key Format
Your API key will look like:
- `xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- Or similar format

### Security
- **Never share your API key publicly**
- **Don't commit it to version control** (it's saved in `.env` which should be in `.gitignore`)
- **Keep it secure** - treat it like a password

### Requirements
- You may need to **subscribe to a plan** that includes API access
- Some features may require **specific API access levels**
- Check **https://docs.x.ai** for API documentation

## 🚀 Quick Setup

### Automatic Setup (Recommended)
```bash
python get_xai_api_key.py
```
This will:
1. Guide you to get the API key
2. Help you enter it
3. Save it automatically
4. Test the connection

### Manual Setup
```bash
python set_xai_key.py "xai-your-actual-api-key-here"
```

## ✅ Verify Setup

After setting up, test the connection:
```bash
python test_xai_connection.py
```

## 🔗 Useful Links

- **xAI Website**: https://x.ai
- **xAI Documentation**: https://docs.x.ai
- **API Status**: Check xAI status page for service availability

## ❓ Troubleshooting

### "API key not found"
- Make sure you ran `python get_xai_api_key.py` or `python set_xai_key.py`
- Check that `.env` file exists in project directory
- Verify the key is in the file: `XAI_API_KEY=your_key_here`

### "Connection failed"
- Verify your API key is correct
- Check if your account has API access enabled
- Ensure you have an active subscription/plan
- Check xAI service status

### "xai-sdk not installed"
```bash
pip install xai-sdk
```

## 💡 Tips

1. **Save your API key** in a secure password manager
2. **Test immediately** after getting it to make sure it works
3. **Check usage limits** in your xAI account dashboard
4. **Monitor usage** to avoid unexpected charges

## 🎉 Ready to Use!

Once your API key is set up, xAI features will automatically work in:
- `one_click_colorizer.py`
- `app.py`
- Any script using xAI integration

