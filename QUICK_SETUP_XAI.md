# 🔑 Quick Setup: xAI API Key

## Method 1: Using the Setup Script (Easiest)

**Windows:**
```bash
python set_xai_key.py "your_api_key_here"
```

**Or use the batch file:**
```bash
set_xai_key.bat "your_api_key_here"
```

**Or PowerShell script:**
```powershell
.\set_xai_key_powershell.ps1 -ApiKey "your_api_key_here"
```

## Method 2: Set Environment Variable Directly

### Windows PowerShell:
```powershell
$env:XAI_API_KEY="your_api_key_here"
```

### Windows CMD:
```cmd
set XAI_API_KEY=your_api_key_here
```

### Linux/Mac:
```bash
export XAI_API_KEY="your_api_key_here"
```

## Method 3: Create .env File Manually

Create a file named `.env` in the project directory with:
```
XAI_API_KEY=your_api_key_here
```

## ✅ Verify It's Set

Run this to check:
```bash
python -c "import os; print('API Key:', os.getenv('XAI_API_KEY', 'NOT SET'))"
```

## 🚀 After Setting

Once the API key is set, xAI features will automatically work in:
- `one_click_colorizer.py`
- `app.py`
- Any script that uses xAI integration

## 📝 Notes

- The `.env` file method persists across sessions
- Environment variable method only works for current session
- The app will automatically load from `.env` file if it exists
- xAI features are optional - app works without it

## 🔗 Get Your API Key

Visit: https://x.ai to get your API key

