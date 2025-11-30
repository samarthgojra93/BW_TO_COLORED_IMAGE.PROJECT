# Example: Setting xAI API Key

## Quick Command

Replace `your_api_key_here` with your actual API key:

**Python script:**
```bash
python set_xai_key.py "your_api_key_here"
```

**PowerShell:**
```powershell
$env:XAI_API_KEY="your_api_key_here"
```

**Windows CMD:**
```cmd
set XAI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export XAI_API_KEY="your_api_key_here"
```

## Example

If your API key is `xai-abc123xyz789`, run:

```bash
python set_xai_key.py "xai-abc123xyz789"
```

Or in PowerShell:
```powershell
$env:XAI_API_KEY="xai-abc123xyz789"
```

## Verify

After setting, verify it works:
```bash
python -c "import os; key = os.getenv('XAI_API_KEY'); print('API Key set:', 'Yes' if key else 'No')"
```

## ✅ Done!

Once set, xAI features will automatically work in all scripts!

