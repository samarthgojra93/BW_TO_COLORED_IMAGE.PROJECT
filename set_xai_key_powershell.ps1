# Set xAI API Key in PowerShell
param(
    [Parameter(Mandatory=$true)]
    [string]$ApiKey
)

# Set for current session
$env:XAI_API_KEY = $ApiKey
Write-Host "✓ xAI API key set for current PowerShell session" -ForegroundColor Green

# Also save to .env file
try {
    $envContent = @()
    
    # Read existing .env if it exists
    if (Test-Path ".env") {
        $envContent = Get-Content ".env" | Where-Object { $_ -notmatch "^XAI_API_KEY=" }
    }
    
    # Add new key
    $envContent += "XAI_API_KEY=$ApiKey"
    
    # Write back
    $envContent | Set-Content ".env"
    
    Write-Host "✓ xAI API key saved to .env file" -ForegroundColor Green
}
catch {
    Write-Host "Could not save to .env file: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 60
Write-Host "xAI API KEY CONFIGURED" -ForegroundColor Green
Write-Host "=" * 60
Write-Host ""
Write-Host "To make it permanent, add to system environment variables" -ForegroundColor Cyan
Write-Host "or run this command each time you open PowerShell:" -ForegroundColor Cyan
Write-Host "  `$env:XAI_API_KEY=`"$ApiKey`"" -ForegroundColor Yellow
Write-Host ""

