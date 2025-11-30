# Model Files Status Checker
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MODEL FILES STATUS CHECK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$modelsDir = "models"
$allValid = $true

# Define required files with expected size ranges
$requiredFiles = @{
    "colorization_deploy_v2.prototxt" = @{MinMB = 0.001; MaxMB = 0.01; Description = "Network architecture"}
    "colorization_release_v2.caffemodel" = @{MinMB = 200; MaxMB = 400; Description = "AI model weights (LARGE)"}
    "pts_in_hull.npy" = @{MinMB = 0.001; MaxMB = 0.1; Description = "Color cluster centers"}
}

foreach ($file in $requiredFiles.Keys) {
    $filePath = Join-Path $modelsDir $file
    $fileInfo = $requiredFiles[$file]
    
    if (-not (Test-Path $filePath)) {
        Write-Host "[X] MISSING: $file" -ForegroundColor Red
        $allValid = $false
    } else {
        $sizeBytes = (Get-Item $filePath).Length
        $sizeMB = [math]::Round($sizeBytes / 1MB, 3)
        
        if ($sizeBytes -eq 0) {
            Write-Host "[X] EMPTY:   $file ($sizeMB MB)" -ForegroundColor Red
            $allValid = $false
        } elseif ($sizeMB -lt $fileInfo.MinMB) {
            Write-Host "[!] TOO SMALL: $file ($sizeMB MB) - Expected at least $($fileInfo.MinMB) MB" -ForegroundColor Yellow
            Write-Host "    Description: $($fileInfo.Description)" -ForegroundColor Gray
            $allValid = $false
        } elseif ($sizeMB -gt $fileInfo.MaxMB) {
            Write-Host "[!] TOO LARGE: $file ($sizeMB MB) - Expected at most $($fileInfo.MaxMB) MB" -ForegroundColor Yellow
            $allValid = $false
        } else {
            Write-Host "[OK] VALID:   $file ($sizeMB MB)" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

if ($allValid) {
    Write-Host "[OK] ALL MODEL FILES ARE VALID!" -ForegroundColor Green
    Write-Host "You can now use real AI colorization mode." -ForegroundColor Green
} else {
    Write-Host "[X] SOME MODEL FILES ARE MISSING OR INVALID" -ForegroundColor Red
    Write-Host ""
    Write-Host "TO FIX THIS:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://drive.google.com/drive/folders/1S8_bUXXZg7f6hYKfEe9nVWKgn2zyl4bP?usp=sharing" -ForegroundColor White
    Write-Host "2. Download all 3 model files" -ForegroundColor White
    Write-Host "3. Place them in the 'models' folder" -ForegroundColor White
    Write-Host "4. Run this script again to verify" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected file sizes:" -ForegroundColor Yellow
    Write-Host "  - colorization_deploy_v2.prototxt: ~1-10 KB" -ForegroundColor White
    Write-Host "  - colorization_release_v2.caffemodel: ~200-300 MB (LARGE FILE)" -ForegroundColor White
    Write-Host "  - pts_in_hull.npy: ~1-100 KB" -ForegroundColor White
}

