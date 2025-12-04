# Rutas
$projectPath = "C:\Users\Usuario\Dev\ProyectosDNI\dni-detection-api"
$pythonExe = "$projectPath\venv\Scripts\python.exe"
$mainPy = "$projectPath\main.py"
$urlFile = "$projectPath\ngrok_url.txt"
$logFile = "$projectPath\ngrok.log"

# Detener procesos anteriores si existen
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "ngrok" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# 1. Iniciar servidor Python en segundo plano
Start-Process -FilePath $pythonExe `
    -ArgumentList $mainPy `
    -WorkingDirectory $projectPath `
    -WindowStyle Hidden

# Esperar a que el servidor arranque
Start-Sleep -Seconds 8

# 2. Iniciar ngrok en segundo plano
Start-Process -FilePath "ngrok" `
    -ArgumentList "http", "8000", "--log=stdout" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile

# Esperar a que ngrok genere la URL
Start-Sleep -Seconds 5

# 3. Obtener la URL de ngrok via API local
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
    $publicUrl = $response.tunnels[0].public_url
    $publicUrl | Out-File -FilePath $urlFile -Force
    Write-Host "URL de ngrok: $publicUrl" -ForegroundColor Green
} catch {
    Write-Host "Esperando URL de ngrok..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
        $publicUrl = $response.tunnels[0].public_url
        $publicUrl | Out-File -FilePath $urlFile -Force
        Write-Host "URL de ngrok: $publicUrl" -ForegroundColor Green
    } catch {
        Write-Host "No se pudo obtener la URL. Revisa manualmente: http://127.0.0.1:4040" -ForegroundColor Red
    }
}