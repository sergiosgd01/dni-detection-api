# Detener Python
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Python detenido" -ForegroundColor Yellow

# Detener ngrok
Get-Process -Name "ngrok" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "ngrok detenido" -ForegroundColor Yellow

Write-Host "Todos los servicios detenidos" -ForegroundColor Green