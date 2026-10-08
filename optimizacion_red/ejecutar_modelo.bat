@echo off
REM Ejecuta el modelo en Windows: crea el entorno virtual la primera vez, instala dependencias y abre el reporte.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual .venv ...
    py -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" modelo_red.py --abrir %*
pause
