@echo off
REM Doble clic para crear el entorno (solo la primera vez) y ejecutar el modelo.
cd /d "%~dp0"
if not exist .venv (
  echo Creando entorno virtual...
  py -m venv .venv || python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python ejecutar_modelo.py %*
pause
