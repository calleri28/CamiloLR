@echo off
REM ===========================================================================
REM  Ejecuta el modelo sin PowerShell (no le afecta la politica de scripts).
REM  - Crea el entorno .venv la primera vez e instala las librerias.
REM  - Si el .venv fue copiado de otro PC y no funciona, lo vuelve a crear.
REM  Uso: doble clic, o desde CMD:  ejecutar_modelo.bat  [--generar-prueba]
REM ===========================================================================
setlocal
cd /d "%~dp0"

if not exist "requirements.txt" (
    echo [ERROR] No encuentro requirements.txt en "%CD%".
    echo         Este archivo .bat debe estar dentro de la carpeta optimizacion_red,
    echo         junto a modelo_red.py.
    pause
    exit /b 1
)

set "PY=.venv\Scripts\python.exe"

if exist "%PY%" (
    "%PY%" -c "import pandas, numpy, pulp, openpyxl" >nul 2>&1
    if errorlevel 1 (
        echo El entorno .venv no funciona en este equipo; se vuelve a crear...
        rmdir /s /q ".venv"
    )
)

if not exist "%PY%" (
    echo Creando entorno virtual .venv ...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if not exist "%PY%" (
        echo [ERROR] No se encontro Python. Instale Python 3.11 o 3.12 ^(python.org o Company Portal^)
        echo         marcando "Add python.exe to PATH".
        pause
        exit /b 1
    )
    echo Instalando librerias ^(pandas, numpy, pulp, openpyxl^) ...
    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] No se pudieron descargar las librerias. En equipos corporativos suele ser
        echo         el proxy o la inspeccion SSL. Vea la seccion "PC corporativo" del README.
        rmdir /s /q ".venv"
        pause
        exit /b 1
    )
)

"%PY%" modelo_red.py --abrir %*
echo.
pause
