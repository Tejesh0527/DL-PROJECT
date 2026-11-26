@echo off
REM RATFA Runner Script
REM Activates environment and starts the RATFA Application / Dashboard

echo ============================================
echo RATFA Application Launcher
echo ============================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Activate virtual environment
call venv\Scripts\activate.bat

python -c "import streamlit" 2>nul
if %errorlevel% equ 0 (
    if exist "app.py" (
        echo Starting Streamlit Dashboard...
        python -m streamlit run app.py --server.port 8501
        goto done
    )
)

echo Running RATFA Main Pipeline...
python main.py --mode video --input data\input_videos\traffic_demo.mp4 --no-display --adapt

:done
echo.
echo Execution complete. Outputs generated in outputs\
pause