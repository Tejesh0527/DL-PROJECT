@echo off
REM RATFA Environment Setup Script
REM Creates virtual environment, installs dependencies, downloads YOLOv8 weights

echo ============================================
echo RATFA Environment Setup
echo ============================================

REM Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Create virtual environment
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo WARNING: Some packages may have failed to install
    )
) else (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

REM Download YOLOv8 weights
echo.
echo Downloading YOLOv8 weights...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
if %errorlevel% neq 0 (
    echo WARNING: Failed to download YOLOv8 weights
    echo You may need to download weights manually
) else (
    echo YOLOv8 weights downloaded successfully
)

REM Create sample video directory
echo.
echo Creating sample directories...
if not exist "data\sample_videos" mkdir "data\sample_videos"
if not exist "outputs" mkdir "outputs"
if not exist "outputs\datasets" mkdir "outputs\datasets"
if not exist "outputs\models" mkdir "outputs\models"

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To activate the environment later, run:
echo     call venv\Scripts\activate.bat
echo.
echo To run the application:
echo     run.bat
echo.
pause