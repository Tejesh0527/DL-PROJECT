@echo off
REM RATFA Test Verification Script
echo ============================================
echo RATFA Test Verification
echo ============================================

REM Activate environment
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
call venv\Scripts\activate.bat

echo.
echo Running test suite...
echo.

REM Run pytest
python -m pytest tests\ -v --tb=short > test_results.txt 2>&1

echo.
echo Test Results Summary:
echo ====================
findstr /i "passed failed error" test_results.txt

echo.
echo ============================================
echo Test Execution Complete
echo ============================================

REM Keep test results visible
pause