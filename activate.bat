@echo off

:: Check if the .venv directory does NOT exist
if not exist .venv (
    echo .venv not found. Creating virtual environment...
    python -m venv .venv
) else (
    echo .venv already exists. Skipping creation.
)

:: Optional: Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Checking for missing packages...
python requirements.py