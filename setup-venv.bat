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

:: Keep the terminal window open if run by double-clicking
cmd /k

python -m pip install --upgrade pip
pip install -r requirements.txt