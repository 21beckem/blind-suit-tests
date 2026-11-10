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

echo Installing missing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Keep the terminal window open if run by double-clicking
cmd /k