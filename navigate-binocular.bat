@echo off
"%~dp0platform-tools/adb.exe" forward --remove-all
"%~dp0platform-tools/adb.exe" reverse tcp:5000 tcp:5000
if errorlevel 1 (
    echo Failed to create the ADB port forwarding.
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0vr\navigate-binocular\cams-to-phone.py"