@echo off
c:\Users\21bec\platform-tools\adb.exe forward --remove-all
c:\Users\21bec\platform-tools\adb.exe reverse tcp:5000 tcp:5000
if errorlevel 1 (
    echo Failed to create the ADB port forwarding.
    exit /b 1
)

c:\Users\21bec\local-git\blind-suit-tests\.venv\Scripts\python.exe c:\Users\21bec\local-git\blind-suit-tests\vr\monocular\phone-feed.py