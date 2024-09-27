@echo off

:showMenu
echo 1. Connect to the frontend server normally
echo 2. Connect to the frontend server with port forwarding
set /p choice="Enter your choice (1 or 2): "

if "%choice%" equ "1" (
    call :connectFrontend 
) else if "%choice%" equ "2" (
    call :connectWithPortForwarding "key"
) else (
    echo Invalid choice. Please enter 1 or 2.
    goto :showMenu
)

:: Optional: Add a pause to keep the command prompt open
pause
goto :eof

:connectFrontend
echo Connecting to the frontend server via SSH...
start ssh  -p 22222 student13@ictlab.usth.edu.vn
goto :eof

:connectWithPortForwarding
echo Connecting to the frontend server via SSH with port forwarding...
start ssh -i %1 -p 22222 -L localhost:25252:localhost:25252 student13@ictlab.usth.edu.vn
goto :eof

