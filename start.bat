@echo off
title TARA AI Core
echo ==============================================================================
echo   Starting TARA Architecture Live Watcher Daemon & Server
echo ==============================================================================

REM Check if architecture_sync daemon is already running
tasklist /FI "IMAGENAME eq architecture_sync.exe" 2>NUL | find /I /N "architecture_sync.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [Architecture Watcher] Daemon is already running in background.
) else (
    echo [Architecture Watcher] Auto-starting Persistent Background Daemon...
    if exist ".\target\x86_64-pc-windows-gnu\release\architecture_sync.exe" (
        start "TARA Architecture Watcher" /B ".\target\x86_64-pc-windows-gnu\release\architecture_sync.exe" --watch
    ) else if exist ".\target\release\architecture_sync.exe" (
        start "TARA Architecture Watcher" /B ".\target\release\architecture_sync.exe" --watch
    ) else if exist ".\target\x86_64-pc-windows-gnu\debug\architecture_sync.exe" (
        start "TARA Architecture Watcher" /B ".\target\x86_64-pc-windows-gnu\debug\architecture_sync.exe" --watch
    ) else (
        start "TARA Architecture Watcher" /B cargo run --release --bin architecture_sync -- --watch
    )
    echo [Architecture Watcher] Persistent Daemon launched.
)

if exist ".\target\x86_64-pc-windows-gnu\release\tara_server.exe" (
    .\target\x86_64-pc-windows-gnu\release\tara_server.exe
) else if exist ".\target\release\tara_server.exe" (
    .\target\release\tara_server.exe
) else (
    cargo run --release --bin tara_server
)
pause

