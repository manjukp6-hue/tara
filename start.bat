@echo off
title TARA AI Core
echo ==============================================================================
echo   Starting TARA AI Core Native Server
echo ==============================================================================

if exist ".\target\x86_64-pc-windows-gnu\release\tara_server.exe" (
    .\target\x86_64-pc-windows-gnu\release\tara_server.exe
) else if exist ".\target\release\tara_server.exe" (
    .\target\release\tara_server.exe
) else (
    cargo run --release --bin tara_server
)
pause
