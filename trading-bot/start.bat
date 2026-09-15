@echo off
rem =====================================================
rem  gnidart trading bot - start both sleeves (VPS)
rem  Place this file in the project root (next to config/, data/, scripts/)
rem  Creates two named windows: gnidart-gold / gnidart-silver
rem  Stop cleanly: Ctrl+C inside each window (server-side SL/TP stay active)
rem =====================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo === gnidart: GOLD sleeve (XAUUSD) ===
start "gnidart-gold" cmd /k py scripts\run_live.py

rem 5s stagger so both runners don't hit MT5 init simultaneously
timeout /t 5 /nobreak >nul

echo === gnidart: SILVER sleeve (XAGUSD) ===
start "gnidart-silver" cmd /k py scripts\run_live.py --symbol XAGUSD

echo.
echo Both runners started in their own windows.
echo To stop: press Ctrl+C inside each window.
timeout /t 10
