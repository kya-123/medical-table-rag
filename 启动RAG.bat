@echo off
chcp 65001 >nul
title RAG service - medical domain QA
color 0A
cd /d "%~dp0"

rem ---- find python: local .venv -> legacy myenv -> PATH ----
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\..\myenv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem ---- models present? ----
if not exist "%~dp0model\models" (
    echo.
    echo [warning] model\models not found.
    echo           run:  python scripts\download_models.py
    echo.
)

rem ---- port 8001 already taken? ----
netstat -ano 2>nul | findstr ":8001" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo.
    echo [info] port 8001 is already in use, not starting again.
    echo        open: http://127.0.0.1:8001/docs
    echo.
    start "" http://127.0.0.1:8001/docs
    timeout /t 6 >nul
    exit /b 0
)

rem ---- offline: models are local, do not touch HuggingFace ----
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false
set ANONYMIZED_TELEMETRY=False

rem ---- identity: generic by default; set RAG_PROFILE=diabetes for the course build ----
if not defined RAG_PROFILE set RAG_PROFILE=generic

echo.
echo ================================================================
echo   Starting... loading two models takes about 15 seconds
echo   *** keep this window open, closing it stops the service ***
echo ================================================================
echo.
echo   API docs : http://127.0.0.1:8001/docs
echo   Ask      : http://127.0.0.1:8001/ask
echo.

rem open the docs page after 15s
start "" /b cmd /c "timeout /t 15 >nul & start "" http://127.0.0.1:8001/docs"

"%PY%" rag_system.py

echo.
echo ================================================================
echo   service stopped.
echo ================================================================
echo.
pause
