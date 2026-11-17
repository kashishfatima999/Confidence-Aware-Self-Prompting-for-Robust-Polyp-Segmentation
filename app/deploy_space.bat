@echo off
REM Run this from the repository root after setting the two model paths.
if "%YOLO_WEIGHTS%"=="" (
  echo Set YOLO_WEIGHTS first.
  exit /b 1
)
if "%SAM2_CHECKPOINT%"=="" (
  echo Set SAM2_CHECKPOINT first.
  exit /b 1
)
python app\app.py
