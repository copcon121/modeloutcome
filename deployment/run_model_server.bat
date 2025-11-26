@echo off
REM Run Outcome Model Inference Server (Windows)

echo Starting Outcome Model Inference Server...

REM Check if venv exists
if not exist "venv" (
    echo Error: Virtual environment not found. Please create one first:
    echo   python -m venv venv
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\outcome_transformer_v1.pt" (
    echo Warning: Model file not found at models\outcome_transformer_v1.pt
    echo Server will start but inference will fail until model is trained.
)

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Run server
echo Server will be available at: http://localhost:5002
echo API docs at: http://localhost:5002/docs
echo.

uvicorn src.layer3_model.inference.server:app --host 0.0.0.0 --port 5002 --reload --log-level info
