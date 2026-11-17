@echo off
title FreshHarvest AI - Model Evaluation
echo ======================================================================
echo           FreshHarvest AI - Evaluate Models on Held-Out Test Set
echo ======================================================================
echo.

set PYTHON_EXE=C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" evaluate.py
) else (
    python evaluate.py
)

pause

