@echo off
title FreshHarvest AI - Model Training
echo ======================================================================
echo           FreshHarvest AI - Train Transfer Learning Models
echo ======================================================================
echo.

set PYTHON_EXE=C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" train.py --model all --epochs 5 --batch-size 16
) else (
    python train.py --model all --epochs 5 --batch-size 16
)

pause

