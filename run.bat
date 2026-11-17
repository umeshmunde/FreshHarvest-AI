@echo off
title FreshHarvest AI - Multi-Model Quality Assessment System
echo ======================================================================
echo           FreshHarvest AI - Fruit Freshness & Quality System
echo ======================================================================
echo.

set PYTHON_EXE=C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe

if exist "%PYTHON_EXE%" (
    echo Using Python at: %PYTHON_EXE%
    "%PYTHON_EXE%" app.py
) else (
    echo Using default system python...
    python app.py
)

pause

