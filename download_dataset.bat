@echo off
title FreshHarvest AI - Download Full Kaggle Dataset (13,500+ Images)
cd /d "%~dp0"
echo =====================================================================
echo  Downloading Full Kaggle Dataset (13,500+ Images) for FreshHarvest AI
echo =====================================================================
"C:\Users\umesh\AppData\Local\Programs\Python\Python313\python.exe" download_kaggle_dataset.py
pause

