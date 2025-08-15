@echo off
echo Creating Python virtual environment for IFRS-16 LBO Engine...

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements-minimal.txt

:: Install package in development mode
pip install -e .

echo.
echo Virtual environment setup complete!
echo To activate: venv\Scripts\activate.bat
echo To run case study: python analysis\scripts\case_study_accor.py
