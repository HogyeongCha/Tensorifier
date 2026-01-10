@echo off
cd /d "%~dp0"
echo Starting Tensorifier Data Preprocessing Agent...
streamlit run program.py
if %errorlevel% neq 0 (
    echo.
    echo Error occurred. Please make sure Streamlit is installed:
    echo pip install streamlit pandas numpy torch google-generativeai matplotlib seaborn scikit-learn
    echo.
    pause
)
