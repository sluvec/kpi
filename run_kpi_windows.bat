@echo off
echo ========================================
echo    KPI Analytics Dashboard Launcher
echo ========================================
echo.

REM Sprawdź czy Python jest dostępny
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python nie jest zainstalowany lub nie jest w PATH
    echo.
    echo Rozwiązania:
    echo 1. Zainstaluj Python z https://www.python.org/downloads/
    echo 2. Upewnij się, że zaznaczysz "Add Python to PATH" podczas instalacji
    echo 3. Uruchom ponownie ten skrypt
    echo.
    pause
    exit /b 1
)

echo [OK] Python znaleziony
python --version

REM Sprawdź czy Streamlit jest zainstalowany
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit nie jest zainstalowany
    echo Instaluję Streamlit...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo [ERROR] Nie udało się zainstalować Streamlit
        pause
        exit /b 1
    )
)

echo [OK] Streamlit zainstalowany

REM Sprawdź czy plik kpi11.py istnieje
if not exist "kpi11.py" (
    echo [ERROR] Plik kpi11.py nie został znaleziony
    echo Upewnij się, że skrypt jest w tym samym folderze co kpi11.py
    pause
    exit /b 1
)

echo [OK] Plik kpi11.py znaleziony
echo.
echo ========================================
echo    Uruchamianie KPI Dashboard...
echo ========================================
echo.
echo Aplikacja będzie dostępna pod adresem:
echo - Lokalnie: http://localhost:8501
echo - W sieci: http://[twoj-ip]:8501
echo.
echo Naciśnij Ctrl+C aby zatrzymać aplikację
echo.

REM Uruchom aplikację
python -m streamlit run kpi11.py

pause 