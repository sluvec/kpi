# PowerShell Script - Instalacja bibliotek dla KPI Analytics Dashboard
# Uruchom jako Administrator: Right-click -> "Run as Administrator"

Write-Host "🚀 Instalacja bibliotek dla KPI Analytics Dashboard" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Sprawdź czy Python jest zainstalowany
Write-Host "`n📋 Sprawdzanie instalacji Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python znaleziony: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python nie jest zainstalowany!" -ForegroundColor Red
    Write-Host "📥 Pobierz Python z: https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "⚠️  Upewnij się, że zaznaczysz 'Add Python to PATH' podczas instalacji" -ForegroundColor Yellow
    exit 1
}

# Sprawdź czy pip jest dostępny
Write-Host "`n📦 Sprawdzanie pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip znaleziony: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip nie jest dostępny!" -ForegroundColor Red
    Write-Host "🔄 Próba naprawy pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

# Aktualizuj pip
Write-Host "`n🔄 Aktualizacja pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Lista wymaganych bibliotek
$libraries = @(
    "streamlit",
    "pandas", 
    "plotly",
    "numpy",
    "openpyxl",
    "xlrd"
)

Write-Host "`n📚 Instalacja wymaganych bibliotek..." -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow

# Instaluj każdą bibliotekę
foreach ($lib in $libraries) {
    Write-Host "📦 Instalowanie $lib..." -ForegroundColor Cyan
    try {
        pip install $lib
        Write-Host "✅ $lib zainstalowana pomyślnie" -ForegroundColor Green
    } catch {
        Write-Host "❌ Błąd podczas instalacji $lib" -ForegroundColor Red
        Write-Host "🔄 Próba alternatywnej instalacji..." -ForegroundColor Yellow
        python -m pip install $lib
    }
}

# Sprawdź instalację Streamlit
Write-Host "`n🔍 Weryfikacja instalacji..." -ForegroundColor Yellow
try {
    $streamlitVersion = streamlit --version 2>&1
    Write-Host "✅ Streamlit zainstalowany: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Streamlit nie został zainstalowany poprawnie!" -ForegroundColor Red
    Write-Host "🔄 Próba ponownej instalacji Streamlit..." -ForegroundColor Yellow
    pip install streamlit --force-reinstall
}

Write-Host "`n🎉 Instalacja zakończona!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "`n📋 Następne kroki:" -ForegroundColor Cyan
Write-Host "1. Skopiuj plik kpi11.py na komputer" -ForegroundColor White
Write-Host "2. Skopiuj plik sale_data.xlsx na komputer" -ForegroundColor White
Write-Host "3. Otwórz PowerShell w folderze z plikami" -ForegroundColor White
Write-Host "4. Uruchom: streamlit run kpi11.py" -ForegroundColor White
Write-Host "`n🌐 Aplikacja będzie dostępna pod adresem: http://localhost:8501" -ForegroundColor Green

Write-Host "`n⏸️  Naciśnij dowolny klawisz, aby zakończyć..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 