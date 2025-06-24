#!/bin/bash

# ==============================================
# Skrypt instalacyjny dla KPI Dashboard
# Dla systemów macOS
# ==============================================

echo "🚀 Instalacja bibliotek dla KPI Dashboard..."
echo "============================================="

# Sprawdź czy pip3 jest zainstalowany
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 nie jest zainstalowany!"
    echo "Zainstaluj Python z https://python.org lub użyj brew install python"
    exit 1
fi

echo "✅ pip3 znaleziony"

# Lista bibliotek do zainstalowania
libraries=(
    "streamlit"
    "pandas" 
    "numpy"
    "openpyxl"
    "plotly"
    "networkx"
)

echo ""
echo "📦 Instaluję następujące biblioteki:"
for lib in "${libraries[@]}"; do
    echo "   - $lib"
done
echo ""

# Instalacja bibliotek
failed_installs=()

for lib in "${libraries[@]}"; do
    echo "🔄 Instaluję $lib..."
    if pip3 install "$lib" --user; then
        echo "✅ $lib zainstalowany pomyślnie"
    else
        echo "❌ Błąd podczas instalacji $lib"
        failed_installs+=("$lib")
    fi
    echo ""
done

# Podsumowanie
echo "============================================="
if [ ${#failed_installs[@]} -eq 0 ]; then
    echo "🎉 Wszystkie biblioteki zainstalowane pomyślnie!"
    echo ""
    echo "Możesz teraz uruchomić dashboard:"
    echo "   streamlit run kpi.py"
else
    echo "⚠️  Niektóre biblioteki nie zostały zainstalowane:"
    for failed_lib in "${failed_installs[@]}"; do
        echo "   - $failed_lib"
    done
    echo ""
    echo "Spróbuj zainstalować je ręcznie:"
    for failed_lib in "${failed_installs[@]}"; do
        echo "   pip3 install $failed_lib"
    done
fi

echo ""
echo "============================================="
