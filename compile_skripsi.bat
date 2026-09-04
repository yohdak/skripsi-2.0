@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Mengompilasi Skripsi Lengkap (main.tex)...
echo ========================================================
latexmk -xelatex -synctex=1 -interaction=nonstopmode main.tex

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   [BERHASIL] main.pdf berhasil dibuat!
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   [GAGAL] Terjadi kesalahan saat kompilasi.
    echo   Silakan periksa pesan log di atas.
    echo ========================================================
)
echo.
pause
