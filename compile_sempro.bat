@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Mengompilasi Seminar Proposal (mainsempro.tex)...
echo ========================================================
latexmk -xelatex -synctex=1 -interaction=nonstopmode mainsempro.tex

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   [BERHASIL] mainsempro.pdf berhasil dibuat!
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
