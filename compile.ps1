param (
    [string]$Target = "sempro"
)

$file = if ($Target -eq "skripsi") { "main.tex" } else { "mainsempro.tex" }
$pdf = [System.IO.Path]::ChangeExtension($file, ".pdf")

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Mengompilasi $file (XeLaTeX + Biber)..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

latexmk -xelatex -synctex=1 -interaction=nonstopmode $file

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[BERHASIL] $pdf berhasil dibuat!" -ForegroundColor Green
} else {
    Write-Host "`n[GAGAL] Terjadi kesalahan saat kompilasi." -ForegroundColor Red
}
