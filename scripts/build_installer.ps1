# Gera release\LoonTranslator-<versão>-Setup.exe: testes → PyInstaller → Inno Setup.
# O build roda fora do OneDrive: a sincronização trava arquivos da pasta build\ no meio do PyInstaller.
param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = ".\.venv\Scripts\python.exe"
$buildRoot = Join-Path $env:LOCALAPPDATA "LoonBuild"
$appDir = Join-Path $buildRoot "dist\LoonTranslator"

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Write-Host "Instalando o Inno Setup..."
    winget install --id JRSoftware.InnoSetup -e --scope user --silent --accept-package-agreements --accept-source-agreements
    $iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
}

if (-not $SkipTests) {
    & $python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "Os testes falharam; build cancelado." }
}

& $python -m tools.make_icon
& $python -m tools.fetch_piper
if ($LASTEXITCODE -ne 0) { throw "Falha ao baixar o Piper; build cancelado." }
& $python -m PyInstaller --noconfirm --clean --workpath "$buildRoot\work" --distpath "$buildRoot\dist" LoonTranslator.spec
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar o executável." }

if (Get-ChildItem $appDir -Recurse -Filter ".env" -Force) { throw "Um arquivo .env entrou no pacote; build cancelado para não vazar chaves." }
Copy-Item "THIRD_PARTY_NOTICES.md" $appDir -Force

& $iscc "/DSourceDir=$appDir" "installer\LoonTranslator.iss"
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar o instalador." }

$setup = Get-ChildItem "release\LoonTranslator-*-Setup.exe" | Sort-Object LastWriteTime | Select-Object -Last 1
$hash = (Get-FileHash $setup.FullName -Algorithm SHA256).Hash
"$hash  $($setup.Name)" | Out-File "$($setup.FullName).sha256.txt" -Encoding ascii
Write-Host ""
Write-Host ("Instalador: {0} ({1:N0} MB)" -f $setup.FullName, ($setup.Length / 1MB))
Write-Host "SHA-256: $hash"
