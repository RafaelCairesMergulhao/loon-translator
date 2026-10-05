$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"

Set-Location (Split-Path -Parent $PSScriptRoot)

uv sync --extra dev
if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar o ambiente." }

uv run pytest
if ($LASTEXITCODE -ne 0) { throw "Os testes falharam; build cancelado." }

uv run pyinstaller --noconfirm --clean LoonTranslator.spec
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar o executável." }

$releaseDir = Join-Path $PWD "dist\LoonTranslator"
Copy-Item "README.md" $releaseDir -Force
Copy-Item "THIRD_PARTY_NOTICES.md" $releaseDir -Force
Copy-Item "docs\COMMERCIAL_RELEASE.md" $releaseDir -Force

$executable = Join-Path $releaseDir "LoonTranslator.exe"
if ($env:LOON_CERT_THUMBPRINT) {
    $signTool = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
    if (-not $signTool) {
        throw "LOON_CERT_THUMBPRINT definido, mas signtool.exe não foi encontrado."
    }
    & $signTool.Source sign /sha1 $env:LOON_CERT_THUMBPRINT /fd SHA256 `
        /tr http://timestamp.digicert.com /td SHA256 $executable
    if ($LASTEXITCODE -ne 0) { throw "Falha ao assinar o executável." }
}

$hash = Get-FileHash $executable -Algorithm SHA256
$hash | Format-List | Out-File (Join-Path $releaseDir "SHA256.txt") -Encoding utf8

Write-Host ""
Write-Host "Build concluído: $releaseDir"
Write-Host "SHA-256: $($hash.Hash)"
