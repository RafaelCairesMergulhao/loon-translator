"""Baixa o binário do Piper e as vozes pt/en para ``vendor/piper``.

Usado pelo instalador. Também pode rodar à mão:

    python -m tools.fetch_piper
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from services.piper_engine import BINARY_URL, HF_VOICE, _download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "vendor" / "piper"
VOICES = (
    "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    "en/en_US/amy/medium/en_US-amy-medium",
    "en/en_US/lessac/medium/en_US-lessac-medium",
)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not any(DEST.rglob("piper.exe")):
        archive = DEST / "piper_windows_amd64.zip"
        print("Baixando piper.exe...")
        _download(BINARY_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(DEST)
        archive.unlink(missing_ok=True)
    voices_dir = DEST / "voices"
    voices_dir.mkdir(exist_ok=True)
    for relative in VOICES:
        name = relative.rsplit("/", 1)[-1]
        onnx = voices_dir / f"{name}.onnx"
        config = voices_dir / f"{name}.onnx.json"
        if onnx.exists() and config.exists():
            print("já tinha", name)
            continue
        print("Baixando voz", name)
        _download(f"{HF_VOICE}/{relative}.onnx", onnx)
        _download(f"{HF_VOICE}/{relative}.onnx.json", config)
    print("Piper pronto em", DEST)


if __name__ == "__main__":
    main()
