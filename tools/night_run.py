"""Ensaio sem microfone: repete frases falsas na pipeline e grava o que quebrar.

Uso, na pasta do projeto:

    .venv\\Scripts\\python.exe -m tools.night_run --rounds 200
    .venv\\Scripts\\python.exe -m tools.night_run --pytest --rounds 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.pipeline import DuplexTranslationPipeline, PipelineOptions
from core.whisper_engine import Transcript
from domain.entities import AudioDirection
from services.audio_devices import AudioDevice
from services.audio_manager import SpeechSegment
from services.lexicon import Lexicon
from services.translation_engine import TranslationResult

LOG_DIR = Path("logs") / "night"


def _device(uid: str, kind: str) -> AudioDevice:
    return AudioDevice(uid=uid, name=uid, kind=kind, channels=2, sample_rate=48_000)


class _Quiet:
    def __init__(self) -> None:
        self.calls = 0
        self.heard = 0

    def synthesize(self, text, *_args, **_kwargs):
        self.calls += 1
        return Path(f"night-{self.calls}.wav")

    def warm_up(self, *_args) -> None:
        return None

    def play(self, *_args, **_kwargs) -> None:
        return None

    @staticmethod
    def cancel() -> None:
        return None

    @staticmethod
    def resume() -> None:
        return None

    def translate(self, text, *_args, **_kwargs):
        return TranslationResult(f"[{text}]", "ensaio")

    def preview(self, text, *_args, **_kwargs):
        return text

    def close(self) -> None:
        return None

    def transcribe(self, audio, language, **_kwargs):
        if audio.size < 8_000:
            raise RuntimeError("trecho curto demais")
        self.heard += 1
        return Transcript(f"frase {self.heard} do ensaio", language or "pt", 0.9)


def soak_once() -> tuple[int, list[str]]:
    """Vinte frases, uma delas impossível de ouvir. A seguinte tem de sair."""
    records: list = []
    errors: list[str] = []
    options = PipelineOptions(
        microphone=_device("sd:1", "microfone"),
        virtual_output=_device("sd:2", "virtual"),
        monitor_output=_device("sd:3", "saída"),
        loopback=_device("wasapi:4", "loopback"),
        outgoing_source="Português",
        outgoing_target="Inglês",
        incoming_source="Inglês",
        incoming_target="Português",
        voice_gender="Feminina",
        translation_mode="Local",
        whisper_model="fast",
        voice_engine="piper",
    )
    quiet = _Quiet()
    pipeline = DuplexTranslationPipeline(
        replace(options),
        on_record=records.append,
        on_status=lambda *_args: None,
        on_error=errors.append,
    )
    pipeline._stt = quiet
    pipeline._translator = quiet
    pipeline._tts = quiet
    pipeline._player = quiet
    pipeline._lexicon = Lexicon()
    feed = []
    for index in range(1, 21):
        size = 4_000 if index == 7 else 16_000
        direction = AudioDirection.INCOMING if index % 2 == 0 else AudioDirection.OUTGOING
        feed.append(
            (
                direction,
                SpeechSegment("final", index, index * 10, np.full(size, 0.1, dtype=np.float32), reason="pause"),
            )
        )
    import asyncio

    asyncio.run(asyncio.wait_for(pipeline.replay(feed), 30))
    return len(records), errors


def write_log(name: str, text: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensaio noturno do Loon sem microfone")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--pytest", action="store_true")
    args = parser.parse_args()
    started = time.strftime("%Y%m%d-%H%M%S")
    lines = [f"início {started}", f"python {sys.version}"]
    code = 0
    if args.pytest:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
        )
        lines.append(completed.stdout)
        lines.append(completed.stderr)
        lines.append(f"pytest {completed.returncode}")
        code = completed.returncode
    for round_index in range(1, args.rounds + 1):
        try:
            spoken, errors = soak_once()
            lines.append(f"rodada {round_index}: frases {spoken}, avisos {len(errors)}")
            if spoken < 19:
                lines.append(f"rodada {round_index} curta: {errors}")
                code = 1
        except Exception:
            lines.append(f"rodada {round_index} quebrou\n{traceback.format_exc()}")
            code = 1
            break
    lines.append(f"fim código {code}")
    path = write_log(f"{started}.log", "\n".join(lines) + "\n")
    print(path)
    print(lines[-1])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
