"""Mede cada etapa do caminho fala → texto → tradução → voz com os motores reais.

Uso:
    python -m tools.benchmark_latency
    python -m tools.benchmark_latency --models base small --rounds 3 --legacy

A frase de teste é gerada pela voz do Windows e transcrita pelo Whisper, então
o resultado vale para esta máquina. Nada é tocado nos alto-falantes.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from core.whisper_engine import WhisperEngine, resolve_model
from services.audio_manager import resample_mono
from services.sapi_worker import SapiWorker
from services.translation_engine import TranslationEngine
from services.tts_engine import TTSEngine, split_for_speech

PHRASE = "Olá, tudo bem? Amanhã eu vou chegar um pouco mais tarde na reunião, pode começar sem mim."

_LEGACY_SAPI = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name.StartsWith('en') } | Select-Object -First 1
$synth.SelectVoice($voice.VoiceInfo.Name)
$synth.SetOutputToWaveFile($env:LOON_OUT)
$synth.Speak([Console]::In.ReadToEnd())
$synth.Dispose()
"""


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):7.0f} ms"


def _sample(directory: Path) -> np.ndarray:
    path = directory / "frase.wav"
    SapiWorker.shared().synthesize(PHRASE, "pt-BR", "Female", path)
    audio, rate = sf.read(path, dtype="float32")
    return resample_mono(audio, rate, 16_000)


def _legacy_sapi(text: str, output: Path) -> None:
    import os

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _LEGACY_SAPI],
        input=text,
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "LOON_OUT": str(output)},
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=["auto"], help="auto, fast, balanced, accurate ou nome do modelo")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mode", default="Local", choices=["Local", "Híbrido", "Nuvem", "IA"])
    parser.add_argument("--legacy", action="store_true", help="inclui distil-large-v3 e SAPI com PowerShell novo por frase")
    args = parser.parse_args()
    load_dotenv()

    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        print("Gerando a frase de teste com a voz do Windows...")
        audio = _sample(directory)
        print(f"Frase: {PHRASE!r} ({audio.size / 16_000:.1f}s de áudio)\n")

        models = list(args.models) + (["distil-large-v3"] if args.legacy else [])
        print("Reconhecimento de fala (Faster-Whisper, após aquecimento)")
        transcript = PHRASE
        for choice in models:
            engine = WhisperEngine(choice)
            plan = resolve_model(choice)
            started = time.perf_counter()
            engine.warm_up()
            load_ms = _ms(started)
            timings = []
            for _ in range(args.rounds):
                started = time.perf_counter()
                result = engine.transcribe(audio, "pt")
                timings.append(_ms(started))
            if choice == models[0]:
                transcript = result.text or PHRASE
            print(f"  {choice:>16} -> {plan.name:<15} {plan.device}/{plan.compute_type:<8}"
                  f" carga {load_ms:6.0f} ms  mediana {_median(timings)}  \"{result.text}\"")

        print(f"\nTradução pt -> en (modo {args.mode})")
        translator = TranslationEngine()
        translator.warm_up([("pt", "Inglês")], args.mode)
        timings = []
        for _ in range(args.rounds):
            translator._cache.clear()
            started = time.perf_counter()
            translated = translator.translate(transcript, "pt", "Inglês", args.mode)
            timings.append(_ms(started))
        print(f"  {translated.provider:<30} mediana {_median(timings)}  \"{translated.text}\"")
        started = time.perf_counter()
        translator.translate(transcript, "pt", "Inglês", args.mode)
        print(f"  {'mesma frase (cache LRU)':<30}         {_ms(started):7.2f} ms")

        print("\nSíntese de voz (Windows SAPI), por frase")
        TTSEngine.warm_up()
        sentences = split_for_speech(translated.text)
        worker_ms = []
        for index in range(args.rounds):
            started = time.perf_counter()
            SapiWorker.shared().synthesize(sentences[0], "en-US", "Female", directory / f"w{index}.wav", 1.1)
            worker_ms.append(_ms(started))
        print(f"  {'processo persistente (novo)':<30} mediana {_median(worker_ms)}")
        if args.legacy:
            legacy_ms = []
            for index in range(args.rounds):
                started = time.perf_counter()
                _legacy_sapi(sentences[0], directory / f"l{index}.wav")
                legacy_ms.append(_ms(started))
            print(f"  {'PowerShell novo por frase (antigo)':<30} mediana {_median(legacy_ms)}")
        whole = directory / "inteira.wav"
        SapiWorker.shared().synthesize(translated.text, "en-US", "Female", whole, 1.1)
        data, rate = sf.read(whole, dtype="float32")
        print(f"  {len(sentences)} trecho(s); a voz começa após o 1º trecho, não após os {data.shape[0] / rate:.1f}s da resposta inteira")
    SapiWorker.shared().close()


if __name__ == "__main__":
    main()
