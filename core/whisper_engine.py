from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

CALL_PROMPTS = {
    "pt": "Alô, oi, tudo bem? Pode falar.",
    "en": "Hello, hi, how are you? Go ahead.",
    "es": "Hola, ¿qué tal? Adelante.",
    "fr": "Allô, bonjour, comment ça va ?",
    "de": "Hallo, wie geht's? Sprechen Sie.",
    "it": "Pronto, ciao, come va?",
}

# distil-large-v3 foi destilado só com inglês: em português, 7 s por frase e texto errado.
# O Whisper sempre codifica uma janela de 30 s, então o custo por frase quase não
# depende do tamanho da fala: em CPU, base leva ~0,6 s e small ~2 s.
STT_PRESETS = {
    "fast": "base",
    "balanced": "small",
    "accurate": "large-v3-turbo",
}


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str
    probability: float


@dataclass(frozen=True, slots=True)
class ModelPlan:
    name: str
    device: str
    compute_type: str
    threads: int


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_model(choice: str) -> ModelPlan:
    """Traduz a escolha da interface (auto, fast, balanced, accurate) para modelo e hardware."""
    gpu = cuda_available()
    normalized = (choice or "auto").strip()
    if normalized in STT_PRESETS:
        name = STT_PRESETS[normalized]
    elif normalized == "auto":
        name = "large-v3-turbo" if gpu else "base"
    else:
        name = normalized
    if gpu:
        return ModelPlan(name, "cuda", "float16", 0)
    return ModelPlan(name, "cpu", "int8", min(8, os.cpu_count() or 4))


_MODELS: dict[ModelPlan, tuple[WhisperModel, threading.Lock]] = {}
_MODELS_LOCK = threading.Lock()


def _shared_model(plan: ModelPlan) -> tuple[WhisperModel, threading.Lock]:
    """Tradução, treino e testes usam o mesmo modelo carregado uma única vez."""
    with _MODELS_LOCK:
        entry = _MODELS.get(plan)
        if entry is None:
            started = time.perf_counter()
            logger.info(
                "Carregando Faster-Whisper '%s' em %s/%s",
                plan.name,
                plan.device,
                plan.compute_type,
            )
            model = WhisperModel(
                plan.name,
                device=plan.device,
                compute_type=plan.compute_type,
                cpu_threads=plan.threads,
            )
            entry = (model, threading.Lock())
            _MODELS[plan] = entry
            logger.info("Modelo '%s' pronto em %.1fs", plan.name, time.perf_counter() - started)
        return entry


class WhisperEngine:
    def __init__(self, model_size: str = "auto") -> None:
        self.model_size = model_size
        self.plan = resolve_model(model_size)

    def load_model(self) -> WhisperModel:
        return _shared_model(self.plan)[0]

    def warm_up(self) -> float:
        """Primeira inferência aloca memória e compila kernels; melhor antes da primeira frase."""
        started = time.perf_counter()
        model, lock = _shared_model(self.plan)
        noise = (np.random.default_rng(7).standard_normal(16_000) * 0.01).astype(np.float32)
        with lock:
            segments, _info = model.transcribe(
                noise,
                language="en",
                beam_size=1,
                without_timestamps=True,
                vad_filter=False,
            )
            for _segment in segments:
                pass
        return time.perf_counter() - started

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        model, lock = _shared_model(self.plan)
        with lock:
            segments, info = model.transcribe(
                np.asarray(audio, dtype=np.float32),
                language=language,
                beam_size=1,
                without_timestamps=True,
                temperature=0.0,
                condition_on_previous_text=False,
                # Sem idioma definido, um prompt em inglês puxaria a fala para o inglês.
                initial_prompt=CALL_PROMPTS.get(language or ""),
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 150,
                },
                hallucination_silence_threshold=1.5,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        text = WhisperEngine.normalize_speech(text)
        if not WhisperEngine.usable(text):
            text = ""
        return Transcript(
            text=text,
            language=str(info.language or language or ""),
            probability=float(info.language_probability or 0.0),
        )

    @staticmethod
    def normalize_speech(text: str) -> str:
        """Collapses a repeated sentence and drops a Whisper loop."""
        pieces = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+", text.strip())
            if piece.strip()
        ]
        if not pieces:
            return ""
        folded = [piece.casefold() for piece in pieces]
        _phrase, count = Counter(folded).most_common(1)[0]
        if count >= 4 and count / len(folded) >= 0.6:
            return ""
        collapsed: list[str] = []
        previous = ""
        for piece, key in zip(pieces, folded, strict=True):
            if key == previous:
                continue
            collapsed.append(piece)
            previous = key
        words = " ".join(collapsed).split()
        if WhisperEngine._word_loop(words):
            return ""
        if len(words) > 40:
            words = words[:40]
        return " ".join(words)

    @staticmethod
    def _word_loop(words: list[str]) -> bool:
        folded = [word.casefold().strip(".,!?¿¡") for word in words]
        total = len(folded)
        for size in range(1, 5):
            index = 0
            while index + size * 4 <= total:
                pattern = tuple(folded[index : index + size])
                if not any(pattern):
                    index += 1
                    continue
                count = 1
                cursor = index + size
                while cursor + size <= total and tuple(folded[cursor : cursor + size]) == pattern:
                    count += 1
                    cursor += size
                if count >= 4 and (cursor - index) / total >= 0.5:
                    return True
                index += 1
        return False

    @staticmethod
    def usable(text: str) -> bool:
        words = [word for word in text.casefold().split() if word.strip(".,!?")]
        if len(words) < 1:
            return False
        if len(words) >= 4 and len(set(words)) == 1:
            return False
        return True
