"""Perfis de chamada: quem ouve, quem traduz e quem fala."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Stack:
    asr: str = "faster-whisper"
    mt: str = "follow"
    tts: str = "follow"
    s2s: str = ""


# follow = obedece o modo e a voz que já estão na tela.
PROFILES: dict[str, Stack] = {
    "": Stack(),
    "local": Stack("faster-whisper", "argos", "piper", ""),
    "hybrid": Stack("faster-whisper", "deepl", "edge", ""),
    "cloud": Stack("deepgram", "groq", "elevenlabs", ""),
    "s2s": Stack("faster-whisper", "follow", "follow", "openai"),
}

MT_MODE = {
    "argos": "Local",
    "deepl": "Nuvem",
    "groq": "IA",
    "openai": "IA",
    "claude": "IA",
}

TTS_ENGINE = {
    "piper": "piper",
    "edge": "natural",
    "elevenlabs": "elevenlabs",
    "xtts": "xtts",
}


def resolve_profile(name: str | None) -> Stack:
    key = (name or "").strip().lower()
    return PROFILES.get(key, PROFILES[""])


def apply_offline(
    profile: str | None,
    mode: str,
    whisper: str,
    asr: str | None,
    mt: str | None,
    tts: str | None,
) -> tuple[Stack, str]:
    """No modo local, troca só o que foi pedido. Nuvem e híbrido ficam como estão.

    A escuta alta usa o modelo small em INT8. Na CPU ele é mais fiel que o base
    e também mais pesado: não promete o mesmo tempo de frase.
    """
    stack = resolve_profile(profile)
    model = whisper or "auto"
    local = (profile or "") == "local" or (not (profile or "").strip() and mode == "Local")
    if not local:
        return stack, model
    asr_key = (asr or "").strip().lower()
    mt_key = (mt or "").strip().lower()
    tts_key = (tts or "").strip().lower()
    asr_name = stack.asr
    if asr_key == "cpp":
        asr_name = "whisper.cpp"
    elif asr_key in {"leve", "alto"}:
        asr_name = "faster-whisper"
    if asr_key == "alto" and model in {"", "auto", "fast", "base"}:
        model = "balanced"
    mt_name = stack.mt
    if mt_key == "nllb":
        mt_name = "nllb"
    elif mt_key == "argos":
        mt_name = "argos"
    tts_name = stack.tts
    if tts_key == "xtts":
        tts_name = "xtts"
    elif tts_key == "piper":
        tts_name = "piper"
    return Stack(asr_name, mt_name, tts_name, ""), model


def translation_mode_for(stack: Stack, selected_mode: str) -> str:
    return MT_MODE.get(stack.mt, selected_mode)


def voice_engine_for(stack: Stack, selected_engine: str) -> str:
    return TTS_ENGINE.get(stack.tts, selected_engine)


class SentenceGate:
    """Legenda pode sair no meio da frase. A voz espera o silêncio fechar."""

    def caption(self, kind: str) -> bool:
        return kind in {"partial", "pause", "final"}

    def speak(self, kind: str) -> bool:
        return kind == "final"
