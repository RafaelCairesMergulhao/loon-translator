"""Provedores trocáveis de escuta, tradução, voz e fala direta.

O caminho padrão do Loon continua Faster-Whisper, o modo escolhido na tela
(Argos, DeepL ou Groq) e a voz já configurada. Um perfil só troca a cadeia.
"""

from services.providers.asr import build_asr
from services.providers.stack import (
    SentenceGate,
    Stack,
    apply_offline,
    resolve_profile,
    translation_mode_for,
    voice_engine_for,
)

__all__ = [
    "SentenceGate",
    "Stack",
    "apply_offline",
    "build_asr",
    "resolve_profile",
    "translation_mode_for",
    "voice_engine_for",
]
