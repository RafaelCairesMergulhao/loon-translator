"""Prepara o texto para soar como fala, não como leitura de máquina."""

from __future__ import annotations

import html
import re

_COMMA = re.compile(r",\s+")
_SENTENCE = re.compile(r"([.!?…])\s+")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?])")


def spoken_text(text: str) -> str:
    """Limpa espaços, cola pontuação e fecha a frase para o TTS acertar a entoação."""
    cleaned = " ".join(text.split())
    cleaned = _SPACE_PUNCT.sub(r"\1", cleaned)
    if cleaned and cleaned[-1].isalnum():
        cleaned += "."
    return cleaned


def sapi_ssml(text: str, culture: str, speed: float = 1.0) -> str:
    """SSML com pausas nas vírgulas: a SAPI sem isso lê tudo no mesmo tom."""
    body = html.escape(spoken_text(text), quote=False)
    body = _COMMA.sub(', <break time="160ms"/> ', body)
    body = _SENTENCE.sub(r'\1 <break time="240ms"/> ', body)
    rate = max(0.7, min(1.6, float(speed)))
    return (
        f"<speak version='1.0' xml:lang='{html.escape(culture)}'>"
        f"<prosody rate='{rate:.2f}'>{body}</prosody></speak>"
    )


def edge_pitch(semitones: float) -> str:
    """O Edge aceita só Hz; ~12 Hz por semitom na faixa da fala."""
    hz = int(round(max(-6.0, min(6.0, float(semitones))) * 12))
    return f"{hz:+d}Hz"
