"""Vozes de nuvem e clone local, gravadas num arquivo que o player já sabe tocar."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class SpeechError(RuntimeError):
    pass


def elevenlabs_speak(text: str, language_name: str, gender: str, output: Path, speed: float) -> Path:
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise SpeechError("Defina ELEVENLABS_API_KEY para a voz ElevenLabs.")
    if output.exists() and output.stat().st_size > 0:
        return output
    voice = os.getenv("ELEVENLABS_VOICE_ID", "").strip() or (
        "EXAVITQu4vr4xnSDxMaL" if gender == "Feminina" else "VR6AewLTigWG4xSOukaG"
    )
    model = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    response = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.7, "speed": max(0.7, min(1.2, speed))},
        },
        timeout=20,
    )
    response.raise_for_status()
    if not response.content:
        raise SpeechError("A ElevenLabs não devolveu áudio.")
    output.write_bytes(response.content)
    logger.info("Voz ElevenLabs em %s (%s)", language_name, model)
    return output


def xtts_speak(text: str, language_name: str, output: Path, speed: float) -> Path:
    del speed
    if output.exists() and output.stat().st_size > 0:
        return output
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise SpeechError("Instale o pacote TTS para o Coqui XTTS v2.") from exc
    codes = {
        "Português": "pt",
        "Inglês": "en",
        "Espanhol": "es",
        "Francês": "fr",
        "Alemão": "de",
        "Italiano": "it",
        "Japonês": "ja",
        "Chinês (Mandarim)": "zh-cn",
        "Russo": "ru",
        "Coreano": "ko",
        "Hindi": "hi",
    }
    language = codes.get(language_name, "en")
    speaker = os.getenv("XTTS_SPEAKER_WAV", "").strip()
    engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if speaker:
        engine.tts_to_file(text=text, file_path=str(output), speaker_wav=speaker, language=language)
    else:
        engine.tts_to_file(text=text, file_path=str(output), language=language)
    if not output.exists() or output.stat().st_size == 0:
        raise SpeechError("O XTTS não gerou áudio.")
    return output
