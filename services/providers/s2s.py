"""Fala direta: áudio entra, áudio sai, sem passar por texto no meio.

O conector abre a sessão. Se a chave ou a biblioteca não estiver no PC,
devolve vazio e o Loon segue a cascata escuta → tradução → voz.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from services.providers.asr import pcm_wav

logger = logging.getLogger(__name__)

_ENDPOINTS = {
    "openai": "wss://api.openai.com/v1/realtime?model=gpt-realtime",
    "gemini": "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
    "inworld": "wss://api.inworld.ai/v1/realtime",
}


@dataclass(frozen=True, slots=True)
class DirectTurn:
    source_text: str
    translated_text: str
    audio_path: Path
    provider: str


def configured(name: str) -> bool:
    key = (name or "").strip().lower()
    if key == "openai":
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    if key == "gemini":
        return bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip())
    if key == "inworld":
        return bool(os.getenv("INWORLD_API_KEY", "").strip())
    return False


def session_update(source_name: str, target_name: str) -> dict[str, Any]:
    instruction = (
        f"You are a live interpreter. The caller speaks {source_name}. "
        f"Reply only with speech in {target_name}. Do not add commentary."
    )
    return {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "instructions": instruction,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
        },
    }


def translate_turn(
    name: str,
    audio: np.ndarray,
    source_name: str,
    target_name: str,
    output: Path,
) -> DirectTurn | None:
    key = (name or "").strip().lower()
    if key not in _ENDPOINTS or not configured(key):
        return None
    if key != "openai":
        logger.info("Fala direta %s ainda usa a cascata: o socket %s está reservado.", key, _ENDPOINTS[key])
        return None
    try:
        import websockets
    except ImportError:
        logger.info("Fala direta OpenAI precisa do pacote websockets.")
        return None
    import asyncio

    pcm = _pcm16(audio)
    try:
        text, spoken = asyncio.run(_openai_turn(websockets, pcm, source_name, target_name))
    except Exception as exc:
        logger.info("Fala direta indisponível: %s", exc)
        return None
    if not spoken:
        return None
    samples = np.frombuffer(spoken, dtype=np.int16).astype(np.float32) / 32767.0
    output.write_bytes(pcm_wav(samples, rate=24_000))
    return DirectTurn(text or source_name, text or target_name, output, "OpenAI Realtime")


def _pcm16(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


async def _openai_turn(websockets: Any, pcm: bytes, source_name: str, target_name: str) -> tuple[str, bytes]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    url = _ENDPOINTS["openai"]
    transcript = ""
    chunks: list[bytes] = []
    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"},
    ) as socket:
        await socket.send(json.dumps(session_update(source_name, target_name)))
        await socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await socket.send(json.dumps({"type": "response.create"}))
        async for raw in socket:
            message = json.loads(raw)
            kind = message.get("type", "")
            if kind == "response.output_audio.delta" or kind == "response.audio.delta":
                payload = message.get("delta") or ""
                if payload:
                    chunks.append(base64.b64decode(payload))
            if kind in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
                transcript = str(message.get("transcript") or transcript)
            if kind in {"response.done", "error"}:
                break
    return transcript, b"".join(chunks)
