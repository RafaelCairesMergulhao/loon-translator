"""Escuta: Faster-Whisper no PC, ou uma frase fechada na nuvem.

O segmentador do Loon já espera o silêncio. Deepgram, AssemblyAI e Watson
recebem essa frase inteira. A legenda parcial continua no Whisper local
quando o perfil de nuvem não está ligado.
"""

from __future__ import annotations

import io
import json
import logging
import os
import wave
from typing import Any

import httpx
import numpy as np

from core.whisper_engine import Transcript, WhisperEngine

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


def pcm_wav(audio: np.ndarray, rate: int = 16_000) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _transcript(text: str, language: str | None, probability: float = 0.0) -> Transcript:
    return Transcript((text or "").strip(), (language or "").split("-")[0].lower(), float(probability))


class FasterWhisperASR(WhisperEngine):
    name = "faster-whisper"


class WhisperCppASR:
    """whisper.cpp, se WHISPER_CPP_BIN apontar para o executável."""

    name = "whisper.cpp"

    def __init__(self, model_size: str = "auto") -> None:
        del model_size
        self.binary = os.getenv("WHISPER_CPP_BIN", "").strip()
        self.model = os.getenv("WHISPER_CPP_MODEL", "").strip()

    def warm_up(self) -> float:
        if not self.binary:
            raise ProviderError("Defina WHISPER_CPP_BIN para usar o whisper.cpp.")
        return 1.0

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: Any = None, hotwords: Any = None) -> Transcript:
        del prompt, hotwords
        if not self.binary or not self.model:
            raise ProviderError("WHISPER_CPP_BIN e WHISPER_CPP_MODEL são obrigatórios.")
        import subprocess
        import tempfile

        wav = pcm_wav(audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav)
            path = handle.name
        command = [self.binary, "-m", self.model, "-f", path, "-nt", "--no-timestamps"]
        if language:
            command.extend(["-l", language])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        os.unlink(path)
        if completed.returncode != 0:
            raise ProviderError(completed.stderr.strip() or "whisper.cpp falhou")
        return _transcript(completed.stdout, language)


class _CloudUtterance:
    name = "cloud"

    def warm_up(self) -> float:
        return 2.0

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: Any = None, hotwords: Any = None) -> Transcript:
        del prompt, hotwords
        return self._recognize(pcm_wav(audio), language)


class DeepgramASR(_CloudUtterance):
    """Nova-3. A frase já vem cortada pelo VAD local; o Deepgram devolve o texto."""

    name = "deepgram"

    def _recognize(self, wav: bytes, language: str | None) -> Transcript:
        key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not key:
            raise ProviderError("Defina DEEPGRAM_API_KEY para a escuta Deepgram.")
        params = {"model": os.getenv("DEEPGRAM_MODEL", "nova-3"), "smart_format": "true", "punctuate": "true"}
        if language:
            params["language"] = language
        else:
            params["detect_language"] = "true"
        response = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            content=wav,
            timeout=20,
        )
        response.raise_for_status()
        return parse_deepgram(response.json(), language)


def parse_deepgram(payload: dict[str, Any], language: str | None) -> Transcript:
    channels = payload.get("results", {}).get("channels") or [{}]
    alternatives = channels[0].get("alternatives") or [{}]
    best = alternatives[0]
    detected = channels[0].get("detected_language") or language or ""
    return _transcript(str(best.get("transcript") or ""), str(detected), float(best.get("confidence") or 0))


class AssemblyAIASR(_CloudUtterance):
    """Universal Streaming. Uma frase PCM vai pelo WebSocket e volta como texto final."""

    name = "assemblyai"

    def _recognize(self, wav: bytes, language: str | None) -> Transcript:
        key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not key:
            raise ProviderError("Defina ASSEMBLYAI_API_KEY para a escuta AssemblyAI.")
        try:
            import websockets
        except ImportError as exc:
            raise ProviderError("Instale o pacote websockets para a escuta AssemblyAI.") from exc
        import asyncio

        pcm = _wav_pcm(wav)
        text = asyncio.run(_assembly_turn(websockets, key, pcm, language))
        return _transcript(text, language)


async def _assembly_turn(websockets: Any, key: str, pcm: bytes, language: str | None) -> str:
    query = "sample_rate=16000&speech_model=universal-streaming-multilingual"
    if language:
        query += f"&language_code={language}"
    url = f"wss://streaming.assemblyai.com/v3/ws?{query}"
    pieces: list[str] = []
    async with websockets.connect(url, additional_headers={"Authorization": key}) as socket:
        await socket.send(pcm)
        await socket.send(json.dumps({"type": "Terminate"}))
        async for raw in socket:
            message = json.loads(raw)
            if message.get("type") == "Turn" and message.get("transcript") and message.get("end_of_turn"):
                pieces.append(str(message["transcript"]))
            if message.get("type") in {"Termination", "Error"}:
                break
    return " ".join(pieces).strip()


class WatsonASR(_CloudUtterance):
    """Speech to Text da IBM. Tradução não vem daqui: o Watson Translator foi retirado em 2024."""

    name = "watson"
    _token = ""

    def _recognize(self, wav: bytes, language: str | None) -> Transcript:
        url = os.getenv("WATSON_STT_URL", "").strip().rstrip("/")
        apikey = os.getenv("WATSON_STT_APIKEY", "").strip()
        if not url or not apikey:
            raise ProviderError("Defina WATSON_STT_URL e WATSON_STT_APIKEY.")
        model = os.getenv("WATSON_STT_MODEL", "").strip()
        if not model:
            model = "pt-BR_Multimedia" if (language or "pt").startswith("pt") else "en-US_Multimedia"
        response = httpx.post(
            f"{url}/v1/recognize",
            params={"model": model},
            headers={"Authorization": f"Bearer {self._iam_token(apikey)}", "Content-Type": "audio/wav"},
            content=wav,
            timeout=20,
        )
        response.raise_for_status()
        return parse_watson(response.json(), language)

    @classmethod
    def _iam_token(cls, apikey: str) -> str:
        if cls._token:
            return cls._token
        response = httpx.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": apikey},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        response.raise_for_status()
        cls._token = str(response.json().get("access_token") or "")
        if not cls._token:
            raise ProviderError("A IBM não devolveu um token de acesso.")
        return cls._token


def parse_watson(payload: dict[str, Any], language: str | None) -> Transcript:
    results = payload.get("results") or []
    text = " ".join(str(item.get("alternatives", [{}])[0].get("transcript") or "") for item in results)
    confidence = 0.0
    if results:
        confidence = float(results[0].get("alternatives", [{}])[0].get("confidence") or 0)
    return _transcript(text, language, confidence)


def _wav_pcm(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav), "rb") as handle:
        return handle.readframes(handle.getnframes())


def build_asr(name: str, model_size: str = "auto") -> Any:
    key = (name or "faster-whisper").strip().lower()
    if key in {"", "faster-whisper", "whisper"}:
        return FasterWhisperASR(model_size)
    if key in {"whisper.cpp", "whispercpp"}:
        return WhisperCppASR(model_size)
    if key == "deepgram":
        return DeepgramASR()
    if key in {"assemblyai", "assembly"}:
        return AssemblyAIASR()
    if key in {"watson", "ibm"}:
        return WatsonASR()
    raise ProviderError(f"Escuta desconhecida: {name}")
