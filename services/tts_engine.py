from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf

from core.config import APP_DIR
from domain.sorting import nlargest
from services.audio_devices import AudioDeviceManager
from services.piper_engine import PiperEngine, PiperError
from services.pitch_shift import shift_pitch
from services.sapi_worker import SapiWorker
from services.speech_prosody import edge_pitch, spoken_text
from services.translation_engine import VOICE_MAP

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?;。！？；])\s+|(?<=[。！？；])")


class TTSError(RuntimeError):
    pass


def split_for_speech(text: str, min_chars: int = 24, max_chars: int = 160) -> list[str]:
    """Divide em frases para a primeira começar a tocar enquanto as seguintes são sintetizadas.

    Pedaços curtos demais se juntam ao seguinte: cada síntese tem custo fixo.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars and not _SENTENCE_END.search(text[:-1]):
        return [text] if text else []
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text) if piece and piece.strip()]
    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(",", 0, max_chars)
            if cut < min_chars:
                cut = piece.rfind(" ", 0, max_chars)
            if cut < min_chars:
                cut = max_chars
            chunks.append(piece[: cut + 1].strip())
            piece = piece[cut + 1 :].strip()
        if not piece:
            continue
        if chunks and len(chunks[-1]) < min_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def trim_silence(samples: np.ndarray, rate: int, threshold: float = 0.005, pad_ms: int = 40) -> np.ndarray:
    """Remove o silêncio que SAPI, Edge e Piper colocam antes e depois da fala."""
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return data
    level = np.max(np.abs(data), axis=1) if data.ndim == 2 else np.abs(data)
    loud = np.flatnonzero(level > threshold)
    if loud.size == 0:
        return data
    pad = int(rate * pad_ms / 1000)
    start = max(0, int(loud[0]) - pad)
    end = min(level.size, int(loud[-1]) + pad + 1)
    return data[start:end]


def edge_rate(speed: float) -> str:
    return f"{round((max(0.5, min(2.0, speed)) - 1.0) * 100):+d}%"


def installed_voice_prefixes() -> set[str]:
    """Duas letras de cada voz do Windows, por exemplo pt e en."""
    if os.name != "nt":
        return set()
    script = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
    if ($_.Enabled) { $_.VoiceInfo.Culture.Name.Substring(0, 2).ToLower() }
}
"""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if len(line.strip()) == 2}


class TTSEngine:
    SAPI_CULTURES = {
        "Português": "pt-BR",
        "Inglês": "en-US",
        "Espanhol": "es-ES",
        "Espanhol (Latino)": "es-MX",
        "Francês": "fr-FR",
        "Alemão": "de-DE",
        "Italiano": "it-IT",
        "Hindi": "hi-IN",
        "Árabe": "ar-SA",
        "Japonês": "ja-JP",
        "Chinês (Mandarim)": "zh-CN",
        "Russo": "ru-RU",
        "Coreano": "ko-KR",
    }

    @classmethod
    def missing_spoken_languages(cls, language_names: list[str]) -> list[str]:
        prefixes = installed_voice_prefixes()
        if not prefixes:
            return []
        missing: list[str] = []
        for name in language_names:
            culture = cls.SAPI_CULTURES.get(name, "")
            prefix = culture.split("-")[0].lower()
            if prefix and prefix not in prefixes and name not in missing:
                missing.append(name)
        return missing

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or APP_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prune_cache()

    def synthesize(
        self,
        text: str,
        language_name: str,
        gender: str,
        mode: str,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        voice_engine: str = "natural",
    ) -> Path:
        text = spoken_text(text)
        if len(text) > 500:
            text = text[:500].rsplit(" ", 1)[0]
        speed = round(max(0.5, min(2.0, float(speed))), 2)
        pitch = round(max(-6.0, min(6.0, float(pitch_semitones))), 1)
        engine = voice_engine if voice_engine in {"natural", "piper", "windows"} else "natural"
        voice = VOICE_MAP.get(language_name, VOICE_MAP["Inglês"]).get(
            gender, VOICE_MAP["Inglês"]["Feminina"]
        )
        key = hashlib.sha256(
            f"{engine}|{voice}|{language_name}|{gender}|{speed}|{pitch}|{text}".encode("utf-8")
        ).hexdigest()
        edge_file = self.cache_dir / f"{key}.neural.mp3"
        piper_file = self.cache_dir / f"{key}.piper.wav"
        sapi_file = self.cache_dir / f"{key}.sapi.wav"

        if engine == "piper":
            try:
                return self._piper(text, language_name, gender, piper_file, speed, pitch)
            except Exception as piper_error:
                logger.info("Piper indisponível; tentando voz neural online: %s", piper_error)
            try:
                return self._edge(text, voice, edge_file, speed, pitch)
            except Exception as edge_error:
                logger.info("Voz neural indisponível: %s", edge_error)
            return self._windows_or_fail(text, language_name, gender, sapi_file, speed, mode)

        if engine == "natural":
            try:
                return self._edge(text, voice, edge_file, speed, pitch)
            except Exception as edge_error:
                logger.info("Voz neural indisponível; tentando Piper local: %s", edge_error)
            try:
                return self._piper(text, language_name, gender, piper_file, speed, pitch)
            except Exception as piper_error:
                logger.info("Piper indisponível: %s", piper_error)
            return self._windows_or_fail(text, language_name, gender, sapi_file, speed, mode)

        try:
            return self._windows_sapi(text, language_name, gender, sapi_file, speed)
        except Exception as sapi_error:
            logger.info("Voz SAPI local indisponível: %s", sapi_error)
        try:
            return self._piper(text, language_name, gender, piper_file, speed, pitch)
        except Exception as piper_error:
            if mode == "Local":
                raise TTSError(f"missing_voice:{language_name}") from piper_error
            logger.info("Piper indisponível; usando voz neural: %s", piper_error)
        return self._edge(text, voice, edge_file, speed, pitch)

    def _windows_or_fail(
        self,
        text: str,
        language_name: str,
        gender: str,
        sapi_file: Path,
        speed: float,
        mode: str,
    ) -> Path:
        try:
            return self._windows_sapi(text, language_name, gender, sapi_file, speed)
        except Exception as sapi_error:
            if mode == "Local":
                raise TTSError(f"missing_voice:{language_name}") from sapi_error
            raise TTSError("O serviço de voz não respondeu.") from sapi_error

    def _edge(
        self,
        text: str,
        voice: str,
        output: Path,
        speed: float,
        pitch_semitones: float,
    ) -> Path:
        if output.exists() and output.stat().st_size > 0:
            return output
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                asyncio.run(
                    edge_tts.Communicate(
                        text,
                        voice,
                        rate=edge_rate(speed),
                        pitch=edge_pitch(pitch_semitones),
                        volume="+4%",
                    ).save(str(output))
                )
                if output.exists() and output.stat().st_size > 0:
                    return output
            except Exception as exc:
                last_error = exc
                logger.warning("Edge TTS falhou na tentativa %d: %s", attempt + 1, exc)
                output.unlink(missing_ok=True)
        raise TTSError("O serviço de voz não respondeu.") from last_error

    def _piper(
        self,
        text: str,
        language_name: str,
        gender: str,
        output: Path,
        speed: float,
        pitch_semitones: float,
    ) -> Path:
        try:
            return PiperEngine().synthesize(text, language_name, gender, output, speed, pitch_semitones)
        except PiperError as exc:
            raise TTSError(str(exc)) from exc

    @classmethod
    def _windows_sapi(
        cls,
        text: str,
        language_name: str,
        gender: str,
        output: Path,
        speed: float = 1.0,
    ) -> Path:
        if os.name != "nt":
            raise TTSError("SAPI está disponível apenas no Windows")
        if output.exists() and output.stat().st_size > 0:
            return output
        culture = cls.SAPI_CULTURES.get(language_name)
        if not culture:
            raise TTSError(f"Cultura SAPI desconhecida: {language_name}")
        sapi_gender = "Female" if gender == "Feminina" else "Male"
        try:
            return SapiWorker.shared().synthesize(text, culture, sapi_gender, output, speed)
        except Exception as exc:
            output.unlink(missing_ok=True)
            raise TTSError(str(exc)) from exc

    @staticmethod
    def warm_up(
        voice_engine: str = "natural",
        language_name: str = "Inglês",
        gender: str = "Feminina",
    ) -> None:
        if os.name == "nt":
            SapiWorker.shared().ensure_started()
        if voice_engine != "piper":
            return
        try:
            engine = PiperEngine()
            engine.ensure_binary()
            engine.ensure_model(language_name, gender)
        except Exception as exc:
            logger.info("Aquecimento do Piper ignorado: %s", exc)

    @staticmethod
    def prune_cache(max_files: int = 100) -> None:
        """Mantém os ``max_files`` mais recentes; o heap evita ordenar o cache inteiro."""
        cache_dir = APP_DIR / "cache"
        if not cache_dir.exists():
            return
        files = [(item, item.stat().st_mtime) for item in cache_dir.iterdir() if item.is_file()]
        keep = {path for path, _mtime in nlargest(files, max_files, key=lambda entry: entry[1])}
        for path, _mtime in files:
            if path not in keep:
                path.unlink(missing_ok=True)


class AudioPlayer:
    _cancel = threading.Event()

    @classmethod
    def cancel(cls) -> None:
        cls._cancel.set()

    @classmethod
    def resume(cls) -> None:
        cls._cancel.clear()

    @staticmethod
    def play(path: Path, device_uid: str, pitch_semitones: float = 0.0) -> None:
        data, source_rate = sf.read(path, dtype="float32", always_2d=True)
        data = trim_silence(data, source_rate)
        AudioPlayer.play_samples(
            data,
            source_rate,
            device_uid,
            pitch_semitones=pitch_semitones,
        )

    @staticmethod
    def play_samples(
        data: np.ndarray,
        source_rate: int,
        device_uid: str,
        pitch_semitones: float = 0.0,
    ) -> None:
        """Reproduz por stream próprio; chamadas paralelas não se interrompem."""
        samples = np.asarray(data, dtype=np.float32)
        if abs(pitch_semitones) >= 0.05:
            samples = shift_pitch(samples, pitch_semitones)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        AudioPlayer._play_with_fallback(
            np.ascontiguousarray(samples),
            source_rate,
            device_uid,
            pitch_semitones,
        )

    @staticmethod
    def _play_with_fallback(
        data: np.ndarray,
        source_rate: int,
        device_uid: str,
        pitch_semitones: float,
    ) -> None:
        del pitch_semitones
        uids = [device_uid]
        last_error: Exception | None = None
        step = 0
        while step < len(uids) and step < 5:
            uid = uids[step]
            try:
                AudioPlayer._play_on_device(data, source_rate, uid)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("Falha ao tocar em %s: %s", uid, exc)
                if step == 0:
                    time.sleep(0.45)
                    uids.append(device_uid)
                elif step == 1:
                    try:
                        extras = AudioDeviceManager().playback_fallbacks(
                            device_uid,
                            AudioDeviceManager().list_outputs(),
                        )
                    except Exception:
                        extras = []
                    for extra in extras:
                        if extra not in uids:
                            uids.append(extra)
            step += 1
        if last_error is not None:
            raise last_error

    @staticmethod
    def _play_on_device(data: np.ndarray, source_rate: int, device_uid: str) -> None:
        index = AudioDeviceManager.index_from_uid(device_uid, "sd")
        device = sd.query_devices(index)
        target_rate = int(device["default_samplerate"])
        samples = np.asarray(data, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        max_channels = max(1, int(device["max_output_channels"]))
        channels = min(samples.shape[1], max_channels)
        if samples.shape[1] != channels:
            samples = samples[:, :channels]
        if source_rate != target_rate and samples.shape[0] > 1:
            target_size = max(1, round(samples.shape[0] * target_rate / source_rate))
            old_positions = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
            new_positions = np.linspace(0.0, 1.0, target_size, endpoint=False)
            samples = np.column_stack(
                [
                    np.interp(new_positions, old_positions, samples[:, channel])
                    for channel in range(samples.shape[1])
                ]
            ).astype(np.float32)
        sd.check_output_settings(
            device=index,
            channels=channels,
            dtype="float32",
            samplerate=target_rate,
        )
        AudioPlayer._play_callback(np.ascontiguousarray(samples), target_rate, index, channels)

    @staticmethod
    def _play_callback(
        samples: np.ndarray,
        sample_rate: int,
        device_index: int,
        channels: int,
    ) -> None:
        """O WDM-KS do Windows recusa stream.write(); o callback abre o mesmo fone."""
        if samples.size == 0:
            return
        finished = threading.Event()
        position = 0

        def callback(outdata, frames, _time_info, status) -> None:
            nonlocal position
            if AudioPlayer._cancel.is_set():
                outdata[:] = 0
                raise sd.CallbackStop()
            if status:
                logger.debug("Status da reprodução: %s", status)
            available = samples.shape[0] - position
            count = min(frames, max(0, available))
            if count:
                outdata[:count] = samples[position : position + count]
            if count < frames:
                outdata[count:] = 0
                position += count
                raise sd.CallbackStop()
            position += count

        with sd.OutputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            latency="high",
            callback=callback,
            finished_callback=finished.set,
        ):
            deadline = time.monotonic() + samples.shape[0] / sample_rate + 3
            while not finished.is_set():
                if AudioPlayer._cancel.is_set() or time.monotonic() >= deadline:
                    break
                finished.wait(timeout=0.2)
            if not finished.is_set() and not AudioPlayer._cancel.is_set():
                raise RuntimeError("A reprodução de áudio não terminou.")
