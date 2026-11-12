"""A configuração que este PC aguenta numa chamada, sem placa inventada."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

from core.whisper_engine import cuda_available
from services.providers.translate import nllb_ready


@dataclass(frozen=True, slots=True)
class PcPlan:
    gpu: bool
    ram_gb: float
    whisper_model: str
    offline_asr: str
    offline_mt: str
    offline_tts: str
    voice_engine: str
    translation_mode: str
    pipeline_profile: str
    live_captions: str


def ram_gb() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024**3)
        except (AttributeError, OSError):
            return 8.0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 8.0
    if page and count:
        return (page * count) / (1024**3)
    return 8.0


def detect_pc() -> PcPlan:
    """Sem NVIDIA, o modelo de placa fica pesado demais. Com memória, cabe o small e o NLLB."""
    gpu = cuda_available()
    ram = ram_gb()
    if gpu:
        whisper, ear = "accurate", ""
    elif ram >= 12:
        whisper, ear = "balanced", "alto"
    else:
        whisper, ear = "fast", ""
    translation = "nllb" if nllb_ready() and (gpu or ram >= 12) else ""
    return PcPlan(
        gpu=gpu,
        ram_gb=ram,
        whisper_model=whisper,
        offline_asr=ear,
        offline_mt=translation,
        offline_tts="",
        voice_engine="piper",
        translation_mode="Local",
        pipeline_profile="",
        live_captions="auto",
    )


def apply_pc_ideal(settings) -> PcPlan:
    """Troca só escuta, tradução e voz. Idioma, microfone e cabo ficam como estão."""
    plan = detect_pc()
    settings.whisper_model = plan.whisper_model
    settings.offline_asr = plan.offline_asr
    settings.offline_mt = plan.offline_mt
    settings.offline_tts = plan.offline_tts
    settings.voice_engine = plan.voice_engine
    settings.translation_mode = plan.translation_mode
    settings.pipeline_profile = plan.pipeline_profile
    settings.live_captions = plan.live_captions
    settings.follow_pc_ideal = True
    return plan


def _rank(name: str, *, gpu: bool) -> int:
    if name in {"accurate", "large-v3-turbo"}:
        return 2
    if name in {"balanced", "small", "alto"}:
        return 1
    if name == "auto":
        return 2 if gpu else 0
    return 0


def deviation_keys(settings) -> list[str]:
    """Chaves de aviso para o que ficou acima ou abaixo do ideal deste PC."""
    plan = detect_pc()
    notes: list[str] = []
    if (settings.pipeline_profile or "") not in {"", "local"} or settings.translation_mode != "Local":
        notes.append("tip_leaves_pc")
    heard = _rank(settings.whisper_model, gpu=plan.gpu)
    ideal = _rank(plan.whisper_model, gpu=plan.gpu)
    if settings.offline_asr == "alto":
        heard = max(heard, 1)
    if heard < ideal:
        notes.append("tip_stt_down")
    elif heard > ideal:
        notes.append("tip_stt_up")
    if settings.offline_asr == "cpp":
        notes.append("tip_asr_cpp")
    chosen_mt = settings.offline_mt or ""
    if chosen_mt == "nllb" and not nllb_ready():
        notes.append("tip_mt_missing")
    elif plan.offline_mt == "nllb" and chosen_mt != "nllb":
        notes.append("tip_mt_down")
    elif plan.offline_mt != "nllb" and chosen_mt == "nllb":
        notes.append("tip_mt_up")
    if settings.offline_tts == "xtts":
        notes.append("tip_voice_xtts")
    elif settings.voice_engine in {"natural", "grok"}:
        notes.append("tip_voice_net")
    elif settings.voice_engine == "windows":
        notes.append("tip_voice_robot")
    return notes


def matches_pc_ideal(settings) -> bool:
    return not deviation_keys(settings)
