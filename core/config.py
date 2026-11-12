from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


APP_DIR = Path(os.getenv("APPDATA", Path.home())) / "LoonTranslator"
USER_ENV = APP_DIR / ".env"


def load_environment() -> None:
    """O .env da pasta de trabalho (desenvolvimento) vence o do usuário (%APPDATA%, app instalado)."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    load_dotenv(USER_ENV)


def save_user_env(values: dict[str, str]) -> None:
    from dotenv import set_key

    USER_ENV.parent.mkdir(parents=True, exist_ok=True)
    USER_ENV.touch(exist_ok=True)
    for key, value in values.items():
        set_key(str(USER_ENV), key, value, quote_mode="never")
        os.environ[key] = value


@dataclass(slots=True)
class AppSettings:
    source_language: str = "Auto"
    incoming_source_language: str = "Auto"
    outgoing_language: str = "Inglês"
    incoming_language: str = "Português"
    voice_gender: str = "Feminina"
    translation_mode: str = "Local"
    microphone_device: str = ""
    virtual_output_device: str = ""
    monitor_output_device: str = ""
    loopback_device: str = ""
    whisper_model: str = "auto"
    save_history: bool = False
    monitor_outgoing: bool = False
    cloud_consent: bool = False
    phrase_silence_ms: int = 280
    max_phrase_seconds: float = 12.0
    native_language: str = "Português"
    onboarding_complete: bool = False
    conversation_context: str = "casual"
    voice_pitch: float = 0.0
    speech_rate: float = 1.0
    last_section: str = "translate"
    voice_engine: str = "piper"
    live_captions: str = "auto"
    appearance: str = "night"
    llm_provider: str = "groq"
    pipeline_profile: str = ""
    offline_asr: str = ""
    offline_mt: str = ""
    offline_tts: str = ""
    follow_pc_ideal: bool = True
    settings_version: int = 3


_WHISPER = {"auto", "fast", "balanced", "accurate", "base", "small", "large-v3-turbo"}
_MODES = {"Local", "Híbrido", "Nuvem", "IA"}
_VOICES = {"piper", "natural", "windows", "grok"}
_CAPTIONS = {"auto", "off", "on"}
_PROFILES = {"", "local", "hybrid", "cloud", "s2s"}
_EARS = {"", "alto", "cpp", "leve"}
_TRANSLATORS = {"", "nllb", "argos"}
_SPEAKERS = {"", "piper", "xtts"}
_LOOKS = {"night", "paper"}


def _number(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _repair(settings: AppSettings) -> AppSettings:
    """Um settings.json de outro PC ou editado à mão não pode escolher um valor que a tela não tem."""
    if settings.whisper_model not in _WHISPER:
        settings.whisper_model = "auto"
    if settings.translation_mode not in _MODES:
        settings.translation_mode = "Local"
    if settings.voice_engine not in _VOICES:
        settings.voice_engine = "piper"
    if settings.live_captions not in _CAPTIONS:
        settings.live_captions = "auto"
    if settings.pipeline_profile not in _PROFILES:
        settings.pipeline_profile = ""
    if settings.offline_asr not in _EARS:
        settings.offline_asr = ""
    if settings.offline_mt not in _TRANSLATORS:
        settings.offline_mt = ""
    if settings.offline_tts not in _SPEAKERS:
        settings.offline_tts = ""
    if settings.appearance not in _LOOKS:
        settings.appearance = "night"
    if not isinstance(settings.follow_pc_ideal, bool):
        settings.follow_pc_ideal = True
    settings.phrase_silence_ms = max(120, min(2000, int(_number(settings.phrase_silence_ms, 280))))
    settings.max_phrase_seconds = max(3.0, min(30.0, _number(settings.max_phrase_seconds, 12.0)))
    settings.speech_rate = max(0.5, min(2.0, _number(settings.speech_rate, 1.0)))
    settings.voice_pitch = max(-12.0, min(12.0, _number(settings.voice_pitch, 0.0)))
    return settings


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or APP_DIR / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            allowed = {item.name for item in fields(AppSettings)}
            settings = AppSettings(**{key: value for key, value in raw.items() if key in allowed})
            # Instalações anteriores já passaram da primeira abertura.
            if "onboarding_complete" not in raw:
                settings.onboarding_complete = True
            # distil-large-v3 só reconhece inglês; o modo automático escolhe pelo hardware.
            if raw.get("whisper_model") in {"base", "distil-large-v3"}:
                settings.whisper_model = "auto"
            if raw.get("phrase_silence_ms") in {600, 650, 380} and raw.get("settings_version", 1) < 3:
                settings.phrase_silence_ms = 280
                settings.settings_version = 3
            # Até a 0.5.0 o Edge (~2 s por frase) era o padrão. A troca pelo Piper vale uma
            # vez só: quem voltar ao Edge depois fica com ele.
            if raw.get("settings_version", 1) < 2 and raw.get("voice_engine") == "natural":
                settings.voice_engine = "piper"
            return _repair(settings)
        except (OSError, ValueError, TypeError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
