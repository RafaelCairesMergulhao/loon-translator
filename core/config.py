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
    phrase_silence_ms: int = 380
    max_phrase_seconds: float = 12.0
    native_language: str = "Português"
    onboarding_complete: bool = False
    conversation_context: str = "casual"
    voice_pitch: float = 0.0
    speech_rate: float = 1.0
    last_section: str = "translate"
    voice_engine: str = "natural"


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
            if raw.get("phrase_silence_ms") in {600, 650}:
                settings.phrase_silence_ms = 380
            return settings
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
