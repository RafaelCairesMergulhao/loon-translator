from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class AudioDirection(StrEnum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"


@dataclass(slots=True)
class TranslationRecord:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    direction: AudioDirection = AudioDirection.OUTGOING
    provider: str = ""
    latency_ms: int = 0
    portuguese_text: str = ""
    english_text: str = ""
    stt_ms: int = 0
    translation_ms: int = 0
    speech_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, str | int]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["direction"] = self.direction.value
        return data