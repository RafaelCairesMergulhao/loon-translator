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
    conversation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, str | int]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["direction"] = self.direction.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TranslationRecord:
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, datetime):
            timestamp = raw_ts
        elif isinstance(raw_ts, str) and raw_ts:
            try:
                timestamp = datetime.fromisoformat(raw_ts)
            except ValueError:
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        raw_dir = data.get("direction", AudioDirection.OUTGOING)
        if isinstance(raw_dir, AudioDirection):
            direction = raw_dir
        else:
            try:
                direction = AudioDirection(str(raw_dir))
            except ValueError:
                direction = AudioDirection.OUTGOING
        return cls(
            source_text=str(data.get("source_text", "")),
            translated_text=str(data.get("translated_text", "")),
            source_lang=str(data.get("source_lang", "")),
            target_lang=str(data.get("target_lang", "")),
            direction=direction,
            provider=str(data.get("provider", "")),
            latency_ms=int(data.get("latency_ms") or 0),
            portuguese_text=str(data.get("portuguese_text", "")),
            english_text=str(data.get("english_text", "")),
            stt_ms=int(data.get("stt_ms") or 0),
            translation_ms=int(data.get("translation_ms") or 0),
            speech_ms=int(data.get("speech_ms") or 0),
            conversation=str(data.get("conversation") or ""),
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class VoiceSnapshot:
    gender: str
    rate: float
    pitch: float
    engine: str
    mode: str
    stt: str


@dataclass(frozen=True, slots=True)
class LiveCaption:
    """Legenda provisória enquanto a frase ainda está sendo falada.

    ``stable`` já foi confirmado por duas transcrições seguidas; ``pending``
    ainda pode mudar quando chegarem as próximas palavras.
    """

    direction: AudioDirection
    stable: str
    pending: str
    translation: str = ""

    @property
    def heard(self) -> str:
        return f"{self.stable} {self.pending}".strip()