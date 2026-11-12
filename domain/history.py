from __future__ import annotations

from datetime import UTC, datetime

from domain.entities import TranslationRecord


LEGACY_CONVERSATION = "teste 1"

_DATE_ONLY = ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y")
_WITH_TIME = ("%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%d.%m.%Y %H:%M")


def suggest_conversation_name(existing: list[str], template: str) -> str:
    """Próximo nome livre. ``template`` precisa de ``{number}``."""
    taken = {name.casefold() for name in existing}
    number = 1
    while number < 10000:
        candidate = template.format(number=number)
        if candidate.casefold() not in taken:
            return candidate
        number += 1
    return template.format(number=number)


def parse_history_bound(text: str, *, end: bool) -> tuple[datetime | None, bool]:
    """Lê um instante local. Vazio é válido e significa sem limite."""
    raw = text.strip()
    if not raw:
        return None, True
    for fmt in _WITH_TIME + _DATE_ONLY:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        local = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
        if end and fmt in _DATE_ONLY:
            local = local.replace(hour=23, minute=59, second=59)
        return local.astimezone(UTC), True
    return None, False


def format_history_time(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M")


def format_history_entry(
    record: TranslationRecord,
    *,
    direction: str,
    provider: str,
    extra: str = "",
) -> str:
    stamp = format_history_time(record.timestamp)
    name = f" [{record.conversation}]" if record.conversation else ""
    bilingual = ""
    if record.portuguese_text:
        bilingual += f"  PT: {record.portuguese_text}\n"
    if record.english_text:
        bilingual += f"  EN: {record.english_text}\n"
    timing = f"{record.latency_ms / 1000:.1f}s"
    if record.stt_ms or record.translation_ms:
        timing += (
            f"  (STT {record.stt_ms / 1000:.1f} · MT {record.translation_ms / 1000:.1f}"
            f" · TTS {record.speech_ms / 1000:.1f})"
        )
    if extra:
        timing += extra
    return (
        f"[{stamp}]{name} [{direction}] {record.source_text}\n"
        f"→ {record.translated_text}\n"
        f"{bilingual}"
        f"  {provider} • {timing}\n"
        f"{'─' * 72}\n"
    )
