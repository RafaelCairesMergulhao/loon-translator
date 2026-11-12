from datetime import UTC, datetime

from domain.entities import AudioDirection, TranslationRecord
from domain.practice import (
    choose_practice_phrase,
    language_code,
    normalize_practice_text,
    score_repetition,
)


def test_normalize_drops_case_accents_and_punctuation() -> None:
    assert normalize_practice_text("Café, por favor!") == "cafe por favor"
    assert normalize_practice_text("¿DÓNDE está el baño?") == "donde esta el bano"


def test_score_marks_correct_missing_and_extra() -> None:
    perfect = score_repetition("Good morning", "good morning!")
    assert perfect.percent == 100
    assert [mark.status for mark in perfect.words] == ["correct", "correct"]

    missing = score_repetition("the cat sat", "the sat")
    assert missing.percent == 67
    assert [mark.status for mark in missing.words] == ["correct", "missing", "correct"]
    assert missing.words[1].text == "cat"

    extra = score_repetition("the cat sat", "the big cat sat")
    assert extra.percent == 100
    assert [mark.status for mark in extra.words] == ["correct", "extra", "correct", "correct"]
    assert extra.words[1].text == "big"

    empty = score_repetition("hello world", "")
    assert empty.percent == 0
    assert [mark.status for mark in empty.words] == ["missing", "missing"]


def test_choose_phrase_prefers_history_then_fallback() -> None:
    record = TranslationRecord(
        source_text="Onde fica o banheiro?",
        translated_text="Where is the bathroom?",
        source_lang="pt",
        target_lang="en",
        direction=AudioDirection.OUTGOING,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    chosen = choose_practice_phrase("Inglês", [record], used=())
    assert chosen.text == "Where is the bathroom?"
    assert chosen.meaning == "Onde fica o banheiro?"

    skipped = choose_practice_phrase("Inglês", [record], used={chosen.text})
    assert skipped.text != chosen.text
    assert skipped.language in {"Inglês", "en"} or skipped.text

    spanish = choose_practice_phrase("Espanhol", [], used=())
    assert "baño" in spanish.text.lower() or "buenos" in spanish.text.lower()
    assert language_code("Espanhol (Latino)") == "es"
