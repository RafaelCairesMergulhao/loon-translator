import pytest

from services.cultural_equivalence import normalize_context
from services.translation_engine import TranslationEngine, TranslationError


def test_context_aliases() -> None:
    assert normalize_context("Viagem") == "travel"
    assert normalize_context("Formal / Corporativo") == "formal"
    assert normalize_context("qualquer") == "casual"
    assert normalize_context(None) == "casual"


def test_english_idiom_skips_literal_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args):
        raise AssertionError("a expressão não deveria ir ao tradutor literal")

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fail))
    result = TranslationEngine().translate(
        "break a leg",
        "en",
        "Português",
        "Local",
        context="casual",
    )
    assert result.text == "Boa sorte"
    assert result.provider == "equivalência cultural"


def test_context_changes_register(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TranslationEngine,
        "_translate_argos",
        staticmethod(lambda *_args: "não deveria"),
    )
    casual = TranslationEngine().translate("tá de boa", "pt", "Inglês", "Local", context="casual")
    formal = TranslationEngine().translate("tá de boa", "pt", "Inglês", "Local", context="formal")
    travel = TranslationEngine().translate("tá de boa", "pt", "Inglês", "Local", context="travel")
    assert casual.text == "It's all good"
    assert formal.text == "Everything is fine"
    assert travel.text == "No problem"


def test_idiom_inside_sentence_keeps_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(text, _source, _target):
        seen.append(text)
        return text.strip().upper()

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fake))
    result = TranslationEngine().translate(
        "please, break a leg",
        "en",
        "Português",
        "Local",
        context="casual",
    )
    assert seen == ["please"]
    assert result.text == "PLEASE, boa sorte"
    assert result.provider.endswith("equivalência")


def test_fragments_lose_invented_punctuation_and_keep_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = {"eu tava sem internet.": "I was out of internet.", "Pode começar sem mim": "You can start without me."}

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(lambda text, *_: replies[text]))
    engine = TranslationEngine()
    assert engine.translate("Foi mal, eu tava sem internet.", "pt", "Inglês", "Local").text == (
        "My bad, I was out of internet."
    )
    assert engine.translate("Pode começar sem mim, tá ligado?", "pt", "Inglês", "Local").text == (
        "You can start without me, you know?"
    )


def test_fragment_in_the_middle_is_not_capitalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(lambda *_: "Isso."))
    result = TranslationEngine().translate("That costs an arm and a leg", "en", "Português", "Local")
    assert result.text == "Isso custa os olhos da cara"


def test_colloquial_portuguese_is_expanded_before_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake(text, _source, _target):
        seen.append(text)
        return "you are going now"

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fake))
    result = TranslationEngine().translate("cê vai agora", "pt", "Inglês", "Local")
    assert seen == ["você vai agora"]
    assert result.provider == "Argos local"


def test_english_slang_contractions_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(text, _source, _target):
        seen.append(text)
        return "vou"

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fake))
    TranslationEngine().translate("I'm gonna go", "en", "Português", "Local", context="formal")
    assert "going to" in seen[0]
    assert "gonna" not in seen[0].lower()


def test_unknown_sentence_still_reports_missing_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args):
        raise RuntimeError("não instalado")

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fail))
    with pytest.raises(TranslationError, match="ainda não foi baixado"):
        TranslationEngine().translate("olá", "pt", "Inglês", "Local")
