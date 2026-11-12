import json

from domain.transcripts import RollingTranscript, drop_repeated_prefix, stable_prefix
from services.lexicon import Lexicon


def test_brands_and_it_terms_get_their_spelling() -> None:
    lexicon = Lexicon()
    assert (
        lexicon.fix_transcript("subi o código no git hub e no gitlab", "pt")
        == "subi o código no GitHub e no GitLab"
    )
    assert lexicon.fix_transcript("I write java script and node js", "en") == "I write JavaScript and Node.js"
    assert lexicon.fix_transcript("pergunta pro chat gpt", "pt-BR") == "pergunta pro ChatGPT"


def test_common_words_are_left_alone() -> None:
    lexicon = Lexicon()
    text = "react quickly, go to teams and open the windows"
    assert lexicon.fix_transcript(text, "en") == text
    assert lexicon.fix_transcript("o zapato", "es") == "o zapato"


def test_portuguese_slang_is_heard_as_the_brand() -> None:
    lexicon = Lexicon()
    assert lexicon.fix_transcript("me chama no zap", "pt") == "me chama no WhatsApp"
    assert lexicon.fix_transcript("ele usa dóquer e pai ton", "pt") == "ele usa Docker e Python"
    assert lexicon.fix_transcript("me chama no zap", "en") == "me chama no zap"


def test_translator_literalisms_are_undone() -> None:
    lexicon = Lexicon()
    assert lexicon.fix_translation("Abra uma solicitação de pull hoje", "pt") == "Abra uma pull request hoje"
    assert lexicon.fix_translation("dois pedidos de pull", "pt") == "dois pull requests"


def test_user_glossary_adds_terms_fixes_and_hotwords(tmp_path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps(
            {
                "terms": ["Loon Translator", "Kafka"],
                "fix": {"cuber netes": "Kubernetes"},
                "after": {"pt": {"bilhete": "ticket"}},
            }
        ),
        encoding="utf-8",
    )
    lexicon = Lexicon(path)
    assert lexicon.hotwords == "Loon Translator Kafka"
    assert lexicon.fix_transcript("o loon  translator usa kafka", "pt") == "o Loon Translator usa Kafka"
    assert lexicon.fix_transcript("deploy no cuber netes", "pt") == "deploy no Kubernetes"
    assert lexicon.fix_translation("abra um bilhete", "pt-BR") == "abra um ticket"


def test_broken_or_missing_glossary_is_ignored(tmp_path) -> None:
    broken = tmp_path / "glossary.json"
    broken.write_text("{ não é json", encoding="utf-8")
    assert Lexicon(broken).hotwords is None
    assert Lexicon(tmp_path / "nao_existe.json").fix_transcript("git hub") == "GitHub"


def test_stable_prefix_is_where_two_hypotheses_agree() -> None:
    assert stable_prefix([], ["eu", "vou"]) == 0
    assert stable_prefix(["Eu", "vou,"], ["eu", "vou", "para", "casa"]) == 2
    assert stable_prefix(["eu", "fui"], ["eu", "vou", "embora"]) == 1


def test_overlap_word_is_not_repeated_after_a_cut() -> None:
    assert drop_repeated_prefix("e aí eu disse para ele", "ele que não ia") == "que não ia"
    assert drop_repeated_prefix("fui ao mercado", "Mercado, padaria e farmácia") == "padaria e farmácia"
    assert drop_repeated_prefix("fui ao mercado", "depois voltei") == "depois voltei"


def test_rolling_transcript_keeps_the_last_seconds() -> None:
    context = RollingTranscript(seconds=15.0, max_chars=30)
    context.add("primeira frase dita", at=0.0)
    context.add("segunda frase", at=10.0)
    assert context.text(now=12.0) == "frase dita segunda frase"
    assert context.text(now=20.0) == "segunda frase"
    assert context.text(now=40.0) == ""


def test_rolling_transcript_offers_the_last_three_phrases() -> None:
    context = RollingTranscript(seconds=15.0, max_chars=240)
    for index, text in enumerate(("um", "dois", "três", "quatro")):
        context.add(text, at=float(index))
    assert context.last_phrases(now=4.0) == "dois três quatro"
    assert context.text(now=4.0) == "um dois três quatro"
    context.add("e nem um pouquinho", at=5.0)
    assert context.already_said("e nem um pouquinho", now=8.0)
    assert not context.already_said("Sim", now=8.0)
    assert not context.already_said("e nem um pouquinho", now=20.0)


def test_lexicon_matches_only_the_slang_that_was_said() -> None:
    from services.lexicon_manager import LexiconManager

    lexicon = LexiconManager()
    assert lexicon.hints("mano, tá ligado que fechou", "pt", "en") == [
        ("mano", "bro"),
        ("tá ligado", "you know what I mean"),
        ("fechou", "deal"),
    ]
    assert lexicon.hints("olá, tudo bem", "pt", "en") == []
    assert lexicon.hints("wey, no manches", "es", "pt") == [("wey", "mano"), ("no manches", "não brinca")]
    assert "mano" in lexicon.hotwords("pt-BR")
    assert len(lexicon.hotwords("pt")) <= 8
    assert lexicon.hotwords("auto") == ()


def test_spoken_hotwords_keep_user_names_ahead_of_slang(tmp_path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"terms": ["Loon Translator"]}), encoding="utf-8")
    spoken = Lexicon(path).hotwords_for("pt")
    assert spoken is not None
    assert spoken.startswith("Loon Translator")
    assert spoken.count(",") <= 7
