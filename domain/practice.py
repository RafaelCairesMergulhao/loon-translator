from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from domain.entities import TranslationRecord

_LANGUAGE_CODES = {
    "Auto": "auto",
    "Português": "pt",
    "Inglês": "en",
    "Espanhol": "es",
    "Espanhol (Latino)": "es",
    "Francês": "fr",
    "Alemão": "de",
    "Italiano": "it",
    "Hindi": "hi",
    "Árabe": "ar",
    "Japonês": "ja",
    "Chinês (Mandarim)": "zh",
    "Russo": "ru",
    "Coreano": "ko",
}

FALLBACK_PHRASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Português": (
        ("Bom dia, tudo bem?", "Good morning, how are you?"),
        ("Onde fica o banheiro?", "Where is the bathroom?"),
        ("Eu gostaria de um café, por favor.", "I would like a coffee, please."),
        ("Pode falar mais devagar?", "Can you speak more slowly?"),
        ("Muito prazer em conhecer você.", "Nice to meet you."),
    ),
    "Inglês": (
        ("Good morning, how are you?", "Bom dia, como vai?"),
        ("Where is the bathroom?", "Onde fica o banheiro?"),
        ("I would like a coffee, please.", "Eu gostaria de um café, por favor."),
        ("Can you speak more slowly?", "Pode falar mais devagar?"),
        ("Nice to meet you.", "Prazer em conhecer você."),
    ),
    "Espanhol": (
        ("Buenos días, ¿cómo estás?", "Bom dia, como vai?"),
        ("¿Dónde está el baño?", "Onde fica o banheiro?"),
        ("Quisiera un café, por favor.", "Eu gostaria de um café, por favor."),
        ("¿Puedes hablar más despacio?", "Pode falar mais devagar?"),
        ("Mucho gusto en conocerte.", "Prazer em conhecer você."),
    ),
    "Francês": (
        ("Bonjour, comment allez-vous ?", "Bom dia, como vai?"),
        ("Où sont les toilettes ?", "Onde fica o banheiro?"),
    ),
    "Alemão": (
        ("Guten Morgen, wie geht es Ihnen?", "Bom dia, como vai?"),
        ("Wo ist die Toilette?", "Onde fica o banheiro?"),
    ),
    "Italiano": (
        ("Buongiorno, come stai?", "Bom dia, como vai?"),
        ("Dov'è il bagno?", "Onde fica o banheiro?"),
    ),
    "Hindi": (
        ("नमस्ते, आप कैसे हैं?", "Olá, como vai?"),
        ("बाथरूम कहाँ है?", "Onde fica o banheiro?"),
        ("मुझे एक कॉफ़ी चाहिए, कृपया।", "Eu gostaria de um café, por favor."),
        ("क्या आप धीरे बोल सकते हैं?", "Pode falar mais devagar?"),
        ("आपसे मिलकर खुशी हुई।", "Prazer em conhecer você."),
    ),
}

_FALLBACK_ALIAS = {
    "Espanhol (Latino)": "Espanhol",
}


@dataclass(frozen=True, slots=True)
class WordMark:
    text: str
    status: str


@dataclass(frozen=True, slots=True)
class PracticeScore:
    percent: int
    words: tuple[WordMark, ...]


@dataclass(frozen=True, slots=True)
class PracticePhrase:
    text: str
    meaning: str
    language: str


def language_code(name_or_code: str) -> str:
    raw = (name_or_code or "").strip()
    if raw in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[raw]
    base = raw.lower().replace("_", "-").split("-")[0]
    return base or "en"


def normalize_practice_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    cleaned = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
    return " ".join(cleaned.split())


def tokenize_practice(text: str) -> list[str]:
    return normalize_practice_text(text).split()


def score_repetition(expected: str, heard: str) -> PracticeScore:
    """Alinha palavra a palavra e marca certo / faltou / a mais."""
    target = tokenize_practice(expected)
    said = tokenize_practice(heard)
    if not target:
        extras = tuple(WordMark(word, "extra") for word in said)
        return PracticeScore(100 if not said else 0, extras)
    marks: list[WordMark] = []
    matched = 0
    matcher = difflib.SequenceMatcher(a=target, b=said, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for word in target[i1:i2]:
                marks.append(WordMark(word, "correct"))
                matched += 1
        elif tag == "delete":
            marks.extend(WordMark(word, "missing") for word in target[i1:i2])
        elif tag == "insert":
            marks.extend(WordMark(word, "extra") for word in said[j1:j2])
        else:
            marks.extend(WordMark(word, "missing") for word in target[i1:i2])
            marks.extend(WordMark(word, "extra") for word in said[j1:j2])
    return PracticeScore(round(100 * matched / len(target)), tuple(marks))


def phrases_from_history(
    language: str,
    records: Sequence[TranslationRecord],
) -> list[PracticePhrase]:
    code = language_code(language)
    phrases: list[PracticePhrase] = []
    seen: set[str] = set()
    for record in reversed(records):
        target = language_code(record.target_lang)
        text = (record.translated_text or "").strip()
        if target != code or not text:
            continue
        key = normalize_practice_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        phrases.append(PracticePhrase(text, (record.source_text or "").strip(), language))
    return phrases


def fallback_phrases(language: str) -> list[PracticePhrase]:
    key = _FALLBACK_ALIAS.get(language, language)
    rows = FALLBACK_PHRASES.get(key) or FALLBACK_PHRASES["Inglês"]
    label = language if language in FALLBACK_PHRASES or language in _FALLBACK_ALIAS else "Inglês"
    return [PracticePhrase(text, meaning, label) for text, meaning in rows]


def choose_practice_phrase(
    language: str,
    records: Sequence[TranslationRecord],
    used: Iterable[str] = (),
) -> PracticePhrase:
    skip = {normalize_practice_text(item) for item in used if item}
    for pool in (phrases_from_history(language, records), fallback_phrases(language)):
        for phrase in pool:
            if normalize_practice_text(phrase.text) not in skip:
                return phrase
    pool = fallback_phrases(language)
    return pool[0]
