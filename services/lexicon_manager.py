"""Glossário de gírias carregado uma vez e consultado pela frase, não pelo dicionário inteiro.

O JSON fica organizado por idioma de origem. O Whisper recebe só uma lista curta
de hotwords desse idioma. O tradutor recebe só os termos que apareceram na fala.
Espanhol da Espanha e espanhol latino usam o mesmo código ``es`` no app, então
as duas falas convivem no mesmo mapa, cada uma com a sua expressão.
"""

from __future__ import annotations

import json
import logging
import sys
import unicodedata
from pathlib import Path

from services.cultural_equivalence import language_base

logger = logging.getLogger(__name__)

_HOTWORD_LIMIT = 8
_HINT_LIMIT = 6
_MAX_WORDS = 4
_PUNCTUATION = ".,!?;:…¿¡\"'“”«»()"


def _bundled_lexicon() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        packed = Path(meipass) / "data" / "lexicon.json"
        if packed.exists():
            return packed
    return Path(__file__).resolve().parent.parent / "data" / "lexicon.json"


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.casefold())
    plain = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return " ".join(plain.strip(_PUNCTUATION).split())


class LexiconManager:
    """Índice ``(origem, destino, expressão) -> equivalente``. A busca segue as palavras da frase."""

    def __init__(self, path: Path | None = None) -> None:
        self._hotwords: dict[str, tuple[str, ...]] = {}
        self._gloss: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
        self._load(path or _bundled_lexicon())

    def hotwords(self, language: str, limit: int = _HOTWORD_LIMIT) -> tuple[str, ...]:
        return self._hotwords.get(language_base(language), ())[:limit]

    def hints(self, text: str, source: str, target: str, limit: int = _HINT_LIMIT) -> list[tuple[str, str]]:
        table = self._gloss.get((language_base(source), language_base(target)))
        if not table or not text.strip():
            return []
        words = [word.strip(_PUNCTUATION) for word in text.split()]
        words = [word for word in words if _fold(word)]
        keys = [_fold(word) for word in words]
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        index = 0
        while index < len(keys) and len(found) < limit:
            match: tuple[int, str, str] | None = None
            for size in range(min(_MAX_WORDS, len(keys) - index), 0, -1):
                phrase = " ".join(keys[index : index + size])
                entry = table.get(phrase)
                if entry is None or phrase in seen:
                    continue
                match = (size, phrase, entry[1])
                break
            if match is None:
                index += 1
                continue
            size, phrase, value = match
            seen.add(phrase)
            found.append((" ".join(words[index : index + size]), value))
            index += size
        return found

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("Léxico ignorado (%s): %s", path, exc)
            return
        if not isinstance(raw, dict):
            logger.warning("Léxico ignorado (%s): esperado um objeto JSON", path)
            return
        for language, block in raw.items():
            if not isinstance(block, dict):
                continue
            source = language_base(str(language))
            hotwords = [
                str(term).strip()
                for term in block.get("hotwords", [])
                if str(term).strip()
            ]
            if hotwords:
                self._hotwords[source] = tuple(hotwords)
            targets = block.get("to", {})
            if not isinstance(targets, dict):
                continue
            for target, pairs in targets.items():
                if not isinstance(pairs, dict):
                    continue
                table: dict[str, tuple[str, str]] = {}
                for raw_term, equivalent in pairs.items():
                    term = str(raw_term).strip()
                    value = str(equivalent).strip()
                    key = _fold(term)
                    if key and value:
                        table[key] = (term, value)
                if table:
                    self._gloss[(source, language_base(str(target)))] = table


_SHARED: LexiconManager | None = None


def shared_lexicon() -> LexiconManager:
    global _SHARED
    if _SHARED is None:
        _SHARED = LexiconManager()
    return _SHARED
