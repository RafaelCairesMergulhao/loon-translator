"""Correções léxicas antes e depois da tradução, sem chamar modelo nenhum.

O Whisper escreve marcas e termos de TI de jeitos previsíveis ("git hub",
"java script"), gírias como "manda no zap" saem literais do tradutor, e o
tradutor às vezes traduz o que devia ficar igual ("solicitação de pull").
Regex com dicionário resolve tudo isso em microssegundos.

Termos próprios (nomes da equipe, produtos) vão em
``%APPDATA%\\LoonTranslator\\glossary.json``::

    {
      "terms": ["Kubernetes", "Loon Translator"],
      "fix": {"cuber netes": "Kubernetes"},
      "after": {"pt": {"solicitação de pull": "pull request"}}
    }

``terms`` fixa a grafia e também vai como dica (hotwords) para o Whisper.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.config import APP_DIR
from services.cultural_equivalence import language_base
from services.lexicon_manager import shared_lexicon

logger = logging.getLogger(__name__)

USER_GLOSSARY = APP_DIR / "glossary.json"

_Rule = tuple[re.Pattern[str], str]

# Só entram nomes que nunca são palavra comum: "react", "go", "teams" e
# "windows" ficariam com maiúscula no meio de uma frase qualquer.
_CANONICAL: tuple[tuple[str, str], ...] = (
    (r"java\s?script", "JavaScript"),
    (r"type\s?script", "TypeScript"),
    (r"git\s?hub", "GitHub"),
    (r"git\s?lab", "GitLab"),
    (r"postgre\s?sql", "PostgreSQL"),
    (r"mysql", "MySQL"),
    (r"mongo\s?db", "MongoDB"),
    (r"node\s?\.?\s?js", "Node.js"),
    (r"next\s?\.?\s?js", "Next.js"),
    (r"kubernetes|kubernets|cubernetes", "Kubernetes"),
    (r"dev\s?ops", "DevOps"),
    (r"chat\s?gpt|chat\s?g\.?\s?p\.?\s?t", "ChatGPT"),
    (r"openai", "OpenAI"),
    (r"you\s?tube", "YouTube"),
    (r"whats\s?app|what'?s\s?app", "WhatsApp"),
    (r"linkedin", "LinkedIn"),
    (r"iphone", "iPhone"),
    (r"ipad", "iPad"),
    (r"mac\s?book", "MacBook"),
    (r"play\s?station", "PlayStation"),
    (r"xbox", "Xbox"),
    (r"tik\s?tok", "TikTok"),
    (r"powerpoint", "PowerPoint"),
    (r"wi-?\s?fi", "Wi-Fi"),
    (r"macos", "macOS"),
)

# Como o Whisper escreve o que ouviu em cada idioma de origem.
_HEARD: dict[str, tuple[tuple[str, str], ...]] = {
    "pt": (
        (r"zap\s?zap|zap", "WhatsApp"),
        (r"uats\s?app?|uatsap[ei]?", "WhatsApp"),
        (r"insta", "Instagram"),
        (r"chat\s?gepet[eéo]|chat\s?jipiti", "ChatGPT"),
        (r"pai\s?ton|p[aá]iton", "Python"),
        (r"d[oó]quer", "Docker"),
        (r"guit\s?r[au]b|guit\s?hub", "GitHub"),
    ),
}

# O que o tradutor automático traduz e o falante do idioma de destino não traduziria.
_TRANSLATED: dict[str, tuple[tuple[str, str], ...]] = {
    "pt": (
        (r"(?:solicitações|pedidos) (?:de )?(?:pull|puxar)", "pull requests"),
        (r"(?:solicitação|pedido) (?:de )?(?:pull|puxar)", "pull request"),
    ),
}


def _rule(pattern: str, replacement: str) -> _Rule:
    return re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])", re.IGNORECASE), replacement


def _literal(text: str) -> str:
    return r"\s*".join(re.escape(word) for word in text.split())


def _apply(rules: list[_Rule], text: str) -> str:
    for pattern, replacement in rules:
        text = pattern.sub(lambda _match, value=replacement: value, text)
    return text


class Lexicon:
    def __init__(self, path: Path | None = None) -> None:
        self._canonical = [_rule(pattern, value) for pattern, value in _CANONICAL]
        self._heard = {lang: [_rule(p, v) for p, v in rules] for lang, rules in _HEARD.items()}
        self._translated = {lang: [_rule(p, v) for p, v in rules] for lang, rules in _TRANSLATED.items()}
        self._fixes: list[_Rule] = []
        self._terms: list[str] = []
        self._hotwords: str | None = None
        if path is not None:
            self._load(path)

    @property
    def hotwords(self) -> str | None:
        return self._hotwords

    def hotwords_for(self, language: str) -> str | None:
        """Nomes do usuário primeiro, depois poucas gírias do idioma. O resto do léxico não entra no Whisper."""
        seen: set[str] = set()
        chosen: list[str] = []
        for term in [*self._terms, *shared_lexicon().hotwords(language)]:
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            chosen.append(term)
            if len(chosen) == 8:
                break
        return ", ".join(chosen) or None

    def fix_transcript(self, text: str, language: str = "") -> str:
        text = _apply(self._heard.get(language_base(language), []), text)
        return _apply(self._canonical, _apply(self._fixes, text))

    def fix_translation(self, text: str, language: str = "") -> str:
        return _apply(self._canonical, _apply(self._translated.get(language_base(language), []), text))

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("Glossário ignorado (%s): %s", path, exc)
            return
        if not isinstance(raw, dict):
            logger.warning("Glossário ignorado (%s): esperado um objeto JSON", path)
            return
        terms = [str(term).strip() for term in raw.get("terms", []) if str(term).strip()]
        self._terms = terms
        self._canonical.extend(_rule(_literal(term), term) for term in terms)
        self._fixes.extend(
            _rule(_literal(str(wrong)), str(right))
            for wrong, right in dict(raw.get("fix", {})).items()
            if str(wrong).strip()
        )
        for lang, pairs in dict(raw.get("after", {})).items():
            rules = self._translated.setdefault(language_base(lang), [])
            rules.extend(_rule(_literal(str(wrong)), str(right)) for wrong, right in dict(pairs).items())
        self._hotwords = " ".join(terms) or None
