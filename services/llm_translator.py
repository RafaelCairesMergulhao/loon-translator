"""Tradução contextual por LLM pela API de chat no formato da OpenAI.

O mesmo cliente atende Ollama e LM Studio (locais, sem custo), Groq, OpenAI,
Gemini e OpenRouter. O modelo recebe o registro da conversa (casual, formal,
viagem) e as últimas falas dos dois lados, o que resolve pronomes, gênero e
gírias que um tradutor frase a frase erra.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from domain.structures import CircularBuffer

logger = logging.getLogger(__name__)

PRESETS: dict[str, tuple[str, str]] = {
    "ollama": ("http://localhost:11434/v1", "qwen2.5:3b"),
    "lmstudio": ("http://localhost:1234/v1", "local-model"),
    "groq": ("https://api.groq.com/openai/v1", "qwen/qwen3.8-27b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"),
}

PROVIDER_LABELS = {
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "groq": "Groq",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "custom": "LLM",
}

_NAMES_BY_LANGUAGE = {
    "Português": "Brazilian Portuguese",
    "Espanhol": "European Spanish",
    "Espanhol (Latino)": "Latin American Spanish",
    "Chinês (Mandarim)": "Simplified Chinese (Mandarin)",
}

_NAMES_BY_CODE = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "hi": "Hindi",
    "ar": "Arabic",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian",
    "ko": "Korean",
}

_REGISTERS = {
    "casual": "casual and friendly; natural slang is welcome",
    "formal": "formal business register; polite and precise, no slang",
    "travel": "clear, simple phrases that a traveller and a local would use",
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL = re.compile(r"^\s*([\w ()]{0,40}\btranslation|tradução|traducción|traduction)\s*:\s*", re.IGNORECASE)
_TAGS = re.compile(r"</?utterance>", re.IGNORECASE)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 4.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider in {"", "none", "off", "false"}:
            return cls()
        default_url, default_model = PRESETS.get(provider, ("", ""))
        try:
            timeout = float(os.getenv("LLM_TIMEOUT", "4.0"))
        except ValueError:
            timeout = 4.0
        return cls(
            provider=provider if provider in PRESETS else "custom",
            base_url=(os.getenv("LLM_BASE_URL", "").strip() or default_url).rstrip("/"),
            model=os.getenv("LLM_MODEL", "").strip() or default_model,
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            timeout_s=max(0.5, timeout),
        )

    @property
    def is_local(self) -> bool:
        host = (urlparse(self.base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model and (self.api_key or self.is_local))

    @property
    def label(self) -> str:
        return f"{PROVIDER_LABELS.get(self.provider, 'LLM')} ({self.model})"


def language_description(code: str, language_name: str = "") -> str:
    if language_name in _NAMES_BY_LANGUAGE:
        return _NAMES_BY_LANGUAGE[language_name]
    base = (code or "").lower().split("-")[0]
    if base == "auto":
        return "the speaker's language (detect it)"
    return _NAMES_BY_CODE.get(base, code or "the source language")


class LLMTranslator:
    """Intérprete por LLM com memória curta da conversa."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: httpx.Client | None = None,
        memory: int = 6,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = client
        self._client_lock = threading.Lock()
        self._history: CircularBuffer[tuple[str, str, str, str]] = CircularBuffer(memory)

    @property
    def configured(self) -> bool:
        return self.config.configured

    @property
    def label(self) -> str:
        return self.config.label

    def _http(self) -> httpx.Client:
        """Um cliente por sessão reaproveita TCP/TLS; poupa o aperto de mão a cada frase."""
        with self._client_lock:
            if self._client is None:
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                self._client = httpx.Client(
                    base_url=self.config.base_url,
                    headers=headers,
                    timeout=httpx.Timeout(self.config.timeout_s, connect=min(2.0, self.config.timeout_s)),
                )
            return self._client

    def build_messages(
        self,
        text: str,
        source_code: str,
        target_code: str,
        target_name: str,
        context: str,
        hints: Sequence[tuple[str, str]] = (),
    ) -> list[dict[str, str]]:
        source = language_description(source_code)
        target = language_description(target_code, target_name)
        lines = [
            "You are a professional simultaneous interpreter in a live voice call.",
            f"Translate the speaker's utterance from {source} into {target}.",
            f"Register: {_REGISTERS.get(context, _REGISTERS['casual'])}.",
            "Rules:",
            "- Reply with the translation only: no quotes, notes, or explanations.",
            f"- Keep meaning, intent, and tone; render idioms and slang with natural {target} equivalents, never word by word.",
            "- The text comes from speech recognition: silently fix obvious recognition mistakes and drop filler words.",
            "- Keep it short and easy to speak: a text-to-speech voice will read it aloud.",
            "- Keep names, numbers, and brands unchanged.",
        ]
        history = self._history.get_all()
        if history:
            lines.append("Recent conversation, for context only (do not translate it again):")
            for spoken_lang, spoken, translated_lang, translated in history:
                lines.append(f"- [{spoken_lang}] {spoken} => [{translated_lang}] {translated}")
        # Modelos pequenos (3B) respondem à fala como se fosse um chat; a ordem precisa vir junto do texto.
        glossary = "".join(f'Slang: "{raw}" means "{value}".\n' for raw, value in hints)
        request = (
            f"Translate into {target}. The text is something a person said; never answer it.\n"
            f"{glossary}"
            f"<utterance>\n{text}\n</utterance>\n"
            f"{target} translation:"
        )
        return [
            {"role": "system", "content": "\n".join(lines)},
            {"role": "user", "content": request},
        ]

    def translate(
        self,
        text: str,
        source_code: str,
        target_code: str,
        target_name: str = "",
        context: str = "casual",
        hints: Sequence[tuple[str, str]] = (),
    ) -> str:
        if not self.configured:
            raise LLMError("LLM não configurado")
        payload = {
            "model": self.config.model,
            "messages": self.build_messages(text, source_code, target_code, target_name, context, hints),
            "temperature": 0.2,
            # Modelos de raciocínio (gpt-oss, DeepSeek-R1) gastam tokens pensando antes da resposta.
            "max_tokens": min(1024, 256 + 4 * len(text.split())),
            "stream": False,
        }
        try:
            response = self._http().post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.config.label} excedeu {self.config.timeout_s:.1f}s") from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.config.label}: {exc}") from exc
        translated = self.clean(content or "")
        if not translated:
            raise LLMError(f"{self.config.label}: resposta vazia")
        self._history.insert((source_code, text, target_code, translated))
        return translated

    @staticmethod
    def clean(content: str) -> str:
        text = _TAGS.sub("", _THINK.sub("", content)).strip()
        text = _LABEL.sub("", text)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”«»":
            text = text[1:-1]
        text = text.strip().strip("“”«»").strip()
        return " ".join(text.split())

    def warm_up(self) -> None:
        """Ollama e LM Studio carregam o modelo na primeira chamada; paga-se esse custo antes da fala."""
        if not self.configured or not self.config.is_local:
            return
        try:
            self._http().post(
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 1,
                },
            )
        except httpx.HTTPError as exc:
            logger.info("Aquecimento do LLM local falhou: %s", exc)

    def forget(self) -> None:
        self._history.clear()

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
