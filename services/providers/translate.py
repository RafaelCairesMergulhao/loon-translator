"""Tradução por cima do motor que o Loon já usa, mais NLLB e Claude.

Argos, DeepL e Groq continuam em TranslationEngine. Este módulo só escolhe
qual modo chamar e acrescenta os dois caminhos que aquele motor não tem.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import httpx

from core.config import APP_DIR
from services.llm_translator import activate_llm
from services.providers.stack import MT_MODE
from services.cultural_equivalence import render_cultural
from services.translation_engine import (
    LANGUAGES_DICT,
    TranslationEngine,
    TranslationError,
    TranslationResult,
    _keep_untranslated_tail,
)

logger = logging.getLogger(__name__)

_NLLB = {
    "pt": "por_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
}


def glossary_block(pairs: list[tuple[str, str]]) -> str:
    """Linhas de glossário no mesmo formato que o prompt da LLM já espera."""
    return "".join(f'Slang: "{raw}" means "{value}".\n' for raw, value in pairs if raw and value)


class RoutedTranslator:
    def __init__(self, inner: TranslationEngine, mt: str) -> None:
        self._inner = inner
        self.mt = mt
        self.llm = inner.llm
        if mt == "groq":
            activate_llm("groq")
        elif mt == "openai":
            activate_llm("openai")

    def translate(
        self,
        text: str,
        source_code: str,
        target_language_name: str,
        mode: str = "Híbrido",
        context: str = "casual",
        speculative: bool = False,
        fragment: bool = False,
    ) -> TranslationResult:
        if self.mt == "nllb":
            try:
                return _nllb_with_idioms(text, source_code, target_language_name, context)
            except Exception as exc:
                logger.info("NLLB indisponível; usando Argos: %s", exc)
                fallback = self._inner.translate(
                    text,
                    source_code,
                    target_language_name,
                    "Local",
                    context=context,
                    speculative=speculative,
                    fragment=fragment,
                )
                return TranslationResult(fallback.text, f"{fallback.provider} (NLLB ausente)")
        if self.mt == "claude":
            return claude_translate(text, source_code, target_language_name, context)
        forced = MT_MODE.get(self.mt, mode)
        return self._inner.translate(
            text,
            source_code,
            target_language_name,
            forced,
            context=context,
            speculative=speculative,
            fragment=fragment,
        )

    def preview(self, text: str, *args: Any, **kwargs: Any) -> str | None:
        return self._inner.preview(text, *args, **kwargs)

    def warm_up(self, pairs: list[tuple[str, str]], mode: str) -> None:
        if self.mt == "nllb":
            try:
                nllb_load()
            except TranslationError as exc:
                logger.info("NLLB não aqueceu; a frase usa o Argos: %s", exc)
        self._inner.warm_up(pairs, MT_MODE.get(self.mt, mode))

    def close(self) -> None:
        self._inner.close()


def nllb_dir() -> Path:
    chosen = os.getenv("LOON_NLLB_DIR", "").strip()
    if chosen:
        return Path(chosen)
    return APP_DIR / "models" / "nllb-200-distilled-600M-int8"


def nllb_ready(root: Path | None = None) -> bool:
    folder = root or nllb_dir()
    return (folder / "model.bin").is_file() and (folder / "sentencepiece.bpe.model").is_file()


_NLLB_LOCK = threading.Lock()
_NLLB_CACHE: tuple[Path, Any, Any] | None = None


def nllb_load(root: Path | None = None) -> tuple[Any, Any]:
    """Carrega o modelo uma vez. Abrir 600 MB a cada frase deixaria a chamada parada."""
    global _NLLB_CACHE
    folder = root or nllb_dir()
    if not nllb_ready(folder):
        raise TranslationError(
            "O NLLB-200 INT8 não está nesta máquina. "
            f"Coloque o modelo convertido em {folder}."
        )
    with _NLLB_LOCK:
        if _NLLB_CACHE is not None and _NLLB_CACHE[0] == folder:
            return _NLLB_CACHE[1], _NLLB_CACHE[2]
        try:
            import ctranslate2
            import sentencepiece as spm
        except ImportError as exc:
            raise TranslationError("O NLLB local precisa de ctranslate2 e sentencepiece.") from exc
        processor = spm.SentencePieceProcessor(model_file=str(folder / "sentencepiece.bpe.model"))
        translator = ctranslate2.Translator(str(folder), device="cpu", compute_type="int8", inter_threads=1, intra_threads=4)
        _NLLB_CACHE = (folder, processor, translator)
        return processor, translator


def _nllb_with_idioms(text: str, source_code: str, target_language_name: str, context: str) -> TranslationResult:
    """O NLLB traduz o que sobrou. Expressão conhecida e 'não saiu' não passam por ele."""
    target_code = LANGUAGES_DICT.get(target_language_name, "")

    def machine(chunk: str) -> str:
        return nllb_translate(chunk, source_code, target_language_name).text

    rendered = render_cultural(text, source_code, target_code, context, machine)
    translated = _keep_untranslated_tail(text, rendered.text, target_code.split("-")[0].lower())
    if rendered.fully_resolved:
        provider = "equivalência cultural"
    elif rendered.used_idiom:
        provider = "NLLB-200 + equivalência"
    else:
        provider = "NLLB-200"
    return TranslationResult(translated, provider)


def nllb_translate(text: str, source_code: str, target_language_name: str) -> TranslationResult:
    """NLLB-200 600M INT8. Traduz a frase inteira e só lê o que já está no disco."""
    target_code = LANGUAGES_DICT.get(target_language_name, "")
    source = _NLLB.get((source_code or "").split("-")[0].lower())
    target = _NLLB.get(target_code.split("-")[0].lower())
    if not source or not target:
        raise TranslationError(f"NLLB não cobre {source_code} → {target_language_name}")
    processor, translator = nllb_load()
    pieces = [source, *processor.encode(text, out_type=str), "</s>"]
    generated = translator.translate_batch([pieces], target_prefix=[[target]], beam_size=2)
    tokens = generated[0].hypotheses[0] if generated and generated[0].hypotheses else []
    skip = {source, target, "</s>", "<s>", "<unk>"}
    translated = processor.decode([token for token in tokens if token not in skip]).strip()
    if not translated:
        raise TranslationError("O NLLB devolveu uma tradução vazia.")
    return TranslationResult(translated, "NLLB-200")


def claude_translate(text: str, source_code: str, target_language_name: str, context: str) -> TranslationResult:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise TranslationError("Defina ANTHROPIC_API_KEY para traduzir com Claude.")
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    prompt = (
        "Translate the utterance. Reply with the translation only. "
        f"From {source_code} to {target_language_name}. Register: {context}.\n{text}"
    )
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=20,
    )
    response.raise_for_status()
    blocks = response.json().get("content") or []
    translated = " ".join(str(block.get("text") or "") for block in blocks if block.get("type") == "text").strip()
    if not translated:
        raise TranslationError("Claude devolveu uma tradução vazia.")
    return TranslationResult(translated, "Claude")
