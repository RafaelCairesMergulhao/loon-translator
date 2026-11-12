from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from deep_translator import DeeplTranslator, GoogleTranslator

from domain.metrics import ProviderHealth
from services.cultural_equivalence import (
    detect_idiom_language,
    glossary,
    language_base,
    normalize_context,
    render_cultural,
)
from services.lexicon_manager import shared_lexicon
from services.llm_translator import LLMTranslator
from services.pt_br import to_brazilian

logger = logging.getLogger(__name__)

_PUNCT_RUN = re.compile(r"([^\w\s]|_)\1{2,}")


_LOOSE_TAIL = re.compile(r"(?i)^(?P<head>.+?[A-Za-zÀ-ÿ0-9])\s+(?P<tail>não saiu)\s*\.?$")
_SAID_NAO_SAIU = re.compile(r"(?i)(?<!\w)não saiu(?!\w)")
_EN_NEGATION = re.compile(r"(?i)(?:\bnot\b|\bnever\b|\bno\b|n't)")


def _separate_loose_tail(text: str) -> str:
    """'boa não saiu' vira duas frases. Numa frase só, o Argos apaga o 'não saiu'."""
    match = _LOOSE_TAIL.match(text.strip())
    if match is None:
        return text
    head = match.group("head").rstrip(" ,;:")
    if head[-1:] in ".!?…":
        return text
    return f"{head}. Não saiu."


def _keep_untranslated_tail(source: str, translated: str, target: str) -> str:
    """Se o modelo local comeu o 'não saiu', a segunda frase volta no inglês."""
    if target != "en" or not _SAID_NAO_SAIU.search(source):
        return translated
    if _EN_NEGATION.search(translated):
        return translated
    spoken = translated.rstrip()
    if spoken and spoken[-1:] not in ".!?":
        spoken += "."
    return f"{spoken} It didn't come out.".strip()


def tidy_for_machine(text: str) -> str:
    """Tira pontuação repetida e a mesma frase colada várias vezes, antes do tradutor local."""
    words = _separate_loose_tail(_PUNCT_RUN.sub(r"\1", text or "")).split()
    if len(words) < 6:
        return " ".join(words)
    limit = min(12, len(words) // 3)
    for size in range(1, limit + 1):
        pattern = tuple(word.casefold() for word in words[:size])
        count = 1
        index = size
        while index + size <= len(words) and tuple(word.casefold() for word in words[index : index + size]) == pattern:
            count += 1
            index += size
        if count >= 3 and index >= len(words) * 0.6:
            return " ".join(words[:size])
    return " ".join(words)

def _slang_hints(text: str, source: str, target: str, context: str) -> list[tuple[str, str]]:
    """A tabela cultural vem primeiro. O JSON só completa o que ela não cobriu, e só o que foi dito."""
    hints = glossary(text, source, target, context)
    seen = {raw.casefold() for raw, _value in hints}
    for raw, value in shared_lexicon().hints(text, source, target):
        if raw.casefold() in seen:
            continue
        seen.add(raw.casefold())
        hints.append((raw, value))
    return hints

LANGUAGES_DICT = {
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
    "Chinês (Mandarim)": "zh-CN",
    "Russo": "ru",
    "Coreano": "ko",
}

VOICE_MAP = {
    "Português": {"Feminina": "pt-BR-FranciscaNeural", "Masculina": "pt-BR-AntonioNeural"},
    "Inglês": {"Feminina": "en-US-JennyNeural", "Masculina": "en-US-AndrewNeural"},
    "Espanhol": {"Feminina": "es-ES-ElviraNeural", "Masculina": "es-ES-AlvaroNeural"},
    "Espanhol (Latino)": {"Feminina": "es-MX-DaliaNeural", "Masculina": "es-MX-JorgeNeural"},
    "Francês": {"Feminina": "fr-FR-DeniseNeural", "Masculina": "fr-FR-HenriNeural"},
    "Alemão": {"Feminina": "de-DE-KatjaNeural", "Masculina": "de-DE-ConradNeural"},
    "Italiano": {"Feminina": "it-IT-ElsaNeural", "Masculina": "it-IT-DiegoNeural"},
    "Hindi": {"Feminina": "hi-IN-SwaraNeural", "Masculina": "hi-IN-MadhurNeural"},
    "Árabe": {"Feminina": "ar-SA-ZariyahNeural", "Masculina": "ar-SA-HamedNeural"},
    "Japonês": {"Feminina": "ja-JP-NanamiNeural", "Masculina": "ja-JP-KeitaNeural"},
    "Chinês (Mandarim)": {"Feminina": "zh-CN-XiaoxiaoNeural", "Masculina": "zh-CN-YunxiNeural"},
    "Russo": {"Feminina": "ru-RU-SvetlanaNeural", "Masculina": "ru-RU-DmitryNeural"},
    "Coreano": {"Feminina": "ko-KR-SunHiNeural", "Masculina": "ko-KR-InJoonNeural"},
}

MODES = ("Local", "Híbrido", "Nuvem", "IA")
LOCAL_FIRST_MODES = {"Local", "Híbrido", "IA"}


@dataclass(frozen=True, slots=True)
class TranslationResult:
    text: str
    provider: str


class TranslationError(RuntimeError):
    pass


class TranslationEngine:
    """Tradução com equivalência cultural, LLM opcional e cadeia de provedores.

    * Local: Argos offline.
    * Híbrido: Argos, depois DeepL/Google.
    * Nuvem: DeepL/Google.
    * IA: LLM com contexto da conversa; se falhar ou demorar, a cadeia do Híbrido.
    """

    CACHE_SIZE = 256

    def __init__(self, llm: LLMTranslator | None = None) -> None:
        self.llm = llm if llm is not None else LLMTranslator()
        self.health = ProviderHealth()
        self._cache: OrderedDict[tuple[str, str, str, str, str, bool], TranslationResult] = OrderedDict()
        self._unconfirmed: dict[tuple[str, str, str, str, str, bool], tuple[str, str, str, str]] = {}
        self._cache_lock = threading.Lock()

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
        """``speculative`` adianta a tradução de uma frase que talvez ainda continue.

        O resultado vai para o cache, e a versão confirmada sai dele na hora; a
        memória da conversa do LLM só recebe a frase quando ela é confirmada.
        """
        target_code = LANGUAGES_DICT.get(target_language_name)
        if not target_code or target_code == "auto":
            raise TranslationError(f"Idioma de destino inválido: {target_language_name}")
        source_code = source_code or "auto"
        context = normalize_context(context)
        if language_base(source_code) == language_base(target_code):
            return TranslationResult(text=text, provider="sem tradução")

        cache_key = (text, language_base(source_code), target_language_name, mode, context, fragment)
        memory = None
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                if not speculative:
                    memory = self._unconfirmed.pop(cache_key, None)
        if cached is not None:
            if memory is not None:
                self.llm.remember(*memory)
            return cached
        effective_source = self._effective_source(text, source_code)
        result = self._translate_uncached(
            text,
            effective_source,
            target_language_name,
            target_code,
            mode,
            context,
            remember=not speculative,
            fragment=fragment,
        )
        with self._cache_lock:
            self._cache[cache_key] = result
            if speculative and result.provider == self.llm.label:
                self._unconfirmed[cache_key] = (effective_source, text, target_code, result.text)
            while len(self._cache) > self.CACHE_SIZE:
                evicted, _result = self._cache.popitem(last=False)
                self._unconfirmed.pop(evicted, None)
        return result

    def preview(self, text: str, source_code: str, target_language_name: str) -> str | None:
        """Tradução provisória para a legenda ao vivo.

        Só o Argos local: não gasta cota de nuvem, não entra na memória do LLM e
        uma falha aqui não derruba o provedor na fila da tradução de verdade.
        """
        target_code = LANGUAGES_DICT.get(target_language_name, "")
        source = self._effective_source(text, source_code or "auto")
        if not target_code or language_base(source) in {"auto", language_base(target_code)}:
            return None
        try:
            return self._translate_argos(text, source, target_code) or None
        except Exception:
            return None

    @staticmethod
    def _effective_source(text: str, source_code: str) -> str:
        if language_base(source_code) == "auto":
            return detect_idiom_language(text) or source_code
        return source_code

    def _translate_uncached(
        self,
        text: str,
        effective_source: str,
        target_language_name: str,
        target_code: str,
        mode: str,
        context: str,
        remember: bool = True,
        fragment: bool = False,
    ) -> TranslationResult:
        if mode == "IA" and self.llm.configured and not self.health.is_open("llm"):
            started = time.perf_counter()
            try:
                translated = self.llm.translate(
                    text,
                    effective_source,
                    target_code,
                    target_language_name,
                    context,
                    _slang_hints(text, effective_source, target_code, context),
                    remember=remember,
                    fragment=fragment,
                )
                self.health.success("llm", (time.perf_counter() - started) * 1000)
                if not translated:
                    return TranslationResult("", self.llm.label)
                return TranslationResult(translated, self.llm.label)
            except Exception as exc:
                self.health.failure("llm")
                logger.warning("LLM indisponível; usando a cadeia do modo Híbrido: %s", exc)

        provider = {"name": ""}

        def machine(chunk: str) -> str:
            result = self._translate_providers(chunk, effective_source, target_language_name, mode)
            provider["name"] = result.provider
            return result.text

        rendered = render_cultural(text, effective_source, target_code, context, machine)
        if rendered.fully_resolved:
            return TranslationResult(rendered.text, "equivalência cultural")
        name = provider["name"] or "tradução"
        if rendered.used_idiom:
            name = f"{name} + equivalência"
        return TranslationResult(rendered.text, name)

    def _translate_providers(
        self,
        text: str,
        source_code: str,
        target_language_name: str,
        mode: str,
    ) -> TranslationResult:
        target_code = LANGUAGES_DICT.get(target_language_name)
        if not target_code or target_code == "auto":
            raise TranslationError(f"Idioma de destino inválido: {target_language_name}")
        source_code = source_code or "auto"
        if language_base(source_code) == language_base(target_code):
            return TranslationResult(text=text, provider="sem tradução")

        chain: list[tuple[str, Callable[[], str]]] = []
        if mode in LOCAL_FIRST_MODES:
            chain.append(("Argos local", lambda: self._translate_argos(text, source_code, target_code)))
        if mode != "Local":
            api_key = os.getenv("DEEPL_API_KEY", "").strip()
            if api_key:
                chain.append(("DeepL", lambda: self._translate_deepl(text, source_code, target_code, api_key)))
            chain.append(("Google", lambda: self._translate_google(text, source_code, target_code)))

        errors: list[str] = []
        for name, call in self.health.order(chain, name=lambda entry: entry[0]):
            started = time.perf_counter()
            try:
                translated = call()
                if not translated:
                    raise TranslationError("resposta vazia")
            except Exception as exc:
                self.health.failure(name)
                errors.append(f"{name}: {exc}")
                if mode == "Local":
                    raise TranslationError(
                        "O idioma ainda não foi baixado. Abra Pacotes de idiomas "
                        "e instale o idioma falado e o idioma de destino."
                    ) from exc
                continue
            self.health.success(name, (time.perf_counter() - started) * 1000)
            return TranslationResult(translated, name)
        logger.warning("Falha nos provedores de tradução: %s", "; ".join(errors))
        raise TranslationError("Não foi possível traduzir. Verifique a internet ou o modo local.")

    @staticmethod
    def _translate_deepl(text: str, source_code: str, target_code: str, api_key: str) -> str:
        options = {
            "api_key": api_key,
            "target": {"zh-cn": "zh"}.get(target_code.lower(), target_code),
            "use_free_api": os.getenv("DEEPL_USE_FREE", "true").lower() in {"1", "true", "yes"},
        }
        if source_code != "auto":
            options["source"] = {"zh-cn": "zh"}.get(source_code.lower(), source_code)
        return DeeplTranslator(**options).translate(text)

    @staticmethod
    def _translate_google(text: str, source_code: str, target_code: str) -> str:
        return GoogleTranslator(source=source_code, target=target_code).translate(text)

    @staticmethod
    def _translate_argos(text: str, source_code: str, target_code: str) -> str:
        text = tidy_for_machine(text)
        if source_code == "auto":
            raise TranslationError("Argos requer idioma de origem conhecido")
        import argostranslate.translate

        source = source_code.lower().split("-")[0]
        target = target_code.lower().split("-")[0]
        installed = argostranslate.translate.get_installed_languages()
        source_language = next((item for item in installed if item.code == source), None)
        target_language = next((item for item in installed if item.code == target), None)
        if source_language is None or target_language is None:
            raise TranslationError(f"idioma local ausente: {source}->{target}")
        translation = source_language.get_translation(target_language)
        if translation is not None:
            result = translation.translate(text)
        else:
            english = next((item for item in installed if item.code == "en"), None)
            if english is None:
                raise TranslationError(f"par {source}->{target} não instalado")
            source_to_english = source_language.get_translation(english)
            english_to_target = english.get_translation(target_language)
            if source_to_english is None or english_to_target is None:
                raise TranslationError(f"rota local {source}->en->{target} não instalada")
            result = english_to_target.translate(source_to_english.translate(text))
        if not result:
            raise TranslationError("resposta vazia")
        if target == "pt":
            result = to_brazilian(result)
        return _keep_untranslated_tail(text, result, target)

    def warm_up(self, pairs: list[tuple[str, str]], mode: str) -> None:
        """Carrega os modelos Argos dos pares da sessão antes da primeira fala."""
        if mode == "IA":
            self.llm.warm_up()
        if mode not in LOCAL_FIRST_MODES:
            return
        for source_code, target_name in pairs:
            target_code = LANGUAGES_DICT.get(target_name, "")
            if language_base(source_code) in {"auto", language_base(target_code)} or not target_code:
                continue
            try:
                self._translate_argos("ok", source_code, target_code)
            except Exception as exc:
                logger.info("Aquecimento Argos %s->%s ignorado: %s", source_code, target_code, exc)

    def close(self) -> None:
        self.llm.close()

    @classmethod
    def translate_text(cls, text: str, target_language_name: str) -> str:
        """Compatibilidade com integrações antigas."""
        return cls().translate(text, "pt", target_language_name, "Nuvem").text
