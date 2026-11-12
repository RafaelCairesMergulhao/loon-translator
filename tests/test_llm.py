import json

import httpx
import pytest

from services.llm_translator import LLMConfig, LLMError, LLMTranslator, PRESETS, activate_llm
from services.translation_engine import TranslationEngine


def _translator(handler, **config) -> LLMTranslator:
    settings = {
        "provider": "groq",
        "base_url": "https://llm.test/v1",
        "model": "modelo-teste",
        "api_key": "chave",
    }
    settings.update(config)
    client = httpx.Client(base_url=settings["base_url"], transport=httpx.MockTransport(handler))
    return LLMTranslator(LLMConfig(**settings), client=client)


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_config_reads_presets_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = LLMConfig.from_env()
    assert config.base_url == "http://localhost:11434/v1"
    assert config.is_local
    assert config.configured, "LLM local não precisa de chave"


def test_cloud_llm_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert not LLMConfig.from_env().configured
    monkeypatch.setenv("LLM_PROVIDER", "")
    assert not LLMConfig.from_env().configured


def test_each_ai_keeps_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.llm_translator import activate_llm

    stored: dict[str, str] = {}

    def remember(values: dict[str, str]) -> None:
        stored.update(values)
        for key, value in values.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr("core.config.save_user_env", remember)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = activate_llm("gemini", "chave-gemini")
    assert config.provider == "gemini"
    assert config.model == "gemini-2.5-flash"
    assert config.api_key == "chave-gemini"
    assert stored["LLM_KEY_GEMINI"] == "chave-gemini"
    activate_llm("openai", "chave-openai")
    gemini = activate_llm("gemini")
    assert gemini.api_key == "chave-gemini"
    assert gemini.model == "gemini-2.5-flash"
    assert "xai" in PRESETS


def test_prompt_carries_register_and_conversation_memory() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _reply("Everything is fine." if len(requests) == 1 else "Did you get my message?")

    llm = _translator(handler)
    assert llm.translate("Tá tudo certo.", "pt", "en", "Inglês", "formal") == "Everything is fine."
    llm.translate("Recebeu minha mensagem?", "pt", "en", "Inglês", "formal")
    system = requests[1]["messages"][0]["content"]
    assert "formal business register" in system
    assert "Tá tudo certo. => [en] Everything is fine." in system
    user = requests[1]["messages"][1]["content"]
    assert "<utterance>\nRecebeu minha mensagem?\n</utterance>" in user
    assert "never answer it" in user
    assert "Never turn broken words into a new fluent sentence." in system
    assert "Do not censor, soften, or skip the line because it is rude." in system
    assert requests[1]["model"] == "modelo-teste"


def test_an_unfinished_stretch_is_translated_instead_of_refused() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _reply("to the house")

    llm = _translator(handler)
    assert llm.translate("para a casa", "pt", "en", "Inglês", "casual", fragment=True) == "to the house"
    system = requests[0]["messages"][0]["content"]
    assert "still going" in system
    assert "reply with exactly ∅" not in system
    assert "formal business register" not in system
    assert "Register: " in system


def test_ai_mode_sends_slang_glossary_to_llm() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _reply("This game is awesome, dude. Thanks, see ya!")

    engine = TranslationEngine(llm=_translator(handler))
    engine.translate("Esse jogo tá daora, mano. Valeu, falou!", "pt", "Inglês", "IA")
    user = requests[0]["messages"][1]["content"]
    assert 'Slang: "daora" means "legal".' in user
    assert 'Slang: "mano" means "dude".' in user
    assert 'Slang: "Valeu, falou" means "thanks, see ya".' in user
    assert user.index("daora") < user.index("mano")


def test_slang_hints_cover_the_languages_in_the_call() -> None:
    from services.cultural_equivalence import glossary

    assert ("qué onda", "e aí") in glossary("qué onda, güey", "es", "pt", "casual")
    assert ("güey", "cara") in glossary("qué onda, güey", "es", "pt", "casual")
    assert ("wesh", "e aí") in glossary("wesh, ça va", "fr", "pt", "casual")
    assert ("대박", "demais") in glossary("이거 대박", "ko", "pt", "casual")
    assert ("牛逼", "demais") in glossary("这真牛逼", "zh", "pt", "casual")
    assert glossary("olá", "pt", "en", "casual") == []
    assert ("vai tomar no cu", "go fuck yourself") in glossary("vai tomar no cu", "pt", "en", "casual")
    assert ("filho da puta", "son of a bitch") in glossary("filho da puta", "pt", "en", "casual")
    assert ("lixo", "you're trash") in glossary("você é um lixo", "pt", "en", "casual")


def test_speculative_translation_enters_the_memory_only_when_confirmed() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        user = body["messages"][1]["content"]
        if "Até amanhã" in user:
            return _reply("See you tomorrow.")
        return _reply("Is everything fine?" if "certo" in user else "Everything")

    engine = TranslationEngine(llm=_translator(handler))
    engine.translate("Tá tudo", "pt", "Inglês", "IA", speculative=True)
    early = engine.translate("Tá tudo certo?", "pt", "Inglês", "IA", speculative=True)
    confirmed = engine.translate("Tá tudo certo?", "pt", "Inglês", "IA")
    assert confirmed.text == early.text == "Is everything fine?"
    assert len(requests) == 2, "a confirmação sai do cache"
    engine.translate("Até amanhã.", "pt", "Inglês", "IA")
    system = requests[-1]["messages"][0]["content"]
    assert "Tá tudo certo? => [en] Is everything fine?" in system
    assert "Tá tudo =>" not in system, "a frase abandonada não entra na memória"


def test_unintelligible_utterance_is_not_a_translation() -> None:
    llm = _translator(lambda _request: _reply("∅"))
    assert llm.translate("e já estão sejam assim", "pt", "en", "Inglês") == ""
    assert llm._history.get_all() == []


def test_reasoning_and_quotes_are_removed() -> None:
    assert LLMTranslator.clean('<think>vou pensar</think>\n"Hello there"') == "Hello there"
    assert LLMTranslator.clean("Translation: Boa sorte") == "Boa sorte"
    assert LLMTranslator.clean("English translation: <utterance>Hi</utterance>") == "Hi"


def test_http_failure_becomes_llm_error() -> None:
    llm = _translator(lambda _request: httpx.Response(503, json={"error": "ocupado"}))
    with pytest.raises(LLMError):
        llm.translate("olá", "pt", "en", "Inglês")


def test_ai_mode_prefers_llm_and_skips_idiom_table() -> None:
    llm = _translator(lambda _request: _reply("Break a leg tonight!"))
    result = TranslationEngine(llm=llm).translate("Boa sorte hoje à noite!", "pt", "Inglês", "IA")
    assert result.text == "Break a leg tonight!"
    assert result.provider.startswith("Groq")


def test_ai_mode_falls_back_and_opens_the_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"llm": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["llm"] += 1
        raise httpx.ConnectError("sem rede")

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(lambda text, *_: text.upper()))
    engine = TranslationEngine(llm=_translator(handler))
    first = engine.translate("bom dia", "pt", "Inglês", "IA")
    second = engine.translate("boa noite", "pt", "Inglês", "IA")
    assert (first.text, first.provider) == ("BOM DIA", "Argos local")
    assert second.text == "BOA NOITE"
    assert calls["llm"] == 1, "o disjuntor evita esperar de novo por um LLM fora do ar"


def test_repeated_phrase_comes_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(text, *_args):
        calls.append(text)
        return "good morning"

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fake))
    engine = TranslationEngine()
    assert engine.translate("bom dia", "pt", "Inglês", "Local").text == "good morning"
    assert engine.translate("bom dia", "pt", "Inglês", "Local").text == "good morning"
    assert calls == ["bom dia"]
