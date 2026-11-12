import numpy as np

from services.providers.asr import parse_deepgram, parse_watson
from services.providers.s2s import session_update
from services.providers.stack import SentenceGate, apply_offline, resolve_profile, translation_mode_for, voice_engine_for
from services.providers.translate import glossary_block


def test_profiles_keep_the_current_screen_until_one_is_chosen() -> None:
    current = resolve_profile("")
    assert translation_mode_for(current, "Local") == "Local"
    assert voice_engine_for(current, "piper") == "piper"
    hybrid = resolve_profile("hybrid")
    assert (hybrid.asr, translation_mode_for(hybrid, "Local"), voice_engine_for(hybrid, "piper")) == (
        "faster-whisper",
        "Nuvem",
        "natural",
    )
    cloud = resolve_profile("cloud")
    assert (cloud.asr, translation_mode_for(cloud, "Local"), voice_engine_for(cloud, "piper")) == (
        "deepgram",
        "IA",
        "elevenlabs",
    )
    assert resolve_profile("s2s").s2s == "openai"
    assert resolve_profile("não existe").asr == "faster-whisper"


def test_offline_choices_stay_on_this_machine() -> None:
    stack, model = apply_offline("local", "Local", "auto", "alto", "nllb", "xtts")
    assert (stack.asr, stack.mt, stack.tts, stack.s2s, model) == (
        "faster-whisper",
        "nllb",
        "xtts",
        "",
        "balanced",
    )
    untouched, same = apply_offline("", "Híbrido", "auto", "alto", "nllb", "xtts")
    assert untouched.mt == "follow"
    assert same == "auto"
    light, base = apply_offline("", "Local", "accurate", "", "", "")
    assert light.mt == "follow"
    assert base == "accurate"


def test_nllb_without_a_local_model_falls_back_to_argos(monkeypatch) -> None:
    from services.providers.translate import RoutedTranslator
    from services.translation_engine import TranslationResult

    class Inner:
        llm = None

        def translate(self, *_args, **_kwargs):
            return TranslationResult("from argos", "Argos local")

        def preview(self, *_args, **_kwargs):
            return None

        def warm_up(self, *_args, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr("services.providers.translate.nllb_ready", lambda root=None: False)
    result = RoutedTranslator(Inner(), "nllb").translate("oi", "pt", "Inglês", "Local")
    assert result.text == "from argos"
    assert "NLLB ausente" in result.provider


def test_a_broken_nllb_falls_back_instead_of_stopping_the_call(monkeypatch) -> None:
    from services.providers.translate import RoutedTranslator
    from services.translation_engine import TranslationResult

    class Inner:
        llm = None

        def translate(self, *_args, **_kwargs):
            return TranslationResult("from argos", "Argos local")

        def preview(self, *_args, **_kwargs):
            return None

        def warm_up(self, *_args, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    def boom(*_args, **_kwargs):
        raise RuntimeError("modelo ilegível")

    monkeypatch.setattr("services.providers.translate.nllb_translate", boom)
    result = RoutedTranslator(Inner(), "nllb").translate("oi", "pt", "Inglês", "Local")
    assert result.text == "from argos"
    assert "NLLB ausente" in result.provider


def test_the_voice_waits_for_the_sentence_to_close() -> None:
    gate = SentenceGate()
    assert gate.caption("partial") is True
    assert gate.speak("partial") is False
    assert gate.speak("final") is True


def test_deepgram_and_watson_payloads_become_a_transcript() -> None:
    heard = parse_deepgram(
        {"results": {"channels": [{"detected_language": "pt", "alternatives": [{"transcript": "olá", "confidence": 0.9}]}]}},
        None,
    )
    assert (heard.text, heard.language) == ("olá", "pt")
    watson = parse_watson(
        {"results": [{"alternatives": [{"transcript": "bom dia", "confidence": 0.8}]}]},
        "pt",
    )
    assert watson.text == "bom dia"


def test_glossary_lines_name_the_term_and_its_meaning() -> None:
    assert glossary_block([("Loon", "the app")]) == 'Slang: "Loon" means "the app".\n'


def test_direct_session_asks_only_for_the_target_language() -> None:
    payload = session_update("Português", "Inglês")
    instructions = payload["session"]["instructions"]
    assert "Português" in instructions
    assert "Inglês" in instructions
    assert payload["session"]["turn_detection"] is None


def test_pcm_wav_is_sixteen_bit_mono() -> None:
    from services.providers.asr import pcm_wav
    import io
    import wave

    wav = pcm_wav(np.zeros(160, dtype=np.float32))
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getsampwidth() == 2
