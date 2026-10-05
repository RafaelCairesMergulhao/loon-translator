import pytest
from dataclasses import dataclass
from pathlib import Path

from services.language_packs import LanguagePackManager
from services.translation_engine import (
    LANGUAGES_DICT,
    VOICE_MAP,
    TranslationEngine,
    TranslationError,
)


def test_launch_catalog_contains_twelve_core_languages() -> None:
    assert len(LanguagePackManager.LAUNCH_LANGUAGES) == 12
    for language in LanguagePackManager.LAUNCH_LANGUAGES:
        assert language in LANGUAGES_DICT
        assert language in VOICE_MAP


def test_hindi_and_arabic_are_supported() -> None:
    assert LANGUAGES_DICT["Hindi"] == "hi"
    assert LANGUAGES_DICT["Árabe"] == "ar"


@dataclass
class FakePackage:
    from_code: str
    to_code: str

    def download(self):
        return Path(f"{self.from_code}-{self.to_code}.argosmodel")


def test_language_pack_installs_both_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = []

    class FakePackageModule:
        @staticmethod
        def update_package_index():
            return None

        @staticmethod
        def get_installed_packages():
            return []

        @staticmethod
        def get_available_packages():
            return [FakePackage("hi", "en"), FakePackage("en", "hi")]

        @staticmethod
        def install_from_path(path):
            installed.append(path.name)

    monkeypatch.setattr(
        LanguagePackManager,
        "_package_module",
        staticmethod(lambda: FakePackageModule),
    )
    LanguagePackManager().install("Hindi")
    assert installed == ["hi-en.argosmodel", "en-hi.argosmodel"]


def test_same_language_skips_provider() -> None:
    result = TranslationEngine().translate("olá", "pt", "Português", "Híbrido")
    assert result.text == "olá"
    assert result.provider == "sem tradução"


def test_hybrid_prefers_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TranslationEngine,
        "_translate_argos",
        staticmethod(lambda text, source, target: "hello"),
    )
    result = TranslationEngine().translate("olá", "pt", "Inglês", "Híbrido")
    assert result.text == "hello"
    assert result.provider == "Argos local"


def test_local_mode_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args):
        raise RuntimeError("não instalado")

    monkeypatch.setattr(TranslationEngine, "_translate_argos", staticmethod(fail))
    with pytest.raises(TranslationError, match="ainda não foi baixado"):
        TranslationEngine().translate("olá", "pt", "Inglês", "Local")


def test_missing_languages_skip_english_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePackageModule:
        @staticmethod
        def get_installed_packages():
            return []

    monkeypatch.setattr(
        LanguagePackManager,
        "_package_module",
        staticmethod(lambda: FakePackageModule),
    )
    missing = LanguagePackManager().missing(["Auto", "Inglês", "Português", "Hindi"])
    assert missing == ["Português", "Hindi"]
