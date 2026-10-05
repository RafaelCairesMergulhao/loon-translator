from ui.i18n import (
    LOCALES,
    STRINGS,
    canonical_choice,
    canonical_language,
    language_label,
    ui_text,
)
from ui.i18n import CONTEXT_LABEL_KEYS, MODE_LABEL_KEYS, VOICE_ENGINE_KEYS


def test_every_string_exists_in_every_locale() -> None:
    for key, values in STRINGS.items():
        missing = [code for code in LOCALES if not values.get(code)]
        assert not missing, (key, missing)


def test_language_names_round_trip_in_each_locale() -> None:
    for ui_language in ("Português", "Inglês", "Japonês", "Árabe"):
        for canonical in ("Português", "Inglês", "Espanhol (Latino)", "Chinês (Mandarim)"):
            assert canonical_language(language_label(canonical, ui_language), ui_language) == canonical


def test_placeholders_survive_translation() -> None:
    for code in LOCALES:
        assert "{seconds}" in STRINGS["status_ready"][code]
        assert "{mic}" in STRINGS["tests_devices"][code]
        assert "{phones}" in STRINGS["tests_devices"][code]
        assert "{cable}" in STRINGS["tests_devices"][code]
        assert "{languages}" in STRINGS["dialog_download_body"][code]
        assert "{language}" in STRINGS["pack_ready_for"][code]
        assert "{message}" in STRINGS["error_clear"][code]


def test_mode_and_context_labels_map_back() -> None:
    assert canonical_choice("Hybrid", MODE_LABEL_KEYS, "Inglês", "Local") == "Híbrido"
    assert canonical_choice("Lokal", MODE_LABEL_KEYS, "Alemão", "Local") == "Local"
    assert (
        canonical_choice(ui_text("Francês", "context_travel"), CONTEXT_LABEL_KEYS, "Francês", "casual")
        == "travel"
    )
    assert canonical_choice("Natural online (recommended)", VOICE_ENGINE_KEYS, "Inglês", "windows") == "natural"
    assert canonical_choice("Local neural (Piper)", VOICE_ENGINE_KEYS, "Inglês", "natural") == "piper"
