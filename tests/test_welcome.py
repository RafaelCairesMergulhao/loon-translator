from ui.welcome import last_section_hint_key


def test_welcome_hint_treats_train_apart_from_translation() -> None:
    assert last_section_hint_key("train") == "welcome_last_train"
    assert last_section_hint_key("tests") == "welcome_last_tests"
    assert last_section_hint_key("translate") == "welcome_last_translate"
    assert last_section_hint_key("history") == "welcome_last_history"
    assert last_section_hint_key("unknown") == "welcome_last_translate"
