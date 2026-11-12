from ui.theme import apply_theme, canonical_theme


def test_unknown_appearance_stays_on_night() -> None:
    assert canonical_theme("arco-íris") == "night"
    target = type("Surface", (), {})()
    assert apply_theme(target, "não-existe") == "night"
    assert target.BG == "#080d14"
    assert target.TEXT == "#eaf1f8"


def test_paper_is_a_light_named_theme() -> None:
    target = type("Surface", (), {})()
    assert apply_theme(target, "paper") == "paper"
    assert target.BG == "#f3f6fb"
    assert target.TEXT == "#122033"


def test_only_night_and_paper_remain() -> None:
    from ui.theme import THEMES, uses_light_chrome

    assert set(THEMES) == {"night", "paper"}
    assert apply_theme(type("Surface", (), {})(), "glacier") == "night"
    assert uses_light_chrome("paper")
    assert uses_light_chrome("night") is False
