"""Duas aparências. A tradução não muda com a cor da tela."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    bg: str
    panel: str
    field: str
    ghost: str
    text: str
    muted: str
    border: str
    blue: str
    blue_hover: str
    ink: str
    green: str
    sidebar: str
    nav_active: str
    nav_border: str
    hover: str
    chip: str


THEMES: dict[str, Theme] = {
    "night": Theme(
        bg="#080d14",
        panel="#0d1620",
        field="#101a25",
        ghost="#111c26",
        text="#eaf1f8",
        muted="#7f8e9e",
        border="#1c2b38",
        blue="#39d7e4",
        blue_hover="#6de4f1",
        ink="#031216",
        green="#45e5ba",
        sidebar="#090e16",
        nav_active="#102b36",
        nav_border="#1c3845",
        hover="#111a25",
        chip="#0d201e",
    ),
    "paper": Theme(
        bg="#f3f6fb",
        panel="#ffffff",
        field="#e7eef6",
        ghost="#e7eef6",
        text="#122033",
        muted="#4d5d70",
        border="#d3dde8",
        blue="#0c7484",
        blue_hover="#1496ab",
        ink="#f4fbfc",
        green="#0c6b45",
        sidebar="#e6eef6",
        nav_active="#d5eef2",
        nav_border="#b7d5dc",
        hover="#dce6f0",
        chip="#e5f6ef",
    ),
}

_FIELDS = (
    ("BG", "bg"),
    ("PANEL", "panel"),
    ("FIELD", "field"),
    ("GHOST", "ghost"),
    ("TEXT", "text"),
    ("MUTED", "muted"),
    ("BORDER", "border"),
    ("BLUE", "blue"),
    ("BLUE_HOVER", "blue_hover"),
    ("INK", "ink"),
    ("GREEN", "green"),
    ("SIDEBAR", "sidebar"),
    ("NAV_ACTIVE", "nav_active"),
    ("NAV_BORDER", "nav_border"),
    ("HOVER", "hover"),
    ("CHIP", "chip"),
)


def canonical_theme(name: str) -> str:
    return name if name in THEMES else "night"


def uses_light_chrome(name: str) -> bool:
    return canonical_theme(name) == "paper"


def apply_theme(target, name: str) -> str:
    chosen = canonical_theme(name)
    theme = THEMES[chosen]
    for attr, field in _FIELDS:
        setattr(target, attr, getattr(theme, field))
    return chosen
