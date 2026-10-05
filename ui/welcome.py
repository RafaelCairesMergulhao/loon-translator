from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk

from services.translation_engine import LANGUAGES_DICT
from ui.i18n import canonical_language, language_choices, language_label, ui_text


class WelcomeScreen(ctk.CTkFrame):
    """Abertura: idioma e a área em que a pessoa quer entrar."""

    def __init__(self, master, app, on_start: Callable[[str, str], None]) -> None:
        super().__init__(master, fg_color=app.BG, corner_radius=0)
        self.app = app
        self.on_start = on_start
        initial = app.settings.native_language
        if initial not in LANGUAGES_DICT or initial == "Auto":
            initial = "Português"
        self._canonical = initial
        self._ui_language = self._canonical
        self._updating = False
        self._locked = False
        suggested = app.settings.last_section if app.settings.last_section in {"translate", "tests"} else "translate"
        self._suggested = suggested

        card = ctk.CTkFrame(
            self,
            fg_color=app.PANEL,
            corner_radius=18,
            border_width=1,
            border_color=app.BORDER,
            width=680,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        self.card = card

        ctk.CTkLabel(
            card,
            text="L",
            width=52,
            height=52,
            corner_radius=14,
            fg_color=app.BLUE,
            text_color="white",
            font=ctk.CTkFont("Segoe UI", 26, "bold"),
        ).pack(anchor="w", padx=32, pady=(28, 14))
        self.title = ctk.CTkLabel(
            card,
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 26, "bold"),
            wraplength=600,
            justify="left",
        )
        self.title.pack(anchor="w", padx=32)
        self.subtitle = ctk.CTkLabel(
            card,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 14),
            wraplength=600,
            justify="left",
        )
        self.subtitle.pack(anchor="w", padx=32, pady=(6, 18))
        self.language_caption = ctk.CTkLabel(
            card,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        )
        self.language_caption.pack(anchor="w", padx=32)
        self.language_var = ctk.StringVar()
        self.combo = ctk.CTkComboBox(
            card,
            variable=self.language_var,
            values=language_choices(self._ui_language, include_auto=False),
            command=self._on_language,
            state="readonly",
            height=40,
            width=600,
            corner_radius=9,
            border_color=app.BORDER,
            fg_color="#F8FAFC",
            button_color="#E8EEF8",
            button_hover_color="#DDE6F4",
            text_color=app.TEXT,
            dropdown_fg_color="white",
            dropdown_text_color=app.TEXT,
        )
        self.combo.pack(anchor="w", padx=32, pady=(6, 16))
        self.choose = ctk.CTkLabel(
            card,
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
        )
        self.choose.pack(anchor="w", padx=32)
        self.last_hint = ctk.CTkLabel(
            card,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        self.last_hint.pack(anchor="w", padx=32, pady=(2, 12))

        choices = ctk.CTkFrame(card, fg_color="transparent")
        choices.pack(fill="x", padx=32, pady=(0, 28))
        choices.grid_columnconfigure((0, 1), weight=1, uniform="welcome")
        self.translate_button = self._choice(
            choices,
            0,
            "translate",
            primary=suggested != "tests",
        )
        self.tests_button = self._choice(
            choices,
            1,
            "tests",
            primary=suggested == "tests",
        )
        self._refresh_copy()

    def _choice(self, parent, column: int, section: str, primary: bool) -> ctk.CTkButton:
        app = self.app
        frame = ctk.CTkFrame(
            parent,
            fg_color="#F8FAFC",
            corner_radius=14,
            border_width=1,
            border_color=app.BLUE if primary else app.BORDER,
        )
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
        button = ctk.CTkButton(
            frame,
            command=lambda: self._choose(section),
            height=42,
            corner_radius=10,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        )
        button.pack(fill="x", padx=14, pady=(14, 8))
        hint = ctk.CTkLabel(
            frame,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=250,
            justify="left",
        )
        hint.pack(anchor="w", padx=14, pady=(0, 14))
        self._style_button(button, primary)
        setattr(self, f"{section}_hint", hint)
        setattr(self, f"{section}_frame", frame)
        return button

    def _style_button(self, button: ctk.CTkButton, primary: bool) -> None:
        app = self.app
        if primary:
            button.configure(
                fg_color=app.BLUE,
                hover_color=app.BLUE_HOVER,
                text_color="white",
            )
        else:
            button.configure(
                fg_color="#EEF2F7",
                hover_color="#E2E8F0",
                text_color=app.TEXT,
            )

    def _refresh_copy(self) -> None:
        language = self._ui_language
        self.title.configure(text=ui_text(language, "welcome_title"))
        self.subtitle.configure(text=ui_text(language, "welcome_body"))
        self.language_caption.configure(text=ui_text(language, "welcome_language"))
        self.choose.configure(text=ui_text(language, "welcome_choose"))
        if self.app.settings.onboarding_complete:
            key = "welcome_last_tests" if self._suggested == "tests" else "welcome_last_translate"
            self.last_hint.configure(text=ui_text(language, key))
        else:
            self.last_hint.configure(text="")
        self.translate_button.configure(text=ui_text(language, "welcome_translate"))
        self.tests_button.configure(text=ui_text(language, "welcome_tests"))
        self.translate_hint.configure(text=ui_text(language, "welcome_translate_hint"))
        self.tests_hint.configure(text=ui_text(language, "welcome_tests_hint"))
        self._updating = True
        try:
            self.combo.configure(values=language_choices(language, include_auto=False))
            self.language_var.set(language_label(self._canonical, language))
        finally:
            self._updating = False

    def _on_language(self, label: str) -> None:
        if self._updating or self._locked:
            return
        canonical = canonical_language(label, self._ui_language)
        if canonical == "Auto":
            return
        self._canonical = canonical
        self._ui_language = canonical
        self._refresh_copy()

    def _choose(self, section: str) -> None:
        if self._locked:
            return
        self._locked = True
        self.translate_button.configure(state="disabled")
        self.tests_button.configure(state="disabled")
        self.combo.configure(state="disabled")
        self.on_start(section, self._canonical)

    def lock(self) -> None:
        self._locked = True
