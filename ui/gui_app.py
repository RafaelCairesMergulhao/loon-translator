from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.config import SettingsStore, save_user_env
from core.pipeline import DuplexTranslationPipeline, PipelineOptions
from domain.entities import AudioDirection, TranslationRecord
from infrastructure.history_store import HistoryStore
from services.audio_devices import AudioDevice, AudioDeviceManager
from services.cultural_equivalence import normalize_context
from services.language_packs import LanguagePackManager
from services.llm_translator import LLMConfig
from services.translation_engine import LANGUAGES_DICT
from services.tts_engine import TTSEngine
from ui.i18n import (
    CONTEXT_LABEL_KEYS,
    GENDER_LABEL_KEYS,
    MODE_LABEL_KEYS,
    STT_LABEL_KEYS,
    VOICE_ENGINE_KEYS,
    canonical_choice,
    canonical_language,
    choice_label,
    choice_values,
    language_choices,
    language_label,
    ui_text,
)
from ui.tests_view import TestsView
from ui.tips import TipsCard, open_tutorial
from ui.training_view import TrainingView
from ui.welcome import WelcomeScreen

logger = logging.getLogger(__name__)


class LoonApp:
    BG = "#F4F7FB"
    PANEL = "#FFFFFF"
    TEXT = "#152238"
    MUTED = "#667085"
    BORDER = "#E4E9F2"
    BLUE = "#2563EB"
    BLUE_HOVER = "#1D4ED8"
    GREEN = "#059669"
    RED = "#DC2626"

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Loon Translator — Windows")
        self.root.geometry("1180x780")
        self.root.minsize(1040, 700)
        self.root.configure(fg_color=self.BG)
        self._center_window()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(900, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.history_store = HistoryStore()
        self.devices = AudioDeviceManager()
        self.pipeline: DuplexTranslationPipeline | None = None
        self._events: queue.SimpleQueue[tuple[int, str, tuple]] = queue.SimpleQueue()
        self._generation = 0
        self._stopping = False
        self._closing = False
        self._shown_errors: set[str] = set()
        self._building_ui = False
        self.device_maps: dict[str, dict[str, AudioDevice]] = {}
        self.welcome: WelcomeScreen | None = None
        self.shell: ctk.CTkFrame | None = None
        self.tests_view: TestsView | None = None
        self.training_view: TrainingView | None = None
        self._section = "translate"
        self._transitioning = False
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build_main()
        self.refresh_devices()
        self._show_welcome()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._drain_events)

    def _center_window(self) -> None:
        width, height = 1180, 780
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def t(self, key: str) -> str:
        return ui_text(self.settings.native_language, key)

    def _fill(self, key: str, **values: str) -> str:
        text = self.t(key)
        for name, value in values.items():
            text = text.replace("{" + name + "}", value)
        return text

    def _show_welcome(self) -> None:
        self.welcome = WelcomeScreen(self.root, self, self.begin_from_welcome)
        self.welcome.grid(row=0, column=0, sticky="nsew")
        self.welcome.lift()

    def begin_from_welcome(self, section: str, language_name: str) -> None:
        if self._transitioning:
            return
        self._transitioning = True
        previous_language = self.settings.native_language
        first_run = not self.settings.onboarding_complete
        self.settings.native_language = language_name
        self.settings.last_section = section if section in {"translate", "tests", "train"} else "translate"
        if first_run:
            self.settings.onboarding_complete = True
            self.settings.source_language = language_name
            self.settings.incoming_language = language_name
            if self.settings.outgoing_language == language_name:
                self.settings.outgoing_language = (
                    "Português" if language_name == "Inglês" else "Inglês"
                )
        self.settings_store.save(self.settings)
        if previous_language != language_name:
            self._build_main()
            self.refresh_devices()
        self._show_section(self.settings.last_section)
        if self.welcome is not None:
            self.welcome.lift()
            self._slide_welcome_away()
        else:
            self._transitioning = False

    def _slide_welcome_away(self) -> None:
        welcome = self.welcome
        if welcome is None:
            self._transitioning = False
            return
        welcome.grid_forget()
        welcome.place(relx=0, rely=0, relwidth=1, relheight=1)
        welcome.lift()
        steps = 18

        def tick(index: int = 0) -> None:
            if self.welcome is None:
                self._transitioning = False
                return
            if index >= steps:
                self.welcome.destroy()
                self.welcome = None
                self._transitioning = False
                self.settings.last_section = self._section
                self.settings_store.save(self.settings)
                return
            progress = (index + 1) / steps
            eased = progress * progress * (3 - 2 * progress)
            self.welcome.place(relx=0, rely=-eased, relwidth=1, relheight=1)
            self.root.after(16, lambda: tick(index + 1))

        tick()

    def _show_section(self, section: str) -> None:
        self._section = section if section in {"translate", "tests", "train"} else "translate"
        if not hasattr(self, "tabs"):
            return
        labels = {"tests": "tab_tests", "train": "tab_train"}
        label = self.t(labels.get(self._section, "tab_translate"))
        self.tabs.set(label)
        self._switch_view(label)

    def _build_main(self) -> None:
        self._building_ui = True
        try:
            if self.tests_view is not None:
                self.tests_view.shutdown()
                self.tests_view = None
            if self.shell is not None:
                self.shell.destroy()
            if self.settings.native_language not in LANGUAGES_DICT or self.settings.native_language == "Auto":
                self.settings.native_language = "Português"
            self.shell = ctk.CTkFrame(self.root, fg_color=self.BG, corner_radius=0)
            self.shell.grid(row=0, column=0, sticky="nsew")
            self.shell.grid_columnconfigure(1, weight=1)
            self.shell.grid_rowconfigure(0, weight=1)

            sidebar = ctk.CTkFrame(
                self.shell,
                width=300,
                corner_radius=0,
                fg_color=self.PANEL,
                border_width=0,
            )
            sidebar.grid(row=0, column=0, sticky="nsew")
            sidebar.grid_propagate(False)
            self._build_sidebar(sidebar)

            content = ctk.CTkFrame(self.shell, fg_color="transparent")
            content.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
            content.grid_columnconfigure(0, weight=1)
            content.grid_rowconfigure(1, weight=1)

            header = ctk.CTkFrame(content, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
            titles = ctk.CTkFrame(header, fg_color="transparent")
            titles.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.header_title = ctk.CTkLabel(
                titles,
                text=self.t("header_title"),
                text_color=self.TEXT,
                font=ctk.CTkFont("Segoe UI", 26, "bold"),
            )
            self.header_title.pack(anchor="w")
            self.header_subtitle = ctk.CTkLabel(
                titles,
                text=self.t("header_subtitle"),
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 13),
                wraplength=560,
                justify="left",
            )
            self.header_subtitle.pack(anchor="w", pady=(3, 0))
            self.tabs = ctk.CTkSegmentedButton(
                header,
                values=[self.t("tab_translate"), self.t("tab_tests"), self.t("tab_train")],
                command=self._switch_view,
                height=36,
                corner_radius=10,
                font=ctk.CTkFont("Segoe UI", 13, "bold"),
                fg_color=self.BG,
                selected_color="#DBEAFE",
                selected_hover_color="#BFDBFE",
                unselected_color="#F8FAFC",
                unselected_hover_color="#E8EEF8",
                text_color=self.TEXT,
            )
            self.tabs.pack(side=tk.RIGHT, anchor="n", padx=(16, 0))
            self.tabs.set(self.t("tab_translate"))

            pages = ctk.CTkFrame(content, fg_color="transparent")
            pages.grid(row=1, column=0, sticky="nsew")
            pages.grid_columnconfigure(0, weight=1)
            pages.grid_rowconfigure(0, weight=1)

            self.translate_page = ctk.CTkFrame(pages, fg_color="transparent")
            self.translate_page.grid(row=0, column=0, sticky="nsew")
            self.translate_page.grid_columnconfigure((0, 1), weight=1, uniform="direction")
            self.translate_page.grid_rowconfigure(2, weight=1)
            TipsCard(self.translate_page, self, "translate").grid(
                row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10)
            )
            self._build_direction_panel(
                self.translate_page, 0, self.t("out_title"), self.t("out_subtitle"), "out"
            )
            self._build_direction_panel(
                self.translate_page, 1, self.t("in_title"), self.t("in_subtitle"), "in"
            )
            self._build_history(self.translate_page)

            self.tests_page = ctk.CTkFrame(pages, fg_color="transparent")
            self.tests_page.grid(row=0, column=0, sticky="nsew")
            self.tests_view = TestsView(self.tests_page, self)
            self.tests_view.pack(fill=tk.BOTH, expand=True)
            self.tests_page.grid_remove()

            self.train_page = ctk.CTkFrame(pages, fg_color="transparent")
            self.train_page.grid(row=0, column=0, sticky="nsew")
            self.training_view = TrainingView(self.train_page, self)
            self.training_view.pack(fill=tk.BOTH, expand=True)
            self.train_page.grid_remove()
        finally:
            self._building_ui = False

    def _switch_view(self, value: str) -> None:
        if self._building_ui or not hasattr(self, "tests_page"):
            return
        if value == self.t("tab_tests"):
            self._section = "tests"
        elif value == self.t("tab_train"):
            self._section = "train"
        else:
            self._section = "translate"
        self.translate_page.grid_remove()
        self.tests_page.grid_remove()
        self.train_page.grid_remove()
        if self._section != "tests" and self.tests_view is not None:
            self.tests_view.shutdown()
        if self._section == "tests":
            self.tests_page.grid()
            if self.tests_view is not None:
                self.tests_view.refresh_summary()
        elif self._section == "train":
            self.train_page.grid()
        else:
            self.translate_page.grid()
        self._apply_header()
        self._sync_mode_button()
        if not self._transitioning:
            self.settings.last_section = self._section
            self.settings_store.save(self.settings)

    def _apply_header(self) -> None:
        if not hasattr(self, "header_title"):
            return
        if self._section == "tests":
            self.header_title.configure(text=self.t("tests_title"))
            self.header_subtitle.configure(text=self.t("tests_subtitle"))
        elif self._section == "train":
            self.header_title.configure(text=self.t("train_title"))
            self.header_subtitle.configure(text=self.t("train_subtitle"))
        else:
            self.header_title.configure(text=self.t("header_title"))
            self.header_subtitle.configure(text=self.t("header_subtitle"))

    def _sync_mode_button(self) -> None:
        if not hasattr(self, "toggle_btn") or self._stopping:
            return
        if self.pipeline and self.pipeline.running:
            self.toggle_btn.configure(
                text=self.t("stop"),
                fg_color=self.RED,
                hover_color="#B91C1C",
                command=self.toggle,
                state=tk.NORMAL,
            )
            return
        if self._section in {"tests", "train"}:
            self.toggle_btn.configure(
                text=self.t("open_translation"),
                fg_color=self.BLUE,
                hover_color=self.BLUE_HOVER,
                command=self._open_translation,
                state=tk.NORMAL,
            )
            return
        self.toggle_btn.configure(
            text=self.t("start"),
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            command=self.toggle,
            state=tk.NORMAL,
        )

    def _open_translation(self) -> None:
        if hasattr(self, "tabs"):
            self.tabs.set(self.t("tab_translate"))
        self._switch_view(self.t("tab_translate"))

    def _build_sidebar(self, parent: ctk.CTkFrame) -> None:
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            header,
            text="L",
            width=44,
            height=44,
            corner_radius=13,
            fg_color=self.BLUE,
            text_color="white",
            font=ctk.CTkFont("Segoe UI", 24, "bold"),
        ).pack(anchor="w", padx=24, pady=(28, 12))
        ctk.CTkLabel(
            header,
            text="Loon Translator",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 21, "bold"),
        ).pack(anchor="w", padx=24)
        ctk.CTkLabel(
            header,
            text=self.t("tagline"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=24, pady=(2, 8))
        ctk.CTkButton(
            header,
            text=self.t("tutorial_button"),
            command=lambda: open_tutorial(self),
            height=32,
            corner_radius=8,
            fg_color="#EEF2F7",
            hover_color="#E2E8F0",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=24, pady=(0, 8))
        TipsCard(header, self, "settings").pack(fill=tk.X, padx=16, pady=(0, 8))

        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.PANEL, corner_radius=0)
        scroll.grid(row=1, column=0, sticky="nsew")

        self._field_label(scroll, self.t("ui_language")).pack(anchor="w", padx=12)
        self.native_var = tk.StringVar(
            value=self._language_label(self.settings.native_language)
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.native_var,
            values=self._language_choices(include_auto=False),
            command=self._on_ui_language,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 14))

        self._section_label(scroll, self.t("section_processing")).pack(anchor="w", padx=12)
        self.mode_var = tk.StringVar(
            value=choice_label(
                MODE_LABEL_KEYS,
                self.settings.translation_mode,
                self.settings.native_language,
                "Local",
            )
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.mode_var,
            values=choice_values(MODE_LABEL_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))

        self._field_label(scroll, self.t("stt_label")).pack(anchor="w", padx=12)
        stt_choice = self.settings.whisper_model if self.settings.whisper_model in STT_LABEL_KEYS else "auto"
        self.stt_var = tk.StringVar(
            value=choice_label(STT_LABEL_KEYS, stt_choice, self.settings.native_language, "auto")
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.stt_var,
            values=choice_values(STT_LABEL_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))

        self._field_label(scroll, self.t("context_label")).pack(anchor="w", padx=12)
        context = normalize_context(self.settings.conversation_context)
        self.context_var = tk.StringVar(
            value=choice_label(
                CONTEXT_LABEL_KEYS,
                context,
                self.settings.native_language,
                "casual",
            )
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.context_var,
            values=choice_values(CONTEXT_LABEL_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 4))
        ctk.CTkLabel(
            scroll,
            text=self.t("context_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))

        self._field_label(scroll, self.t("voice_label")).pack(anchor="w", padx=12)
        self.voice_var = tk.StringVar(
            value=choice_label(
                GENDER_LABEL_KEYS,
                self.settings.voice_gender,
                self.settings.native_language,
                "Feminina",
            )
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.voice_var,
            values=choice_values(GENDER_LABEL_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))

        self._field_label(scroll, self.t("voice_engine_label")).pack(anchor="w", padx=12)
        engine = self.settings.voice_engine if self.settings.voice_engine in VOICE_ENGINE_KEYS else "natural"
        self.voice_engine_var = tk.StringVar(
            value=choice_label(VOICE_ENGINE_KEYS, engine, self.settings.native_language, "natural")
        )
        ctk.CTkComboBox(
            scroll,
            variable=self.voice_engine_var,
            values=choice_values(VOICE_ENGINE_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 4))
        ctk.CTkLabel(
            scroll,
            text=self.t("voice_engine_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))

        self._field_label(scroll, self.t("pitch_label")).pack(anchor="w", padx=12)
        try:
            pitch = float(self.settings.voice_pitch)
        except (TypeError, ValueError):
            pitch = 0.0
        pitch = max(-6.0, min(6.0, pitch))
        self.pitch_var = tk.DoubleVar(value=pitch)
        self.pitch_caption = ctk.CTkLabel(
            scroll,
            text=self._pitch_caption(pitch),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        pitch_slider = ctk.CTkSlider(
            scroll,
            from_=-6,
            to=6,
            number_of_steps=24,
            variable=self.pitch_var,
            command=self._on_pitch,
            progress_color=self.BLUE,
            button_color=self.BLUE,
            button_hover_color=self.BLUE_HOVER,
            fg_color="#E8EEF8",
        )
        pitch_slider.pack(fill=tk.X, padx=12, pady=(8, 2))
        pitch_slider.bind("<ButtonRelease-1>", lambda _event: self._persist_preferences())
        self.pitch_caption.pack(anchor="w", padx=12, pady=(0, 12))

        self._field_label(scroll, self.t("rate_label")).pack(anchor="w", padx=12)
        try:
            rate = float(self.settings.speech_rate)
        except (TypeError, ValueError):
            rate = 1.0
        self.rate_var = tk.DoubleVar(value=max(0.8, min(1.5, rate)))
        self.rate_caption = ctk.CTkLabel(
            scroll,
            text=self._rate_caption(self.rate_var.get()),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        rate_slider = ctk.CTkSlider(
            scroll,
            from_=0.8,
            to=1.5,
            number_of_steps=14,
            variable=self.rate_var,
            command=self._on_rate,
            progress_color=self.BLUE,
            button_color=self.BLUE,
            button_hover_color=self.BLUE_HOVER,
            fg_color="#E8EEF8",
        )
        rate_slider.pack(fill=tk.X, padx=12, pady=(8, 2))
        rate_slider.bind("<ButtonRelease-1>", lambda _event: self._persist_preferences())
        self.rate_caption.pack(anchor="w", padx=12)
        ctk.CTkLabel(
            scroll,
            text=self.t("rate_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))

        self.save_history_var = tk.BooleanVar(value=self.settings.save_history)
        ctk.CTkCheckBox(
            scroll,
            text=self.t("save_history"),
            variable=self.save_history_var,
            command=self._persist_preferences,
            text_color=self.TEXT,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            border_color="#98A2B3",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=12, pady=5)
        self.monitor_var = tk.BooleanVar(value=self.settings.monitor_outgoing)
        ctk.CTkCheckBox(
            scroll,
            text=self.t("monitor_self"),
            variable=self.monitor_var,
            command=self._persist_preferences,
            text_color=self.TEXT,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            border_color="#98A2B3",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=12, pady=(5, 14))

        ctk.CTkButton(
            scroll,
            text=self.t("refresh_devices"),
            command=self.refresh_devices,
            height=38,
            corner_radius=9,
            fg_color="#EEF2F7",
            hover_color="#E2E8F0",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(fill=tk.X, padx=12, pady=4)
        ctk.CTkButton(
            scroll,
            text=self.t("language_packs"),
            command=self.open_language_packs,
            height=38,
            corner_radius=9,
            fg_color="#EEF2F7",
            hover_color="#E2E8F0",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(fill=tk.X, padx=12, pady=(4, 16))

        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 18))
        self.toggle_btn = ctk.CTkButton(
            footer,
            text=self.t("start"),
            command=self.toggle,
            height=48,
            corner_radius=11,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            text_color="white",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
        )
        self.toggle_btn.pack(fill=tk.X, pady=(0, 10))
        self.global_status_var = tk.StringVar(value=self.t("status_configure"))
        ctk.CTkLabel(
            footer,
            textvariable=self.global_status_var,
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=250,
            justify="left",
        ).pack(fill=tk.X)

    def _on_pitch(self, value: float) -> None:
        if hasattr(self, "pitch_caption"):
            self.pitch_caption.configure(text=self._pitch_caption(float(value)))
        self._persist_preferences()

    def _pitch_caption(self, value: float) -> str:
        rounded = round(float(value), 1)
        if rounded >= 0.5:
            tone = self.t("pitch_higher")
        elif rounded <= -0.5:
            tone = self.t("pitch_lower")
        else:
            tone = self.t("pitch_natural")
        return f"{rounded:+.1f}  {tone}"

    def _pitch_value(self) -> float:
        return round(float(self.pitch_var.get()), 1)

    def _on_rate(self, value: float) -> None:
        if hasattr(self, "rate_caption"):
            self.rate_caption.configure(text=self._rate_caption(float(value)))

    @staticmethod
    def _rate_caption(value: float) -> str:
        return f"{float(value):.2f}×"

    def _rate_value(self) -> float:
        return round(float(self.rate_var.get()), 2)

    def _stt_key(self) -> str:
        return canonical_choice(
            self.stt_var.get(),
            STT_LABEL_KEYS,
            self.settings.native_language,
            "auto",
        )

    def _on_ui_language(self, label: str) -> None:
        if self._building_ui:
            return
        canonical = canonical_language(label, self.settings.native_language)
        if canonical in {"", "Auto"} or canonical == self.settings.native_language:
            return
        if self.pipeline and self.pipeline.running:
            self._show_error(self.t("dialog_stop_first"))
            self._building_ui = True
            self.native_var.set(self._language_label(self.settings.native_language))
            self._building_ui = False
            return
        self._persist_preferences()
        self.settings.native_language = canonical
        self.settings_store.save(self.settings)
        self._build_main()
        self.refresh_devices()

    def _build_direction_panel(
        self,
        parent: ctk.CTkFrame,
        column: int,
        title: str,
        subtitle: str,
        prefix: str,
    ) -> None:
        frame = ctk.CTkFrame(
            parent,
            fg_color=self.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=self.BORDER,
        )
        frame.grid(
            row=1,
            column=column,
            sticky="nsew",
            padx=(0, 7) if column == 0 else (7, 0),
            pady=(0, 14),
        )
        ctk.CTkLabel(
            frame,
            text=title,
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 17, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 0))
        ctk.CTkLabel(
            frame,
            text=subtitle,
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
        ).pack(anchor="w", padx=20, pady=(0, 14))
        source_default = (
            self.settings.source_language
            if prefix == "out"
            else self.settings.incoming_source_language
        )
        target_default = (
            self.settings.outgoing_language if prefix == "out" else self.settings.incoming_language
        )
        source_var = tk.StringVar(value=self._language_label(source_default))
        target_var = tk.StringVar(value=self._language_label(target_default))
        setattr(self, f"{prefix}_source_var", source_var)
        setattr(self, f"{prefix}_target_var", target_var)

        self._field_label(frame, self.t("spoken_language")).pack(anchor="w", padx=20)
        ctk.CTkComboBox(
            frame,
            variable=source_var,
            values=self._language_choices(include_auto=True),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=20, pady=(5, 12))
        self._field_label(frame, self.t("translate_to")).pack(anchor="w", padx=20)
        ctk.CTkComboBox(
            frame,
            variable=target_var,
            values=self._language_choices(include_auto=False),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=20, pady=(5, 12))

        device_name = self.t("mic_device") if prefix == "out" else self.t("loopback_device")
        self._field_label(frame, device_name).pack(anchor="w", padx=20)
        device_var = tk.StringVar()
        setattr(self, f"{prefix}_input_var", device_var)
        combo = ctk.CTkComboBox(
            frame,
            variable=device_var,
            values=[self.t("devices_loading")],
            state="readonly",
            **self._combo_style(),
        )
        combo.pack(fill=tk.X, padx=20, pady=(5, 12))
        setattr(self, f"{prefix}_input_combo", combo)

        output_name = self.t("cable_device") if prefix == "out" else self.t("phones_device")
        self._field_label(frame, output_name).pack(anchor="w", padx=20)
        output_var = tk.StringVar()
        setattr(self, f"{prefix}_output_var", output_var)
        combo = ctk.CTkComboBox(
            frame,
            variable=output_var,
            values=[self.t("devices_loading")],
            state="readonly",
            **self._combo_style(),
        )
        combo.pack(fill=tk.X, padx=20, pady=(5, 12))
        setattr(self, f"{prefix}_output_combo", combo)

        status = tk.StringVar(value=self.t("status_waiting"))
        setattr(self, f"{prefix}_status_var", status)
        ctk.CTkLabel(
            frame,
            textvariable=status,
            text_color=self.GREEN,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=20, pady=(1, 16))

    def _build_history(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(
            parent,
            fg_color=self.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=self.BORDER,
        )
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        heading = ctk.CTkFrame(frame, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
        ctk.CTkLabel(
            heading,
            text=self.t("history_title"),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
        ).pack(side=tk.LEFT)
        ctk.CTkButton(
            heading,
            text=self.t("clear_data"),
            command=self.clear_private_data,
            width=120,
            height=30,
            corner_radius=8,
            fg_color="#F1F4F8",
            hover_color="#E2E8F0",
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(side=tk.RIGHT)
        self.text_output = ctk.CTkTextbox(
            frame,
            fg_color="#F8FAFC",
            text_color=self.TEXT,
            border_width=0,
            corner_radius=10,
            font=ctk.CTkFont("Consolas", 12),
            wrap="word",
        )
        self.text_output.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 18))
        self.text_output.configure(state=tk.DISABLED)

    def _field_label(self, parent: ctk.CTkBaseClass, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        )

    def _section_label(self, parent: ctk.CTkBaseClass, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            text_color="#98A2B3",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        )

    def _combo_style(self) -> dict:
        return {
            "height": 36,
            "corner_radius": 8,
            "border_color": self.BORDER,
            "fg_color": "#F8FAFC",
            "button_color": "#E8EEF8",
            "button_hover_color": "#DDE6F4",
            "text_color": self.TEXT,
            "dropdown_fg_color": "white",
            "dropdown_text_color": self.TEXT,
            "font": ctk.CTkFont("Segoe UI", 11),
        }

    def _language_label(self, canonical: str) -> str:
        return language_label(canonical, self.settings.native_language)

    def _language_choices(self, *, include_auto: bool) -> list[str]:
        return language_choices(self.settings.native_language, include_auto=include_auto)

    def _language_key(self, label: str) -> str:
        return canonical_language(label, self.settings.native_language)

    def _mode_key(self) -> str:
        return canonical_choice(
            self.mode_var.get(),
            MODE_LABEL_KEYS,
            self.settings.native_language,
            "Local",
        )

    def _gender_key(self) -> str:
        return canonical_choice(
            self.voice_var.get(),
            GENDER_LABEL_KEYS,
            self.settings.native_language,
            "Feminina",
        )

    def _voice_engine_key(self) -> str:
        if not hasattr(self, "voice_engine_var"):
            return self.settings.voice_engine or "natural"
        return canonical_choice(
            self.voice_engine_var.get(),
            VOICE_ENGINE_KEYS,
            self.settings.native_language,
            "natural",
        )

    def _context_key(self) -> str:
        return canonical_choice(
            self.context_var.get(),
            CONTEXT_LABEL_KEYS,
            self.settings.native_language,
            "casual",
        )

    def _format_status(self, status: str) -> str:
        if status.startswith("ready:"):
            seconds = f"{int(status.split(':', 1)[1]) / 1000:.1f}"
            return self._fill("status_ready", seconds=seconds)
        mapped = self.t(f"status_{status}")
        if mapped == f"status_{status}":
            return status
        return mapped

    def _format_provider(self, provider: str) -> str:
        return provider.replace("equivalência cultural", self.t("provider_cultural")).replace(
            "sem tradução", self.t("provider_same")
        )

    def refresh_devices(self) -> None:
        if self.pipeline and self.pipeline.running:
            messagebox.showinfo("Loon Translator", self.t("dialog_stop_first"))
            return
        try:
            microphones = self.devices.list_microphones()
            outputs = self.devices.list_outputs()
            loopbacks = self.devices.list_loopbacks()
        except Exception as exc:
            self._show_error(self._fill("error_devices", message=str(exc)))
            return
        cables = [item for item in outputs if self.devices.is_cable_input(item.name)]
        headphones = [item for item in outputs if not self.devices.is_virtual_output(item.name)]
        self._set_device_combo("out_input", microphones, self.settings.microphone_device, "microphone")
        self._set_device_combo(
            "out_output",
            cables,
            self.settings.virtual_output_device,
            "cable",
        )
        self._set_device_combo(
            "in_output", headphones, self.settings.monitor_output_device, "phones"
        )
        self._set_device_combo("in_input", loopbacks, self.settings.loopback_device, "loopback")
        cable = self.devices.find_virtual_cable()
        if any(
            not getattr(self, f"{key}_var").get()
            for key in ("out_input", "out_output", "in_input", "in_output")
        ):
            self.global_status_var.set(self.t("status_devices_missing"))
        elif not cable:
            self.global_status_var.set(self.t("status_no_cable"))
        elif not loopbacks:
            self.global_status_var.set(self.t("status_no_loopback"))
        else:
            self.global_status_var.set(self.t("status_devices_ready"))

    def _set_device_combo(
        self,
        key: str,
        devices: list[AudioDevice],
        preferred_uid: str = "",
        role: str = "",
    ) -> None:
        mapping = {f"{item.name} ({item.uid})": item for item in devices}
        self.device_maps[key] = mapping
        combo: ctk.CTkComboBox = getattr(self, f"{key}_combo")
        variable: tk.StringVar = getattr(self, f"{key}_var")
        labels = list(mapping)
        combo.configure(values=labels or [self.t("devices_none")])
        selected = self.devices.select_device(role, devices, preferred_uid) if role else None
        preferred = next(
            (label for label, device in mapping.items() if selected and device.uid == selected.uid),
            "",
        )
        if preferred:
            variable.set(preferred)
        elif preferred_uid and not labels:
            variable.set("")
            self.global_status_var.set(self.t("status_device_gone"))
        else:
            variable.set(labels[0] if labels else "")
        fields = {
            "out_input": "microphone_device",
            "out_output": "virtual_output_device",
            "in_output": "monitor_output_device",
            "in_input": "loopback_device",
        }
        if selected is not None and selected.uid != preferred_uid and key in fields:
            setattr(self.settings, fields[key], selected.uid)
            self.settings_store.save(self.settings)

    def toggle(self) -> None:
        if self._stopping:
            return
        if self.pipeline and self.pipeline.running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _required_offline_languages(self) -> list[str]:
        names = [
            self._language_key(self.out_source_var.get()),
            self._language_key(self.out_target_var.get()),
            self._language_key(self.in_source_var.get()),
            self._language_key(self.in_target_var.get()),
        ]
        if "Auto" in names and "Português" not in names:
            names.append("Português")
        return names

    def _ensure_local_packs(self) -> bool:
        try:
            missing = LanguagePackManager().missing(self._required_offline_languages())
        except Exception as exc:
            self._show_error(self._fill("error_packs_check", message=str(exc)))
            return False
        if not missing:
            return True
        listed = ", ".join(self._language_label(name) for name in missing)
        if not messagebox.askyesno(
            self.t("dialog_download_title"),
            self._fill("dialog_download_body", languages=listed),
        ):
            self.global_status_var.set(self.t("status_download_local"))
            return False
        self._download_required_packs(missing)
        return False

    def _download_required_packs(self, languages: list[str]) -> None:
        self.toggle_btn.configure(state=tk.DISABLED, text=self.t("pack_downloading"))
        self.global_status_var.set(self.t("status_downloading"))

        def worker() -> None:
            manager = LanguagePackManager()
            try:
                for name in languages:
                    manager.install(
                        name,
                        progress=lambda message: self._post_event(-1, "pack_status", message),
                    )
                self._post_event(-1, "packs_ready")
            except Exception as exc:
                self._post_event(-1, "pack_error_start", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _start_pipeline(self) -> None:
        self._generation += 1
        generation = self._generation
        if self.tests_view is not None:
            self.tests_view.shutdown()
        if (self._mode_key() != "Local" or self._voice_engine_key() == "natural") and not self.settings.cloud_consent:
            consent = messagebox.askyesno(
                self.t("dialog_cloud_title"),
                self.t("dialog_cloud_body"),
            )
            if not consent:
                self.global_status_var.set(self.t("status_cloud_denied"))
                return
            self.settings.cloud_consent = True
        if self._mode_key() == "IA" and not LLMConfig.from_env().configured:
            key = ctk.CTkInputDialog(title=self.t("dialog_llm_title"), text=self.t("dialog_llm_body")).get_input()
            if key and key.strip():
                save_user_env({"LLM_PROVIDER": "groq", "LLM_API_KEY": key.strip()})
        if self._mode_key() == "Local" and not self._ensure_local_packs():
            return
        selected = {
            key: self.device_maps.get(key, {}).get(getattr(self, f"{key}_var").get())
            for key in ("out_input", "out_output", "in_input", "in_output")
        }
        missing = [key for key, value in selected.items() if value is None]
        if missing:
            self._show_error(self.t("dialog_select_devices"))
            return
        if self._voice_engine_key() == "windows":
            unspoken = TTSEngine.missing_spoken_languages(
                [
                    self._language_key(self.out_target_var.get()),
                    self._language_key(self.in_target_var.get()),
                ]
            )
            if unspoken:
                messagebox.showwarning(
                    self.t("dialog_missing_voice_title"),
                    self._fill(
                        "dialog_missing_voice_body",
                        languages=", ".join(self._language_label(name) for name in unspoken),
                    ),
                )
        if selected["out_output"].kind != "virtual":
            if not messagebox.askyesno(
                self.t("dialog_virtual_title"),
                self.t("dialog_virtual_body"),
            ):
                return

        options = PipelineOptions(
            microphone=selected["out_input"],
            virtual_output=selected["out_output"],
            loopback=selected["in_input"],
            monitor_output=selected["in_output"],
            outgoing_source=self._language_key(self.out_source_var.get()),
            outgoing_target=self._language_key(self.out_target_var.get()),
            incoming_source=self._language_key(self.in_source_var.get()),
            incoming_target=self._language_key(self.in_target_var.get()),
            voice_gender=self._gender_key(),
            translation_mode=self._mode_key(),
            whisper_model=self._stt_key(),
            silence_ms=self.settings.phrase_silence_ms,
            max_phrase_seconds=self.settings.max_phrase_seconds,
            monitor_outgoing=self.monitor_var.get(),
            conversation_context=self._context_key(),
            voice_pitch=self._pitch_value(),
            speech_rate=self._rate_value(),
            voice_engine=self._voice_engine_key(),
        )
        self.pipeline = DuplexTranslationPipeline(
            options,
            on_record=lambda record: self._post_event(generation, "record", record),
            on_status=lambda direction, status: self._post_event(
                generation, "status", direction, status
            ),
            on_error=lambda error: self._post_event(generation, "error", error),
        )
        self._save_settings()
        try:
            self.pipeline.start()
        except Exception as exc:
            self.pipeline = None
            self._show_error(str(exc))
            return
        self.toggle_btn.configure(text=self.t("stop"), fg_color=self.RED)

    def _stop_pipeline(self) -> None:
        pipeline = self.pipeline
        if not pipeline or self._stopping:
            return
        self._stopping = True
        self.toggle_btn.configure(text=self.t("stopping"), fg_color="#64748B", state=tk.DISABLED)
        self.global_status_var.set(self.t("status_stopping"))

        def stop_worker() -> None:
            pipeline.stop()
            while not pipeline.terminated:
                time.sleep(0.2)
            self._post_event(self._generation, "stop_done", pipeline)

        threading.Thread(target=stop_worker, daemon=True).start()

    def _persist_preferences(self) -> None:
        if self._building_ui or not hasattr(self, "mode_var"):
            return
        self.settings.translation_mode = self._mode_key()
        self.settings.voice_gender = self._gender_key()
        self.settings.voice_engine = self._voice_engine_key()
        self.settings.save_history = bool(self.save_history_var.get())
        self.settings.monitor_outgoing = bool(self.monitor_var.get())
        self.settings.conversation_context = self._context_key()
        self.settings.voice_pitch = self._pitch_value()
        self.settings.speech_rate = self._rate_value()
        self.settings.whisper_model = self._stt_key()
        if hasattr(self, "out_source_var"):
            self.settings.source_language = self._language_key(self.out_source_var.get())
            self.settings.incoming_source_language = self._language_key(self.in_source_var.get())
            self.settings.outgoing_language = self._language_key(self.out_target_var.get())
            self.settings.incoming_language = self._language_key(self.in_target_var.get())
            fields = {
                "out_input": "microphone_device",
                "out_output": "virtual_output_device",
                "in_output": "monitor_output_device",
                "in_input": "loopback_device",
            }
            for key, field_name in fields.items():
                device = self.device_maps.get(key, {}).get(getattr(self, f"{key}_var").get())
                if device is not None:
                    setattr(self.settings, field_name, device.uid)
        self.settings.onboarding_complete = True
        self.settings_store.save(self.settings)

    def _save_settings(self) -> None:
        self._persist_preferences()

    def _on_record(self, record: TranslationRecord) -> None:
        direction = (
            self.t("direction_out")
            if record.direction == AudioDirection.OUTGOING
            else self.t("direction_in")
        )
        bilingual = ""
        if record.portuguese_text:
            bilingual += f"  PT: {record.portuguese_text}\n"
        if record.english_text:
            bilingual += f"  EN: {record.english_text}\n"
        timing = f"{record.latency_ms / 1000:.1f}s"
        if record.stt_ms or record.translation_ms:
            timing += (
                f"  (STT {record.stt_ms / 1000:.1f} · MT {record.translation_ms / 1000:.1f}"
                f" · TTS {record.speech_ms / 1000:.1f})"
            )
        summary = self.pipeline.latency.summary() if self.pipeline else {}
        if summary.get("count", 0) >= 3:
            timing += f"  p50 {summary['p50'] / 1000:.1f}s · p95 {summary['p95'] / 1000:.1f}s"
        message = (
            f"[{direction}] {record.source_text}\n"
            f"→ {record.translated_text}\n"
            f"{bilingual}"
            f"  {self._format_provider(record.provider)} • {timing}\n"
            f"{'─' * 72}\n"
        )
        self.text_output.configure(state=tk.NORMAL)
        self.text_output.insert(tk.END, message)
        self.text_output.see(tk.END)
        self.text_output.configure(state=tk.DISABLED)
        if self.save_history_var.get():
            try:
                self.history_store.add(record)
            except Exception as exc:
                logger.warning("Não foi possível salvar histórico: %s", exc)

    def _on_status(self, direction: AudioDirection | None, status: str) -> None:
        label = self._format_status(status)
        if direction == AudioDirection.OUTGOING:
            self.out_status_var.set(label)
        elif direction == AudioDirection.INCOMING:
            self.in_status_var.set(label)
        else:
            self.global_status_var.set(label)
            if status == "listening_stopped":
                self._sync_mode_button()
                if self.pipeline and self.pipeline.terminated and not self._stopping:
                    self.pipeline = None

    def _show_error(self, message: str) -> None:
        if message.startswith("missing_voice:"):
            message = self._fill(
                "error_missing_voice",
                language=self._language_label(message.split(":", 1)[1]),
            )
        if any(token in message for token in ("PaErrorCode", "WdmSyncIoctl", "Unanticipated host error")):
            logger.error("Falha de áudio: %s", message)
            message = self.t("error_playback")
        if hasattr(self, "global_status_var"):
            self.global_status_var.set(f"{self.t('error_prefix')} {message}")
        if message in self._shown_errors:
            return
        self._shown_errors.add(message)
        messagebox.showerror("Loon Translator", message)

    def open_language_packs(self) -> None:
        window = ctk.CTkToplevel(self.root)
        window.title(self.t("language_packs"))
        window.geometry("620x650")
        window.transient(self.root)
        window.lift()
        ctk.CTkLabel(
            window,
            text=self.t("packs_title"),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
        ).pack(anchor="w", padx=24, pady=(22, 4))
        ctk.CTkLabel(
            window,
            text=self.t("packs_intro"),
            text_color=self.MUTED,
            wraplength=560,
            justify="left",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=24, pady=(0, 16))
        listing = ctk.CTkScrollableFrame(window, fg_color="#F8FAFC", corner_radius=12)
        listing.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))
        listing.grid_columnconfigure(0, weight=1)
        try:
            statuses = LanguagePackManager().statuses()
        except Exception as exc:
            ctk.CTkLabel(
                listing,
                text=str(exc),
                text_color=self.RED,
                wraplength=500,
            ).grid(row=0, column=0, padx=16, pady=20)
            return
        for row, pack in enumerate(statuses):
            ctk.CTkLabel(
                listing,
                text=self._language_label(pack.name),
                text_color=self.TEXT,
                font=ctk.CTkFont("Segoe UI", 13, "bold"),
            ).grid(row=row, column=0, sticky="w", padx=14, pady=10)
            status_label = ctk.CTkLabel(
                listing,
                text=self.t("pack_installed") if pack.installed else self.t("pack_missing"),
                text_color=self.GREEN if pack.installed else self.MUTED,
                font=ctk.CTkFont("Segoe UI", 11),
            )
            status_label.grid(row=row, column=1, padx=10)
            button = ctk.CTkButton(
                listing,
                text=self.t("pack_installed") if pack.installed else self.t("pack_download"),
                width=110,
                height=30,
                state=tk.DISABLED if pack.installed else tk.NORMAL,
                fg_color="#DDE4EE" if pack.installed else self.BLUE,
                hover_color=self.BLUE_HOVER,
            )
            button.configure(
                command=lambda name=pack.name, btn=button, label=status_label: (
                    self._install_language_pack(name, btn, label)
                )
            )
            button.grid(row=row, column=2, padx=14, pady=8)

    def _install_language_pack(
        self,
        language_name: str,
        button: ctk.CTkButton,
        status_label: ctk.CTkLabel,
    ) -> None:
        button.configure(state=tk.DISABLED, text=self.t("pack_downloading"))
        manager = LanguagePackManager()

        def install() -> None:
            try:
                manager.install(
                    language_name,
                    progress=lambda message: self._post_event(
                        -1, "pack_progress", status_label, message
                    ),
                )
                self._post_event(-1, "pack_done", button, status_label, language_name)
            except Exception as exc:
                self._post_event(-1, "pack_error", button, status_label, str(exc))

        threading.Thread(target=install, daemon=True).start()

    def clear_private_data(self) -> None:
        if not messagebox.askyesno(self.t("dialog_clear_title"), self.t("dialog_clear_body")):
            return
        try:
            self.history_store.clear()
            TTSEngine.prune_cache(max_files=0)
            self.text_output.configure(state=tk.NORMAL)
            self.text_output.delete("1.0", tk.END)
            self.text_output.configure(state=tk.DISABLED)
            self.global_status_var.set(self.t("status_cleared"))
        except Exception as exc:
            self._show_error(self._fill("error_clear", message=str(exc)))

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.tests_view is not None:
            self.tests_view.shutdown()
        pipeline = self.pipeline
        if not pipeline or not hasattr(self, "toggle_btn"):
            self.root.destroy()
            return
        self.toggle_btn.configure(state=tk.DISABLED, text=self.t("closing"))

        def shutdown() -> None:
            pipeline.stop(wait_timeout=3)
            self._post_event(self._generation, "close_done")

        threading.Thread(target=shutdown, daemon=True).start()

    def _post_event(self, generation: int, event_type: str, *payload) -> None:
        self._events.put((generation, event_type, payload))

    def _drain_events(self) -> None:
        try:
            while True:
                generation, event_type, payload = self._events.get_nowait()
                if generation not in {-1, self._generation}:
                    continue
                if event_type == "record" and hasattr(self, "text_output"):
                    self._on_record(payload[0])
                elif event_type == "status" and hasattr(self, "global_status_var"):
                    self._on_status(payload[0], payload[1])
                elif event_type == "error":
                    self._show_error(payload[0])
                elif event_type == "stop_done" and hasattr(self, "toggle_btn"):
                    if self.pipeline is payload[0]:
                        self.pipeline = None
                    self._stopping = False
                    self._sync_mode_button()
                    self.global_status_var.set(self.t("status_listening_stopped"))
                elif event_type == "close_done":
                    self.root.destroy()
                    return
                elif event_type == "pack_progress":
                    payload[0].configure(text=payload[1], text_color=self.MUTED)
                elif event_type == "pack_done":
                    payload[0].configure(
                        text=self.t("pack_installed"),
                        state=tk.DISABLED,
                        fg_color="#DDE4EE",
                    )
                    payload[1].configure(
                        text=self._fill(
                            "pack_ready_for",
                            language=self._language_label(payload[2]),
                        ),
                        text_color=self.GREEN,
                    )
                elif event_type == "pack_error":
                    payload[0].configure(text=self.t("pack_retry"), state=tk.NORMAL)
                    payload[1].configure(text=self.t("pack_failed"), text_color=self.RED)
                    self._show_error(payload[2])
                elif event_type == "pack_status" and hasattr(self, "global_status_var"):
                    self.global_status_var.set(payload[0])
                elif event_type == "packs_ready" and hasattr(self, "toggle_btn"):
                    self.toggle_btn.configure(
                        text=self.t("start"), fg_color=self.BLUE, state=tk.NORMAL
                    )
                    self.global_status_var.set(self.t("status_languages_ready"))
                    self._start_pipeline()
                elif event_type == "pack_error_start" and hasattr(self, "toggle_btn"):
                    self.toggle_btn.configure(
                        text=self.t("start"), fg_color=self.BLUE, state=tk.NORMAL
                    )
                    self._show_error(self._fill("error_pack_download", message=payload[0]))
                elif event_type == "mic_level" and self.tests_view is not None:
                    self.tests_view.show_level(payload[0])
                elif event_type == "train_level" and self.training_view is not None:
                    self.training_view.show_level(payload[0])
                elif event_type == "train_status" and self.training_view is not None:
                    self.training_view.show_status(payload[0])
                elif event_type == "train_pair" and self.training_view is not None:
                    self.training_view.show_pair(payload[0], payload[1])
                elif event_type == "train_idle" and self.training_view is not None:
                    self.training_view.finish()
                elif event_type == "test_status" and self.tests_view is not None:
                    self.tests_view.show_status(payload[0])
                elif event_type == "mic_idle" and self.tests_view is not None:
                    self.tests_view.show_idle()
                elif event_type == "test_idle" and self.tests_view is not None:
                    self.tests_view.finish_busy()
        except queue.Empty:
            pass
        try:
            self.root.after(50, self._drain_events)
        except tk.TclError:
            return
