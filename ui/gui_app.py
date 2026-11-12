from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.config import SettingsStore, save_user_env
from core.pipeline import DuplexTranslationPipeline, PipelineOptions
from domain.entities import AudioDirection, LiveCaption, TranslationRecord, VoiceSnapshot
from domain.history import format_history_entry, parse_history_bound, suggest_conversation_name
from infrastructure.history_store import HistoryStore
from services.audio_devices import AudioDevice, AudioDeviceManager
from services.cultural_equivalence import normalize_context
from services.language_packs import LanguagePackManager
from services.licensing import licensee_name
from services.platform_host import current_host
from services.llm_translator import LLMConfig, PRESETS, PROVIDER_LABELS, PROVIDER_SITES, activate_llm
from services.piper_engine import PiperEngine
from services.translation_engine import LANGUAGES_DICT
from services.tts_engine import AudioPlayer, TTSEngine
from services.virtual_cable import VirtualCable
from services.win_com import com_apartment
from ui.i18n import (
    APPEARANCE_KEYS,
    CONTEXT_LABEL_KEYS,
    GENDER_LABEL_KEYS,
    LLM_PROVIDER_KEYS,
    MODE_LABEL_KEYS,
    OFFLINE_ASR_KEYS,
    OFFLINE_MT_KEYS,
    OFFLINE_TTS_KEYS,
    PROFILE_LABEL_KEYS,
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
from ui.theme import apply_theme, canonical_theme, uses_light_chrome
from ui.tips import open_tutorial
from ui.training_view import TrainingView
from ui.welcome import WelcomeScreen
from ui.win_chrome import bind_opaque

logger = logging.getLogger(__name__)


class LoonApp:
    BG = "#080d14"
    PANEL = "#0d1620"
    FIELD = "#101a25"
    GHOST = "#111c26"
    TEXT = "#eaf1f8"
    MUTED = "#7f8e9e"
    BORDER = "#1c2b38"
    BLUE = "#39d7e4"
    BLUE_HOVER = "#6de4f1"
    INK = "#031216"
    GREEN = "#45e5ba"
    PAUSE = "#442029"
    PAUSE_TEXT = "#ffd7d9"
    RED = "#ff8b95"
    SIDEBAR = "#090e16"
    NAV_ACTIVE = "#102b36"
    NAV_BORDER = "#1c3845"
    HOVER = "#111a25"
    CHIP = "#0d201e"

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title(f"Loon Translator — {current_host().label}")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)
        self._center_window()
        self.root.deiconify()
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        if self.settings.follow_pc_ideal:
            from core.machine import apply_pc_ideal

            apply_pc_ideal(self.settings)
            self.settings_store.save(self.settings)
        apply_theme(self, self.settings.appearance)
        ctk.set_appearance_mode("light" if uses_light_chrome(self.settings.appearance) else "dark")
        self.root.configure(fg_color=self.BG)
        self.devices = AudioDeviceManager()
        self._cable = VirtualCable()
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
        self._ui_ready = False
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._show_welcome()
        # A tela inteira é pesada. A boas-vindas aparece primeiro; o resto monta no primeiro ciclo.
        self.root.after(1, self._finish_startup)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self._report_callback_exception
        threading.excepthook = self._thread_excepthook
        self.root.after(50, self._drain_events)
        bind_opaque(self.root)

    @staticmethod
    def _report_callback_exception(exc, value, tb) -> None:
        logger.error("Exceção na interface", exc_info=(exc, value, tb))

    @staticmethod
    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.error(
            "Exceção na thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _center_window(self) -> None:
        width, height = 1280, 820
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _finish_startup(self) -> None:
        if self._closing or getattr(self, "_startup_done", False):
            return
        if self._building_ui:
            self.root.after(40, self._finish_startup)
            return
        if self.shell is None:
            self._ensure_shell()
        self._startup_done = True
        self._refresh_devices_quietly()
        threading.Thread(target=self._open_virtual_cable, name="loon-cable", daemon=True).start()

    def _ensure_shell(self) -> None:
        if self.shell is not None:
            return
        if not hasattr(self, "history_store"):
            self.history_store = HistoryStore()
        self._build_main()
        if self.welcome is not None:
            self.welcome.lift()

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
        if self.shell is None:
            if self._building_ui:
                self._transitioning = False
                self.root.after(40, lambda: self.begin_from_welcome(section, language_name))
                return
            self._ensure_shell()
        previous_language = self.settings.native_language
        first_run = not self.settings.onboarding_complete
        self.settings.native_language = language_name
        self.settings.last_section = (
            section if section in {"translate", "tests", "train", "history"} else "translate"
        )
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
                self._ui_ready = True
                self.settings.last_section = self._section
                self.settings_store.save(self.settings)
                return
            progress = (index + 1) / steps
            eased = progress * progress * (3 - 2 * progress)
            self.welcome.place(relx=0, rely=-eased, relwidth=1, relheight=1)
            self.root.after(16, lambda: tick(index + 1))

        tick()

    def _show_section(self, section: str) -> None:
        self._switch_view(section if section in {"translate", "tests", "train", "history"} else "translate")

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
            settings_window = getattr(self, "_settings_window", None)
            if settings_window is not None:
                try:
                    settings_window.destroy()
                except tk.TclError:
                    pass
                self._settings_window = None
            self._init_preference_vars()
            self._history_count = 0
            self.history_search_var = tk.StringVar()
            self.history_from_var = tk.StringVar()
            self.history_to_var = tk.StringVar()
            self.history_conversation_var = tk.StringVar(value=self.t("history_all"))
            self.conversation_var = tk.StringVar(value=self.t("conversation_idle"))
            self._conversation_name = ""
            self._history_records: list[TranslationRecord] = []
            self.latency_var = tk.StringVar(value=self.t("latency"))
            self.shell = ctk.CTkFrame(self.root, fg_color=self.BG, corner_radius=0)
            self.shell.grid(row=0, column=0, sticky="nsew")
            self.shell.grid_columnconfigure(1, weight=1)
            self.shell.grid_rowconfigure(0, weight=1)

            sidebar = ctk.CTkFrame(
                self.shell,
                width=244,
                corner_radius=0,
                fg_color=self.SIDEBAR,
                border_width=0,
            )
            sidebar.grid(row=0, column=0, sticky="nsew")
            sidebar.grid_propagate(False)
            self._build_sidebar(sidebar)

            content = ctk.CTkFrame(self.shell, fg_color="transparent")
            content.grid(row=0, column=1, sticky="nsew")
            content.grid_columnconfigure(0, weight=1)
            content.grid_rowconfigure(2, weight=1)

            topbar = ctk.CTkFrame(content, fg_color=self.BG, height=61, corner_radius=0)
            topbar.grid(row=0, column=0, sticky="ew")
            topbar.grid_propagate(False)
            topbar.grid_rowconfigure(0, weight=1)
            topbar.grid_columnconfigure(0, weight=1)
            self.breadcrumb = ctk.CTkLabel(
                topbar,
                text="LoonTranslator",
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 13),
            )
            self.breadcrumb.grid(row=0, column=0, sticky="w", padx=28)
            self.global_status_var = tk.StringVar(value=self.t("status_configure"))
            ctk.CTkLabel(
                topbar,
                textvariable=self.global_status_var,
                text_color=self.GREEN,
                font=ctk.CTkFont("Segoe UI", 12),
            ).grid(row=0, column=1, sticky="e", padx=(0, 10))
            ctk.CTkButton(
                topbar,
                text=self.t("settings_title"),
                width=118,
                height=31,
                corner_radius=8,
                fg_color="transparent",
                hover_color=self.HOVER,
                border_width=1,
                border_color=self.BORDER,
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 16),
                command=self.open_settings,
            ).grid(row=0, column=2, sticky="e", padx=(0, 22))

            heading = ctk.CTkFrame(content, fg_color="transparent")
            heading.grid(row=1, column=0, sticky="ew", padx=28, pady=(18, 8))
            heading.grid_columnconfigure(0, weight=1)
            titles = ctk.CTkFrame(heading, fg_color="transparent")
            titles.grid(row=0, column=0, sticky="w")
            self.header_title = ctk.CTkLabel(
                titles,
                text=self.t("nav_live"),
                text_color=self.TEXT,
                font=ctk.CTkFont("Segoe UI", 26, "bold"),
            )
            self.header_title.pack(anchor="w")
            self.header_subtitle = ctk.CTkLabel(
                titles,
                text=self.t("header_subtitle"),
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 13),
                wraplength=640,
                justify="left",
            )
            self.header_subtitle.pack(anchor="w", pady=(3, 0))
            self.pc_notice = ctk.CTkLabel(
                titles,
                text="",
                text_color=self.TEXT,
                font=ctk.CTkFont("Segoe UI", 12),
                wraplength=640,
                justify="left",
            )
            self.pc_notice.pack(anchor="w", pady=(8, 0))
            ctk.CTkLabel(
                heading,
                textvariable=self.latency_var,
                text_color=self.GREEN,
                fg_color=self.CHIP,
                corner_radius=10,
                font=ctk.CTkFont("Segoe UI", 12, "bold"),
                width=150,
                height=28,
            ).grid(row=0, column=1, sticky="e")

            pages = ctk.CTkFrame(content, fg_color="transparent")
            pages.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 18))
            pages.grid_columnconfigure(0, weight=1)
            pages.grid_rowconfigure(0, weight=1)

            self.translate_page = ctk.CTkFrame(pages, fg_color="transparent")
            self.translate_page.grid(row=0, column=0, sticky="nsew")
            self.translate_page.grid_columnconfigure((0, 1), weight=1, uniform="direction")
            self.translate_page.grid_rowconfigure(0, weight=1)
            self._build_direction_panel(
                self.translate_page, 0, self.t("out_title"), self.t("out_subtitle"), "out"
            )
            self._build_direction_panel(
                self.translate_page, 1, self.t("in_title"), self.t("in_subtitle"), "in"
            )
            self._build_control_hub(self.translate_page)

            self.history_page = ctk.CTkFrame(pages, fg_color="transparent")
            self.history_page.grid(row=0, column=0, sticky="nsew")
            self._build_history(self.history_page)
            self.history_page.grid_remove()

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
            self._refresh_pc_notice()
        finally:
            self._building_ui = False
        if self._ui_ready:
            self._show_section(self._section)

    def _switch_view(self, value: str) -> None:
        if self._building_ui or not hasattr(self, "tests_page"):
            return
        if value in {"tests", "train", "history", "translate"}:
            self._section = value
        elif value == self.t("tab_tests"):
            self._section = "tests"
        elif value == self.t("tab_train"):
            self._section = "train"
        else:
            self._section = "translate"
        self.translate_page.grid_remove()
        self.history_page.grid_remove()
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
        elif self._section == "history":
            self.history_page.grid()
        else:
            self.translate_page.grid()
        self._apply_header()
        self._paint_nav()
        self._sync_mode_button()
        if not self._transitioning:
            self.settings.last_section = self._section
            self.settings_store.save(self.settings)

    def _apply_header(self) -> None:
        if not hasattr(self, "header_title"):
            return
        titles = {
            "tests": ("tests_title", "tests_subtitle"),
            "train": ("train_title", "train_subtitle"),
            "history": ("history_title", "history_page_subtitle"),
        }
        title_key, subtitle_key = titles.get(self._section, ("nav_live", "header_subtitle"))
        self.header_title.configure(text=self.t(title_key))
        self.header_subtitle.configure(text=self.t(subtitle_key))
        if hasattr(self, "breadcrumb"):
            self.breadcrumb.configure(text=f"LoonTranslator   ›   {self.t(title_key)}")

    def _sync_mode_button(self) -> None:
        if not hasattr(self, "toggle_btn") or self._stopping:
            return
        running = bool(self.pipeline and self.pipeline.running)
        if running:
            self.toggle_btn.configure(
                text=self.t("stop"),
                fg_color=self.PAUSE,
                hover_color="#5a2a34",
                text_color=self.PAUSE_TEXT,
                command=self.toggle,
                state=tk.NORMAL,
            )
        else:
            self.toggle_btn.configure(
                text=self.t("start"),
                fg_color=self.BLUE,
                hover_color=self.BLUE_HOVER,
                text_color=self.INK,
                command=self.toggle,
                state=tk.NORMAL,
            )
        if hasattr(self, "session_state_var"):
            self.session_state_var.set(self.t("session_live") if running else self.t("session_ready"))
            if not running:
                self.session_time_var.set("00:00")

    def _open_translation(self) -> None:
        self._switch_view("translate")

    def _build_sidebar(self, parent: ctk.CTkFrame) -> None:
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        brand = ctk.CTkFrame(parent, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 8))
        mark = ctk.CTkFrame(brand, fg_color="transparent")
        mark.pack(anchor="w")
        ctk.CTkLabel(
            mark,
            text="L",
            width=40,
            height=40,
            corner_radius=11,
            fg_color=self.BLUE,
            text_color=self.INK,
            font=ctk.CTkFont("Segoe UI", 20, "bold"),
        ).pack(side=tk.LEFT)
        names = ctk.CTkFrame(mark, fg_color="transparent")
        names.pack(side=tk.LEFT, padx=(10, 0))
        ctk.CTkLabel(
            names,
            text="Loon",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            names,
            text="Translator",
            text_color=self.BLUE,
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w")
        owner = licensee_name()
        if owner:
            ctk.CTkLabel(
                names,
                text=self._fill("licensed_to", name=owner),
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 11),
                wraplength=150,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
        ctk.CTkLabel(
            parent,
            text=self.t("workspace").upper(),
            text_color="#5d6b7a",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=22, pady=(12, 6))
        nav = ctk.CTkFrame(parent, fg_color="transparent")
        nav.grid(row=2, column=0, sticky="new", padx=12)
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label_key in (
            ("translate", "nav_live"),
            ("history", "history_title"),
            ("train", "tab_train"),
            ("tests", "tab_tests"),
        ):
            button = ctk.CTkButton(
                nav,
                text=self.t(label_key),
                anchor="w",
                height=40,
                corner_radius=8,
                border_width=0,
                fg_color="transparent",
                hover_color=self.HOVER,
                text_color=self.MUTED,
                font=ctk.CTkFont("Segoe UI", 13, "bold"),
                command=lambda section=key: self._switch_view(section),
            )
            button.pack(fill=tk.X, pady=2)
            self._nav_buttons[key] = button
        ctk.CTkButton(
            nav,
            text=self.t("tutorial_button"),
            anchor="w",
            height=40,
            corner_radius=8,
            border_width=1,
            border_color=self.BORDER,
            fg_color=self.FIELD,
            hover_color=self.HOVER,
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            command=lambda: open_tutorial(self),
        ).pack(fill=tk.X, pady=(12, 2))
        session = ctk.CTkFrame(
            parent,
            fg_color=self.FIELD,
            corner_radius=12,
            border_width=1,
            border_color=self.BORDER,
        )
        session.grid(row=3, column=0, sticky="sew", padx=14, pady=(8, 18))
        ctk.CTkLabel(
            session,
            textvariable=self.session_state_var,
            text_color=self.GREEN,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            session,
            textvariable=self.session_time_var,
            text_color=self.TEXT,
            font=ctk.CTkFont("Consolas", 16),
        ).pack(anchor="w", padx=12, pady=(0, 10))
        self._clock_generation = getattr(self, "_clock_generation", 0) + 1
        generation = self._clock_generation
        self.root.after(1000, lambda gen=generation: self._tick_session(gen))
        self._paint_nav()

    def _tick_session(self, generation: int) -> None:
        if generation != getattr(self, "_clock_generation", None):
            return
        started = getattr(self, "_session_started", None)
        if self.pipeline and self.pipeline.running and started:
            elapsed = max(0, int(time.time() - started))
            self.session_time_var.set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
            self.session_state_var.set(self.t("session_live"))
        try:
            self.root.after(1000, lambda gen=generation: self._tick_session(gen))
        except tk.TclError:
            return

    def _paint_nav(self) -> None:
        labels = {
            "translate": "nav_live",
            "history": "history_title",
            "train": "tab_train",
            "tests": "tab_tests",
        }
        history_count = getattr(self, "_history_count", 0)
        for key, button in getattr(self, "_nav_buttons", {}).items():
            active = key == self._section
            label = self.t(labels[key])
            if key == "history" and history_count:
                label = f"{label}   {history_count}"
            button.configure(
                text=label,
                fg_color=self.NAV_ACTIVE if active else "transparent",
                text_color=self.BLUE if active else self.MUTED,
                border_width=1 if active else 0,
                border_color=self.NAV_BORDER,
            )

    def _init_preference_vars(self) -> None:
        self.native_var = tk.StringVar(value=self._language_label(self.settings.native_language))
        self.mode_var = tk.StringVar(
            value=choice_label(
                MODE_LABEL_KEYS,
                self.settings.translation_mode,
                self.settings.native_language,
                "Local",
            )
        )
        profile = self.settings.pipeline_profile if self.settings.pipeline_profile in PROFILE_LABEL_KEYS else ""
        self.profile_var = tk.StringVar(
            value=choice_label(PROFILE_LABEL_KEYS, profile, self.settings.native_language, "")
        )
        asr = self.settings.offline_asr if self.settings.offline_asr in OFFLINE_ASR_KEYS else ""
        mt = self.settings.offline_mt if self.settings.offline_mt in OFFLINE_MT_KEYS else ""
        voice = self.settings.offline_tts if self.settings.offline_tts in OFFLINE_TTS_KEYS else ""
        self.offline_asr_var = tk.StringVar(
            value=choice_label(OFFLINE_ASR_KEYS, asr, self.settings.native_language, "")
        )
        self.offline_mt_var = tk.StringVar(
            value=choice_label(OFFLINE_MT_KEYS, mt, self.settings.native_language, "")
        )
        self.offline_tts_var = tk.StringVar(
            value=choice_label(OFFLINE_TTS_KEYS, voice, self.settings.native_language, "")
        )
        stt_choice = self.settings.whisper_model if self.settings.whisper_model in STT_LABEL_KEYS else "auto"
        self.stt_var = tk.StringVar(
            value=choice_label(STT_LABEL_KEYS, stt_choice, self.settings.native_language, "auto")
        )
        context = normalize_context(self.settings.conversation_context)
        self.context_var = tk.StringVar(
            value=choice_label(CONTEXT_LABEL_KEYS, context, self.settings.native_language, "casual")
        )
        self.voice_var = tk.StringVar(
            value=choice_label(
                GENDER_LABEL_KEYS,
                self.settings.voice_gender,
                self.settings.native_language,
                "Feminina",
            )
        )
        engine = self.settings.voice_engine if self.settings.voice_engine in VOICE_ENGINE_KEYS else "piper"
        self.voice_engine_var = tk.StringVar(
            value=choice_label(VOICE_ENGINE_KEYS, engine, self.settings.native_language, "piper")
        )
        try:
            pitch = float(self.settings.voice_pitch)
        except (TypeError, ValueError):
            pitch = 0.0
        self.pitch_var = tk.DoubleVar(value=max(-6.0, min(6.0, pitch)))
        try:
            rate = float(self.settings.speech_rate)
        except (TypeError, ValueError):
            rate = 1.0
        self.rate_var = tk.DoubleVar(value=max(0.8, min(1.5, rate)))
        self.save_history_var = tk.BooleanVar(value=self.settings.save_history)
        self.monitor_var = tk.BooleanVar(value=self.settings.monitor_outgoing)
        self.session_state_var = tk.StringVar(value=self.t("session_ready"))
        self.session_time_var = tk.StringVar(value="00:00")
        self._session_started = getattr(self, "_session_started", None)
        look = canonical_theme(self.settings.appearance)
        self.appearance_var = tk.StringVar(
            value=choice_label(APPEARANCE_KEYS, look, self.settings.native_language, "night")
        )
        provider = self.settings.llm_provider if self.settings.llm_provider in LLM_PROVIDER_KEYS else "groq"
        self.llm_var = tk.StringVar(
            value=choice_label(LLM_PROVIDER_KEYS, provider, self.settings.native_language, "groq")
        )

    def open_settings(self) -> None:
        existing = getattr(self, "_settings_window", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus()
                    return
            except tk.TclError:
                pass
        window = ctk.CTkToplevel(self.root)
        self._settings_window = window
        window.title(self.t("settings_title"))
        window.geometry("440x720")
        window.configure(fg_color=self.BG)
        window.transient(self.root)
        window.lift()
        bind_opaque(window)
        ctk.CTkLabel(
            window,
            text=self.t("settings_title"),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 8))
        scroll = ctk.CTkScrollableFrame(window, fg_color=self.BG, corner_radius=0)
        scroll.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 12))
        self._populate_settings(scroll)

    def _ghost_button(self, parent, text: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=38,
            corner_radius=9,
            fg_color=self.GHOST,
            hover_color="#1c3845",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )

    def _populate_settings(self, scroll: ctk.CTkScrollableFrame) -> None:
        self.settings_pc_notice = ctk.CTkLabel(
            scroll,
            text="",
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=340,
            justify="left",
        )
        self.settings_pc_notice.pack(anchor="w", padx=12, pady=(4, 8))
        self._ghost_button(scroll, self.t("pc_use_ideal"), self._use_pc_ideal).pack(
            fill=tk.X, padx=12, pady=(0, 12)
        )
        self._refresh_pc_notice()
        self._field_label(scroll, self.t("appearance_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.appearance_var,
            values=choice_values(APPEARANCE_KEYS, self.settings.native_language),
            command=self._on_appearance_label,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))
        self._field_label(scroll, self.t("ui_language")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.native_var,
            values=self._language_choices(include_auto=False),
            command=self._on_ui_language,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 14))
        self._section_label(scroll, self.t("section_processing")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.mode_var,
            values=choice_values(MODE_LABEL_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))
        self._field_label(scroll, self.t("profile_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.profile_var,
            values=choice_values(PROFILE_LABEL_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 4))
        ctk.CTkLabel(
            scroll,
            text=self.t("profile_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        self._section_label(scroll, self.t("offline_title")).pack(anchor="w", padx=12)
        self._field_label(scroll, self.t("offline_asr_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.offline_asr_var,
            values=choice_values(OFFLINE_ASR_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 8))
        self._field_label(scroll, self.t("offline_mt_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.offline_mt_var,
            values=choice_values(OFFLINE_MT_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 8))
        self._field_label(scroll, self.t("offline_tts_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.offline_tts_var,
            values=choice_values(OFFLINE_TTS_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 4))
        ctk.CTkLabel(
            scroll,
            text=self.t("offline_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        self._field_label(scroll, self.t("llm_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.llm_var,
            values=choice_values(LLM_PROVIDER_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 8))
        self._ghost_button(scroll, self.t("llm_key_button"), self._ask_llm_key).pack(
            fill=tk.X, padx=12, pady=(0, 4)
        )
        ctk.CTkLabel(
            scroll,
            text=self.t("llm_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        self._field_label(scroll, self.t("stt_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.stt_var,
            values=choice_values(STT_LABEL_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))
        self._field_label(scroll, self.t("context_label")).pack(anchor="w", padx=12)
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
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        self._field_label(scroll, self.t("voice_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.voice_var,
            values=choice_values(GENDER_LABEL_KEYS, self.settings.native_language),
            command=lambda _value: self._persist_preferences(),
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 12))
        self._field_label(scroll, self.t("voice_engine_label")).pack(anchor="w", padx=12)
        ctk.CTkComboBox(
            scroll,
            variable=self.voice_engine_var,
            values=choice_values(VOICE_ENGINE_KEYS, self.settings.native_language),
            command=self._on_quality_choice,
            state="readonly",
            **self._combo_style(),
        ).pack(fill=tk.X, padx=12, pady=(7, 4))
        ctk.CTkLabel(
            scroll,
            text=self.t("voice_engine_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        self._field_label(scroll, self.t("pitch_label")).pack(anchor="w", padx=12)
        self.pitch_caption = ctk.CTkLabel(
            scroll,
            text=self._pitch_caption(self.pitch_var.get()),
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
            fg_color=self.FIELD,
        )
        pitch_slider.pack(fill=tk.X, padx=12, pady=(8, 2))
        pitch_slider.bind("<ButtonRelease-1>", lambda _event: self._persist_preferences())
        self.pitch_caption.pack(anchor="w", padx=12, pady=(0, 12))
        self._field_label(scroll, self.t("rate_label")).pack(anchor="w", padx=12)
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
            fg_color=self.FIELD,
        )
        rate_slider.pack(fill=tk.X, padx=12, pady=(8, 2))
        rate_slider.bind("<ButtonRelease-1>", lambda _event: self._persist_preferences())
        self.rate_caption.pack(anchor="w", padx=12)
        ctk.CTkLabel(
            scroll,
            text=self.t("rate_hint"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))
        ctk.CTkCheckBox(
            scroll,
            text=self.t("save_history"),
            variable=self.save_history_var,
            command=self._persist_preferences,
            text_color=self.TEXT,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            border_color=self.BORDER,
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=12, pady=5)
        ctk.CTkCheckBox(
            scroll,
            text=self.t("monitor_self"),
            variable=self.monitor_var,
            command=self._persist_preferences,
            text_color=self.TEXT,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            border_color=self.BORDER,
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(anchor="w", padx=12, pady=(5, 14))
        self._ghost_button(scroll, self.t("refresh_devices"), self.refresh_devices).pack(
            fill=tk.X, padx=12, pady=4
        )
        self._ghost_button(scroll, self.t("language_packs"), self.open_language_packs).pack(
            fill=tk.X, padx=12, pady=(4, 16)
        )

    def _build_control_hub(self, parent: ctk.CTkFrame) -> None:
        hub = ctk.CTkFrame(
            parent,
            fg_color=self.PANEL,
            corner_radius=12,
            border_width=1,
            border_color=self.BORDER,
        )
        hub.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.toggle_btn = ctk.CTkButton(
            hub,
            text=self.t("start"),
            command=self.toggle,
            height=46,
            corner_radius=11,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            text_color=self.INK,
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=12, pady=12)
        ctk.CTkLabel(
            hub,
            textvariable=self.conversation_var,
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=360,
            justify="left",
        ).pack(side=tk.LEFT, padx=(0, 8), pady=12)
        self._ghost_button(hub, self.t("copy_last"), self._copy_last_translation).pack(
            side=tk.LEFT, padx=(0, 8), pady=12
        )
        self._ghost_button(hub, self.t("listen_last"), self._speak_last_translation).pack(
            side=tk.LEFT, padx=(0, 8), pady=12
        )
        link = ctk.CTkButton(
            hub,
            text=self.t("open_history") + "  →",
            command=lambda: self._switch_view("history"),
            fg_color="transparent",
            hover_color="#111a25",
            text_color=self.BLUE,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        )
        link.pack(side=tk.RIGHT, padx=12, pady=12)

    def _copy_last_translation(self) -> None:
        text = getattr(self, "_last_translated", "").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.global_status_var.set(self.t("copied"))

    def _speak_last_translation(self) -> None:
        text = getattr(self, "_last_translated", "").strip()
        if not text or getattr(self, "_speaking", False):
            return
        self._speaking = True
        snapshot = self.voice_snapshot()
        language = self._language_key(self.in_target_var.get())
        device = self.device_maps.get("in_output", {}).get(self.in_output_var.get())
        device_uid = device.uid if device is not None else ""

        def work() -> None:
            try:
                audio_file = TTSEngine().synthesize(
                    text,
                    language,
                    snapshot.gender,
                    snapshot.mode,
                    snapshot.rate,
                    pitch_semitones=snapshot.pitch,
                    voice_engine=snapshot.engine,
                )
                with com_apartment():
                    AudioPlayer.play(audio_file, device_uid, pitch_semitones=snapshot.pitch)
            except Exception as exc:
                self._post_event(-1, "error", str(exc))
            finally:
                self._speaking = False

        threading.Thread(target=work, daemon=True, name="loon-listen").start()

    def _on_pitch(self, value: float) -> None:
        label = getattr(self, "pitch_caption", None)
        if label is not None:
            try:
                label.configure(text=self._pitch_caption(float(value)))
            except tk.TclError:
                pass
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
        label = getattr(self, "rate_caption", None)
        if label is not None:
            try:
                label.configure(text=self._rate_caption(float(value)))
            except tk.TclError:
                pass

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

    def _on_appearance_label(self, label: str) -> None:
        self._on_appearance(
            canonical_choice(label, APPEARANCE_KEYS, self.settings.native_language, "night")
        )

    def _on_appearance(self, name: str) -> None:
        if self._building_ui:
            return
        chosen = canonical_theme(name)
        if chosen == canonical_theme(self.settings.appearance):
            return
        if self.pipeline and self.pipeline.running:
            self._show_error(self.t("dialog_stop_first"))
            return
        apply_theme(self, chosen)
        ctk.set_appearance_mode("light" if uses_light_chrome(chosen) else "dark")
        self.settings.appearance = chosen
        self.settings_store.save(self.settings)
        self.root.configure(fg_color=self.BG)
        welcome = self.welcome is not None
        self._build_main()
        if welcome:
            self._show_welcome()
        self._refresh_devices_quietly()

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
            row=0,
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
        ).pack(anchor="w", padx=20, pady=(1, 2))
        caption = tk.StringVar(value="")
        setattr(self, f"{prefix}_caption_var", caption)
        ctk.CTkLabel(
            frame,
            textvariable=caption,
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, slant="italic"),
            wraplength=440,
            justify="left",
            anchor="w",
        ).pack(anchor="w", fill=tk.X, padx=20, pady=(0, 14))

    def _build_history(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(
            parent,
            fg_color=self.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=self.BORDER,
        )
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        heading = ctk.CTkFrame(frame, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
        heading.grid_columnconfigure(1, weight=1)
        self._field_label(heading, self.t("history_title")).grid(row=0, column=0, sticky="w")
        self.history_conversation = ctk.CTkComboBox(
            heading,
            variable=self.history_conversation_var,
            values=[self.t("history_all")],
            state="readonly",
            command=lambda _value: self._load_history_panel(),
            **self._combo_style(),
        )
        self.history_conversation.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        self._ghost_button(heading, self.t("tutorial_button"), lambda: open_tutorial(self)).grid(
            row=0, column=2, padx=(10, 0)
        )
        span = ctk.CTkFrame(heading, fg_color="transparent")
        span.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        span.grid_columnconfigure((1, 3), weight=1)
        self._field_label(span, self.t("history_from")).grid(row=0, column=0, padx=(0, 8))
        self.history_from = ctk.CTkEntry(
            span,
            textvariable=self.history_from_var,
            placeholder_text=self.t("history_when_hint"),
            height=36,
            corner_radius=8,
            border_color=self.BORDER,
            fg_color=self.FIELD,
            text_color=self.TEXT,
            placeholder_text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        self.history_from.grid(row=0, column=1, sticky="ew")
        self.history_from.bind("<Return>", lambda _event: self._load_history_panel())
        self.history_from.bind("<FocusOut>", lambda _event: self._load_history_panel())
        self._field_label(span, self.t("history_to")).grid(row=0, column=2, padx=(12, 8))
        self.history_to = ctk.CTkEntry(
            span,
            textvariable=self.history_to_var,
            placeholder_text=self.t("history_when_hint"),
            height=36,
            corner_radius=8,
            border_color=self.BORDER,
            fg_color=self.FIELD,
            text_color=self.TEXT,
            placeholder_text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        self.history_to.grid(row=0, column=3, sticky="ew")
        self.history_to.bind("<Return>", lambda _event: self._load_history_panel())
        self.history_to.bind("<FocusOut>", lambda _event: self._load_history_panel())
        self.history_search = ctk.CTkEntry(
            heading,
            textvariable=self.history_search_var,
            placeholder_text=self.t("history_search"),
            height=36,
            corner_radius=8,
            border_color=self.BORDER,
            fg_color=self.FIELD,
            text_color=self.TEXT,
            placeholder_text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
        )
        self.history_search.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.history_search_var.trace_add("write", lambda *_args: self._render_history())
        ctk.CTkButton(
            heading,
            text=self.t("clear_data"),
            command=self.clear_private_data,
            width=140,
            height=32,
            corner_radius=8,
            fg_color="#1c141b",
            hover_color="#2a1c24",
            border_width=1,
            border_color="#3a2a32",
            text_color="#b87980",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).grid(row=2, column=2, sticky="e", padx=(10, 0), pady=(8, 0))
        self.text_output = ctk.CTkTextbox(
            frame,
            fg_color=self.FIELD,
            text_color=self.TEXT,
            border_width=0,
            corner_radius=10,
            font=ctk.CTkFont("Consolas", 12),
            wrap="word",
        )
        self.text_output.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 18))
        self.text_output.configure(state=tk.DISABLED)
        self._load_history_panel()

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
            text_color="#5d6b7a",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        )

    def _combo_style(self) -> dict:
        return {
            "height": 36,
            "corner_radius": 8,
            "border_color": self.BORDER,
            "fg_color": self.FIELD,
            "button_color": "#1c2b38",
            "button_hover_color": "#255261",
            "text_color": self.TEXT,
            "dropdown_fg_color": self.PANEL,
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

    def _profile_key(self) -> str:
        if not hasattr(self, "profile_var"):
            profile = self.settings.pipeline_profile or ""
            return profile if profile in PROFILE_LABEL_KEYS else ""
        return canonical_choice(
            self.profile_var.get(),
            PROFILE_LABEL_KEYS,
            self.settings.native_language,
            "",
        )

    def _offline_key(self, var_name: str, mapping: dict[str, str], stored: str) -> str:
        var = getattr(self, var_name, None)
        if var is None:
            return stored if stored in mapping else ""
        return canonical_choice(var.get(), mapping, self.settings.native_language, "")

    def _llm_provider_key(self) -> str:
        if not hasattr(self, "llm_var"):
            provider = self.settings.llm_provider or "groq"
            return provider if provider in PRESETS else "groq"
        return canonical_choice(
            self.llm_var.get(),
            LLM_PROVIDER_KEYS,
            self.settings.native_language,
            "groq",
        )

    def _ask_llm_key(self) -> None:
        provider = self._llm_provider_key()
        self.settings.llm_provider = provider
        if provider in {"ollama", "lmstudio"}:
            activate_llm(provider, "")
            self.settings_store.save(self.settings)
            return
        name = PROVIDER_LABELS.get(provider, provider)
        site = PROVIDER_SITES.get(provider, "")
        key = ctk.CTkInputDialog(
            title=self.t("dialog_llm_title"),
            text=self._fill("dialog_llm_body", name=name, site=site),
        ).get_input()
        if key and key.strip():
            activate_llm(provider, key.strip())
            self.settings_store.save(self.settings)

    def _gender_key(self) -> str:
        return canonical_choice(
            self.voice_var.get(),
            GENDER_LABEL_KEYS,
            self.settings.native_language,
            "Feminina",
        )

    def _voice_engine_key(self) -> str:
        if not hasattr(self, "voice_engine_var"):
            return self.settings.voice_engine or "piper"
        return canonical_choice(
            self.voice_engine_var.get(),
            VOICE_ENGINE_KEYS,
            self.settings.native_language,
            "piper",
        )

    def voice_snapshot(self) -> VoiceSnapshot:
        return VoiceSnapshot(
            gender=self._gender_key(),
            rate=self._rate_value(),
            pitch=self._pitch_value(),
            engine=self._voice_engine_key(),
            mode=self._mode_key(),
            stt=self._stt_key(),
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

    def _open_virtual_cable(self) -> None:
        try:
            session = self._cable.open()
        except Exception:
            logger.exception("Falha ao criar o cabo virtual")
            return
        logger.info("Cabo virtual %s: %s", session.name, session.detail)
        if session.created and not self._closing:
            try:
                self.root.after(0, self._refresh_devices_quietly)
            except Exception:
                logger.debug("A janela fechou antes de listar o cabo", exc_info=True)

    def _refresh_devices_quietly(self) -> None:
        self.refresh_devices()

    def refresh_devices(self) -> None:
        if self._closing:
            return
        if self.pipeline and self.pipeline.running:
            messagebox.showinfo("Loon Translator", self.t("dialog_stop_first"))
            return
        if getattr(self, "_devices_loading", False):
            return
        self._devices_loading = True

        def scan() -> None:
            error: Exception | None = None
            microphones: list[AudioDevice] = []
            outputs: list[AudioDevice] = []
            loopbacks: list[AudioDevice] = []
            try:
                with com_apartment():
                    microphones = self.devices.list_microphones()
                    outputs = self.devices.list_outputs()
                    loopbacks = self.devices.list_loopbacks()
            except Exception as exc:
                error = exc

            def apply() -> None:
                self._devices_loading = False
                if self._closing:
                    return
                if error is not None:
                    logger.error("Falha ao listar os aparelhos: %s", error)
                    self._show_error(self._fill("error_devices", message=str(error)))
                    return
                self._apply_device_lists(microphones, outputs, loopbacks)

            try:
                self.root.after(0, apply)
            except tk.TclError:
                self._devices_loading = False

        threading.Thread(target=scan, name="loon-devices", daemon=True).start()

    def _apply_device_lists(
        self,
        microphones: list[AudioDevice],
        outputs: list[AudioDevice],
        loopbacks: list[AudioDevice],
    ) -> None:
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

    def _ask_conversation_name(self) -> str | None:
        try:
            existing = self.history_store.conversations()
        except Exception:
            existing = []
        suggested = suggest_conversation_name(existing, self.t("conversation_default"))
        chosen: dict[str, str | None] = {"name": None}
        window = ctk.CTkToplevel(self.root)
        bind_opaque(window)
        window.title(self.t("conversation_dialog_title"))
        window.geometry("460x230")
        window.resizable(False, False)
        window.configure(fg_color=self.BG)
        window.transient(self.root)
        window.lift()
        ctk.CTkLabel(
            window,
            text=self.t("conversation_dialog_title"),
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=22, pady=(18, 6))
        ctk.CTkLabel(
            window,
            text=self.t("conversation_dialog_body"),
            text_color=self.MUTED,
            font=ctk.CTkFont("Segoe UI", 13),
            wraplength=410,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 10))
        entry = ctk.CTkEntry(
            window,
            height=38,
            corner_radius=8,
            border_color=self.BORDER,
            fg_color=self.FIELD,
            text_color=self.TEXT,
            font=ctk.CTkFont("Segoe UI", 13),
        )
        entry.pack(fill=tk.X, padx=22)
        entry.insert(0, suggested)
        entry.select_range(0, tk.END)
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(fill=tk.X, padx=22, pady=16)

        def accept() -> None:
            chosen["name"] = entry.get().strip() or suggested
            window.destroy()

        def cancel() -> None:
            window.destroy()

        ctk.CTkButton(
            actions,
            text=self.t("conversation_ok"),
            command=accept,
            height=36,
            corner_radius=8,
            fg_color=self.BLUE,
            hover_color=self.BLUE_HOVER,
            text_color=self.INK,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT)
        self._ghost_button(actions, self.t("conversation_cancel"), cancel).pack(side=tk.LEFT, padx=(8, 0))
        window.protocol("WM_DELETE_WINDOW", cancel)
        entry.bind("<Return>", lambda _event: accept())
        window.grab_set()
        entry.focus()
        window.wait_window()
        return chosen["name"]

    def _start_pipeline(self) -> None:
        name = self._ask_conversation_name()
        if not name:
            return
        self._generation += 1
        generation = self._generation
        if self.tests_view is not None:
            self.tests_view.shutdown()
        cloud_profile = self._profile_key() in {"hybrid", "cloud", "s2s"}
        if (self._mode_key() != "Local" or self._voice_engine_key() in {"natural", "grok"} or cloud_profile) and not self.settings.cloud_consent:
            consent = messagebox.askyesno(
                self.t("dialog_cloud_title"),
                self.t("dialog_cloud_body"),
            )
            if not consent:
                self.global_status_var.set(self.t("status_cloud_denied"))
                return
            self.settings.cloud_consent = True
        if self._mode_key() == "IA":
            provider = self._llm_provider_key()
            self.settings.llm_provider = provider
            activate_llm(provider)
            if not LLMConfig.from_env().configured and provider not in {"ollama", "lmstudio"}:
                name = PROVIDER_LABELS.get(provider, provider)
                site = PROVIDER_SITES.get(provider, "")
                key = ctk.CTkInputDialog(
                    title=self.t("dialog_llm_title"),
                    text=self._fill("dialog_llm_body", name=name, site=site),
                ).get_input()
                if key and key.strip():
                    activate_llm(provider, key.strip())
        if self._voice_engine_key() == "grok" and not os.getenv("XAI_API_KEY", "").strip():
            key = ctk.CTkInputDialog(title=self.t("dialog_grok_title"), text=self.t("dialog_grok_body")).get_input()
            if key and key.strip():
                save_user_env({"XAI_API_KEY": key.strip()})
                os.environ["XAI_API_KEY"] = key.strip()
        if self._mode_key() == "Local" and self._profile_key() in {"", "local"} and not self._ensure_local_packs():
            return
        selected = {
            key: self.device_maps.get(key, {}).get(getattr(self, f"{key}_var").get())
            for key in ("out_input", "out_output", "in_input", "in_output")
        }
        missing = [key for key, value in selected.items() if value is None]
        if missing:
            self._show_error(self.t("dialog_select_devices"))
            return
        engine = self._voice_engine_key()
        if engine == "windows" or (engine == "piper" and self._mode_key() == "Local"):
            targets = [
                self._language_key(self.out_target_var.get()),
                self._language_key(self.in_target_var.get()),
            ]
            # No modo Local, o que o Piper não fala só sai pela voz do Windows.
            if engine == "piper":
                targets = [name for name in targets if not PiperEngine().available_for(name)]
            unspoken = TTSEngine.missing_spoken_languages(targets) if targets else []
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
            live_captions=self.settings.live_captions,
            pipeline_profile=self._profile_key(),
            offline_asr=self._offline_key("offline_asr_var", OFFLINE_ASR_KEYS, self.settings.offline_asr),
            offline_mt=self._offline_key("offline_mt_var", OFFLINE_MT_KEYS, self.settings.offline_mt),
            offline_tts=self._offline_key("offline_tts_var", OFFLINE_TTS_KEYS, self.settings.offline_tts),
        )
        self.pipeline = DuplexTranslationPipeline(
            options,
            on_record=lambda record: self._post_event(generation, "record", record),
            on_status=lambda direction, status: self._post_event(
                generation, "status", direction, status
            ),
            on_error=lambda error: self._post_event(generation, "error", error),
            on_partial=lambda caption: self._post_event(generation, "partial", caption),
        )
        self._save_settings()
        self._conversation_name = name
        try:
            self.pipeline.start()
        except Exception as exc:
            self.pipeline = None
            self._conversation_name = ""
            self._show_error(str(exc))
            return
        if hasattr(self, "conversation_var"):
            self.conversation_var.set(self._fill("conversation_live", name=name))
        self._session_started = time.time()
        self._sync_mode_button()

    def _stop_pipeline(self) -> None:
        pipeline = self.pipeline
        if not pipeline or self._stopping:
            return
        self._stopping = True
        self.toggle_btn.configure(
            text=self.t("stopping"),
            fg_color="#1c2b38",
            text_color=self.MUTED,
            state=tk.DISABLED,
        )
        self.global_status_var.set(self.t("status_stopping"))

        def stop_worker() -> None:
            pipeline.stop()
            while not pipeline.terminated:
                time.sleep(0.2)
            self._post_event(self._generation, "stop_done", pipeline)

        threading.Thread(target=stop_worker, daemon=True).start()

    def _on_quality_choice(self, _value: str | None = None) -> None:
        self._persist_preferences(from_quality=True)

    def _use_pc_ideal(self) -> None:
        from core.machine import apply_pc_ideal

        apply_pc_ideal(self.settings)
        lang = self.settings.native_language
        self.mode_var.set(choice_label(MODE_LABEL_KEYS, self.settings.translation_mode, lang, "Local"))
        self.profile_var.set(choice_label(PROFILE_LABEL_KEYS, self.settings.pipeline_profile or "", lang, ""))
        self.offline_asr_var.set(choice_label(OFFLINE_ASR_KEYS, self.settings.offline_asr, lang, ""))
        self.offline_mt_var.set(choice_label(OFFLINE_MT_KEYS, self.settings.offline_mt, lang, ""))
        self.offline_tts_var.set(choice_label(OFFLINE_TTS_KEYS, self.settings.offline_tts, lang, ""))
        self.stt_var.set(choice_label(STT_LABEL_KEYS, self.settings.whisper_model, lang, "auto"))
        self.voice_engine_var.set(choice_label(VOICE_ENGINE_KEYS, self.settings.voice_engine, lang, "piper"))
        self.settings_store.save(self.settings)
        self._refresh_pc_notice()

    def _refresh_pc_notice(self) -> None:
        from core.machine import deviation_keys

        notes = deviation_keys(self.settings)
        if notes:
            text = self.t("pc_custom_title") + "\n" + "\n".join(self.t(key) for key in notes)
        else:
            text = self.t("pc_ideal_title") + "\n" + self.t("pc_ideal_body")
        text += "\n" + self.t("tip_echo")
        if hasattr(self, "pc_notice"):
            self.pc_notice.configure(text=text)
        if hasattr(self, "settings_pc_notice"):
            self.settings_pc_notice.configure(text=text)

    def _persist_preferences(self, from_quality: bool = False) -> None:
        if self._building_ui or not hasattr(self, "mode_var"):
            return
        self.settings.translation_mode = self._mode_key()
        self.settings.pipeline_profile = self._profile_key()
        self.settings.offline_asr = self._offline_key("offline_asr_var", OFFLINE_ASR_KEYS, self.settings.offline_asr)
        self.settings.offline_mt = self._offline_key("offline_mt_var", OFFLINE_MT_KEYS, self.settings.offline_mt)
        self.settings.offline_tts = self._offline_key("offline_tts_var", OFFLINE_TTS_KEYS, self.settings.offline_tts)
        self.settings.voice_gender = self._gender_key()
        self.settings.voice_engine = self._voice_engine_key()
        self.settings.save_history = bool(self.save_history_var.get())
        self.settings.monitor_outgoing = bool(self.monitor_var.get())
        self.settings.conversation_context = self._context_key()
        self.settings.voice_pitch = self._pitch_value()
        self.settings.speech_rate = self._rate_value()
        self.settings.whisper_model = self._stt_key()
        provider = self._llm_provider_key()
        if provider != self.settings.llm_provider:
            self.settings.llm_provider = provider
            activate_llm(provider)
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
        if from_quality:
            from core.machine import matches_pc_ideal

            self.settings.follow_pc_ideal = matches_pc_ideal(self.settings)
        self.settings_store.save(self.settings)
        self._refresh_pc_notice()

    def _save_settings(self) -> None:
        self._persist_preferences()

    def _caption_var(self, direction: AudioDirection) -> tk.StringVar | None:
        prefix = "out" if direction == AudioDirection.OUTGOING else "in"
        return getattr(self, f"{prefix}_caption_var", None)

    def _on_partial(self, caption: LiveCaption) -> None:
        variable = self._caption_var(caption.direction)
        if variable is None:
            return
        text = f"{caption.heard}…" if caption.heard else ""
        if text and caption.translation:
            text += f"\n→ {caption.translation}"
        variable.set(text)

    def _clear_captions(self) -> None:
        for direction in AudioDirection:
            variable = self._caption_var(direction)
            if variable is not None:
                variable.set("")

    def _history_line(self, record: TranslationRecord, extra: str = "") -> str:
        direction = (
            self.t("direction_out")
            if record.direction == AudioDirection.OUTGOING
            else self.t("direction_in")
        )
        return format_history_entry(
            record,
            direction=direction,
            provider=self._format_provider(record.provider),
            extra=extra,
        )

    def _selected_conversation(self) -> str:
        selected = ""
        if hasattr(self, "history_conversation_var"):
            selected = self.history_conversation_var.get().strip()
        if not selected or selected == self.t("history_all"):
            return ""
        return selected

    def _refresh_conversation_choices(self, names: list[str]) -> None:
        if not hasattr(self, "history_conversation"):
            return
        current = self._selected_conversation()
        values = [self.t("history_all"), *names]
        self.history_conversation.configure(values=values)
        if current and current in names:
            self.history_conversation_var.set(current)
        elif self.history_conversation_var.get() not in values:
            self.history_conversation_var.set(self.t("history_all"))

    def _load_history_panel(self) -> None:
        if not hasattr(self, "text_output") or getattr(self, "_history_loading", False):
            return
        self._history_loading = True
        try:
            start, start_ok = parse_history_bound(
                self.history_from_var.get() if hasattr(self, "history_from_var") else "",
                end=False,
            )
            finish, finish_ok = parse_history_bound(
                self.history_to_var.get() if hasattr(self, "history_to_var") else "",
                end=True,
            )
            if not start_ok or not finish_ok:
                self._history_records = []
                self._history_notice = self.t("history_bad_range")
                self._render_history()
                return
            records: list[TranslationRecord] = []
            names: list[str] = []
            try:
                names = self.history_store.conversations()
                records = self.history_store.search(
                    conversation=self._selected_conversation(),
                    start=start,
                    end=finish,
                )
            except Exception as exc:
                logger.warning("Não foi possível ler o histórico: %s", exc)
            self._refresh_conversation_choices(names)
            self._history_records = records
            self._history_count = len(records)
            notice = ""
            if hasattr(self, "save_history_var") and not self.save_history_var.get():
                notice = self.t("history_not_saving")
            self._history_notice = notice
            self._render_history()
        finally:
            self._history_loading = False

    def _render_history(self) -> None:
        if not hasattr(self, "text_output"):
            return
        query = ""
        if hasattr(self, "history_search_var"):
            query = self.history_search_var.get().strip().casefold()
        lines: list[str] = []
        notice = getattr(self, "_history_notice", "")
        if notice:
            lines.append(notice + "\n\n")
        matched = 0
        for record in getattr(self, "_history_records", []):
            line = self._history_line(record)
            if query and query not in line.casefold():
                continue
            lines.append(line)
            matched += 1
        self.text_output.configure(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        if lines:
            self.text_output.insert(tk.END, "".join(lines))
        elif query or self._selected_conversation():
            self.text_output.insert(tk.END, self.t("history_empty"))
        self.text_output.see(tk.END)
        self.text_output.configure(state=tk.DISABLED)
        self._history_count = matched
        self._paint_nav()

    def _refresh_latency(self) -> None:
        summary = self.pipeline.latency.summary() if self.pipeline else {}
        if summary.get("count", 0) >= 1 and hasattr(self, "latency_var"):
            seconds = summary["p50"] / 1000
            self.latency_var.set(f"{self.t('latency')}  {seconds:.1f}s")
        elif hasattr(self, "latency_var"):
            self.latency_var.set(self.t("latency"))

    def _on_record(self, record: TranslationRecord) -> None:
        caption = self._caption_var(record.direction)
        if caption is not None:
            caption.set("")
        self._last_translated = record.translated_text
        if not record.conversation:
            record.conversation = getattr(self, "_conversation_name", "")
        self._refresh_latency()
        if self.save_history_var.get():
            try:
                self.history_store.add(record)
            except Exception as exc:
                logger.warning("Não foi possível salvar histórico: %s", exc)
        self._load_history_panel()

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
        bind_opaque(window)
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
        listing = ctk.CTkScrollableFrame(window, fg_color=self.FIELD, corner_radius=12)
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
                fg_color=self.GHOST if pack.installed else self.BLUE,
                text_color=self.MUTED if pack.installed else self.INK,
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
            self._load_history_panel()
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
            self._cable.close()
            self.root.destroy()
            return
        self.toggle_btn.configure(state=tk.DISABLED, text=self.t("closing"))

        def shutdown() -> None:
            pipeline.stop(wait_timeout=3)
            self._cable.close()
            self._post_event(self._generation, "close_done")

        threading.Thread(target=shutdown, daemon=True).start()

    def _post_event(self, generation: int, event_type: str, *payload) -> None:
        self._events.put((generation, event_type, payload))

    def _drain_events(self) -> None:
        try:
            while True:
                try:
                    generation, event_type, payload = self._events.get_nowait()
                except queue.Empty:
                    break
                if generation not in {-1, self._generation}:
                    continue
                try:
                    self._handle_event(event_type, payload)
                except Exception:
                    logger.exception("Falha ao processar evento da interface: %s", event_type)
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(50, self._drain_events)
            except tk.TclError:
                return

    def _handle_event(self, event_type: str, payload: tuple) -> None:
        if event_type == "record" and hasattr(self, "text_output"):
            self._on_record(payload[0])
        elif event_type == "status" and hasattr(self, "global_status_var"):
            self._on_status(payload[0], payload[1])
        elif event_type == "partial":
            self._on_partial(payload[0])
        elif event_type == "error":
            self._show_error(payload[0])
        elif event_type == "stop_done" and hasattr(self, "toggle_btn"):
            if self.pipeline is payload[0]:
                self.pipeline = None
            self._stopping = False
            self._session_started = None
            self._clear_captions()
            self._sync_mode_button()
            self.global_status_var.set(self.t("status_listening_stopped"))
        elif event_type == "close_done":
            self.root.destroy()
        elif event_type == "pack_progress":
            payload[0].configure(text=payload[1], text_color=self.MUTED)
        elif event_type == "pack_done":
            payload[0].configure(
                text=self.t("pack_installed"),
                state=tk.DISABLED,
                fg_color=self.GHOST,
                text_color=self.MUTED,
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
                text=self.t("start"),
                fg_color=self.BLUE,
                text_color=self.INK,
                state=tk.NORMAL,
            )
            self.global_status_var.set(self.t("status_languages_ready"))
            self._start_pipeline()
        elif event_type == "pack_error_start" and hasattr(self, "toggle_btn"):
            self.toggle_btn.configure(
                text=self.t("start"),
                fg_color=self.BLUE,
                text_color=self.INK,
                state=tk.NORMAL,
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
        elif event_type == "train_score" and self.training_view is not None:
            self.training_view.show_score(payload[0], payload[1])
        elif event_type == "train_idle" and self.training_view is not None:
            self.training_view.finish()
        elif event_type == "test_status" and self.tests_view is not None:
            self.tests_view.show_status(payload[0])
        elif event_type == "mic_idle" and self.tests_view is not None:
            self.tests_view.show_idle()
        elif event_type == "test_idle" and self.tests_view is not None:
            self.tests_view.finish_busy()
