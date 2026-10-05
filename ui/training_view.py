from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk
import numpy as np

from core.whisper_engine import WhisperEngine
from services.translation_engine import LANGUAGES_DICT, TranslationEngine
from services.tts_engine import AudioPlayer, TTSEngine
from ui.tests_view import TestsView
from ui.tips import TipsCard
from ui.waveform import WaveformMeter


class TrainingView(ctk.CTkFrame):
    """A pessoa fala e ouve a própria voz de volta, sem traduzir."""

    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._busy = False
        self._translator = TranslationEngine()
        self._tts = TTSEngine()
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill=tk.BOTH, expand=True)
        ctk.CTkLabel(
            scroll,
            text=app.t("train_title"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            scroll,
            text=app.t("train_subtitle"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 13),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))
        TipsCard(scroll, app, "train").pack(fill=tk.X, pady=(0, 12))
        card = ctk.CTkFrame(
            scroll,
            fg_color=app.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=app.BORDER,
        )
        card.pack(fill=tk.X, pady=(0, 8))
        languages = ctk.CTkFrame(card, fg_color="transparent")
        languages.pack(fill=tk.X, padx=18, pady=(16, 12))
        languages.grid_columnconfigure((0, 1), weight=1)
        self.language_var = tk.StringVar(value=app._language_label(app.settings.native_language))
        native = app.settings.native_language
        target_name = "Inglês" if native == "Português" else "Português"
        self.target_var = tk.StringVar(value=app._language_label(target_name))
        self._language_box(languages, 0, app.t("train_language"), self.language_var)
        self._language_box(languages, 1, app.t("train_target"), self.target_var)
        self.wave = WaveformMeter(card, color=app.BLUE)
        self.wave.pack(fill=tk.X, padx=18, pady=(0, 12))
        compare = ctk.CTkFrame(card, fg_color="transparent")
        compare.pack(fill=tk.X, padx=18, pady=(0, 8))
        compare.grid_columnconfigure((0, 1), weight=1)
        self.heard = self._phrase_card(compare, 0, app.t("train_you_said"), app.t("train_idle_text"))
        self.translated = self._phrase_card(
            compare, 1, app.t("train_translation"), app.t("train_translation_idle")
        )
        self.status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            card,
            textvariable=self.status_var,
            text_color=app.GREEN,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=18)
        self.speak_button = ctk.CTkButton(
            card,
            text=app.t("train_speak"),
            command=self.practice,
            height=40,
            corner_radius=10,
            fg_color=app.BLUE,
            hover_color=app.BLUE_HOVER,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        )
        self.speak_button.pack(anchor="w", padx=18, pady=(8, 16))

    def show_level(self, level: float) -> None:
        self.wave.push(level)

    def show_status(self, text: str) -> None:
        self.status_var.set(text)

    def show_pair(self, spoken: str, translated: str) -> None:
        self.heard.configure(text=spoken or self.app.t("train_empty"))
        self.translated.configure(text=translated or self.app.t("train_empty"))

    def _language_box(self, parent, column: int, caption: str, variable: tk.StringVar) -> None:
        app = self.app
        ctk.CTkLabel(
            parent,
            text=caption,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).grid(row=0, column=column, sticky="w", padx=(0, 8) if column == 0 else (8, 0))
        ctk.CTkComboBox(
            parent,
            variable=variable,
            values=app._language_choices(include_auto=False),
            state="readonly",
            **app._combo_style(),
        ).grid(row=1, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0), pady=(4, 0))

    def _phrase_card(self, parent, column: int, caption: str, placeholder: str) -> ctk.CTkLabel:
        app = self.app
        frame = ctk.CTkFrame(
            parent,
            fg_color="#F8FAFC",
            corner_radius=12,
            border_width=1,
            border_color=app.BORDER,
        )
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
        ctk.CTkLabel(
            frame,
            text=caption,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        body = ctk.CTkLabel(
            frame,
            text=placeholder,
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 15),
            wraplength=320,
            justify="left",
        )
        body.pack(anchor="w", padx=12, pady=(0, 12))
        return body

    def finish(self) -> None:
        self._busy = False
        self.speak_button.configure(state=tk.NORMAL)
        self.wave.clear()

    def shutdown(self) -> None:
        return

    def practice(self) -> None:
        if self._busy or (self.app.pipeline and self.app.pipeline.running):
            self.app._show_error(self.app.t("test_busy"))
            return
        microphone = self.app.device_maps.get("out_input", {}).get(
            getattr(self.app, "out_input_var").get()
        )
        phones = self.app.device_maps.get("in_output", {}).get(
            getattr(self.app, "in_output_var").get()
        )
        if microphone is None or phones is None:
            self.app._show_error(self.app.t("test_need_mic"))
            return
        self._busy = True
        self.speak_button.configure(state=tk.DISABLED)
        self.heard.configure(text=self.app.t("train_listening"))
        self.translated.configure(text=self.app.t("train_translation_idle"))
        source = self.app._language_key(self.language_var.get())
        target = self.app._language_key(self.target_var.get())
        threading.Thread(
            target=self._worker,
            args=(microphone.uid, phones.uid, source, target),
            daemon=True,
        ).start()

    def _worker(
        self,
        microphone_uid: str,
        phones_uid: str,
        source_name: str,
        target_name: str,
    ) -> None:
        try:
            self.app._post_event(-1, "train_status", self.app.t("test_recording"))
            audio, rate = TestsView._capture(
                microphone_uid,
                4.0,
                on_level=lambda level: self.app._post_event(-1, "train_level", level),
            )
            self.app._post_event(-1, "train_status", self.app.t("status_transcribing"))
            code = LANGUAGES_DICT.get(source_name, "pt").split("-")[0].lower()
            if code == "auto":
                code = "pt"
            mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
            if rate != 16_000 and mono.size > 1:
                size = max(1, round(mono.size * 16_000 / rate))
                mono = np.interp(
                    np.linspace(0.0, 1.0, size, endpoint=False),
                    np.linspace(0.0, 1.0, mono.size, endpoint=False),
                    mono,
                ).astype(np.float32)
            stt = WhisperEngine(self.app._stt_key())
            spoken = stt.transcribe(mono.astype(np.float32), code).text
            if not spoken:
                self.app._post_event(-1, "train_pair", "", "")
                self.app._post_event(-1, "train_status", self.app.t("train_empty"))
                return
            self.app._post_event(-1, "train_status", self.app.t("status_translating"))
            translated = self._translator.translate(
                spoken,
                code,
                target_name,
                self.app._mode_key(),
                context=self.app.settings.conversation_context,
            )
            self.app._post_event(-1, "train_pair", spoken, translated.text)
            self.app._post_event(-1, "train_status", self.app.t("train_replay"))
            AudioPlayer.play_samples(audio, rate, phones_uid)
            self.app._post_event(-1, "train_status", self.app.t("train_translated_voice"))
            voice = self._tts.synthesize(
                translated.text,
                target_name,
                self.app._gender_key(),
                self.app._mode_key(),
                self.app._rate_value(),
                pitch_semitones=self.app._pitch_value(),
                voice_engine=self.app._voice_engine_key(),
            )
            AudioPlayer.play(
                voice,
                phones_uid,
                pitch_semitones=0.0 if self.app._voice_engine_key() != "windows" else self.app._pitch_value(),
            )
            self.app._post_event(-1, "train_status", self.app.t("train_done"))
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "train_idle")
