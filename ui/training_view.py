from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk
import numpy as np

from core.whisper_engine import WhisperEngine
from domain.entities import VoiceSnapshot
from domain.practice import (
    PracticePhrase,
    PracticeScore,
    choose_practice_phrase,
    language_code,
    score_repetition,
)
from services.translation_engine import LANGUAGES_DICT, TranslationEngine
from services.tts_engine import AudioPlayer, TTSEngine
from ui.tests_view import TestsView
from ui.tips import TipsCard
from ui.waveform import WaveformMeter

_WORD_COLORS = {
    "correct": "#059669",
    "missing": "#DC2626",
    "extra": "#D97706",
}


class TrainingView(ctk.CTkFrame):
    """A pessoa fala e ouve a própria voz de volta, e pode repetir frases."""

    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._busy = False
        self._translator = TranslationEngine()
        self._tts = TTSEngine()
        self._used_phrases: set[str] = set()
        self._phrase: PracticePhrase | None = None
        self.last_score: PracticeScore | None = None
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
        self._language_box(
            languages,
            1,
            app.t("train_target"),
            self.target_var,
            command=self._on_target_change,
        )
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
            text_color=app.INK,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        )
        self.speak_button.pack(anchor="w", padx=18, pady=(8, 16))
        self._build_repeat_card(scroll)
        self._phrase = self._pick_phrase()
        self._show_phrase()

    def _build_repeat_card(self, parent) -> None:
        app = self.app
        card = ctk.CTkFrame(
            parent,
            fg_color=app.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=app.BORDER,
        )
        card.pack(fill=tk.X, pady=(0, 8))
        self.practice_card = card
        ctk.CTkLabel(
            card,
            text=app.t("train_repeat_title"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            card,
            text=app.t("train_repeat_body"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))
        phrase_box = ctk.CTkFrame(
            card,
            fg_color=app.FIELD,
            corner_radius=12,
            border_width=1,
            border_color=app.BORDER,
        )
        phrase_box.pack(fill=tk.X, padx=18, pady=(0, 8))
        ctk.CTkLabel(
            phrase_box,
            text=app.t("train_repeat_prompt"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        self.practice_phrase = ctk.CTkLabel(
            phrase_box,
            text=app.t("train_repeat_idle"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 18),
            wraplength=680,
            justify="left",
        )
        self.practice_phrase.pack(anchor="w", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            phrase_box,
            text=app.t("train_meaning"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(4, 0))
        self.practice_meaning = ctk.CTkLabel(
            phrase_box,
            text="",
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 13),
            wraplength=680,
            justify="left",
        )
        self.practice_meaning.pack(anchor="w", padx=12, pady=(0, 12))
        legend = ctk.CTkFrame(card, fg_color="transparent")
        legend.pack(fill=tk.X, padx=18, pady=(0, 6))
        for status, key in (
            ("correct", "train_word_correct"),
            ("missing", "train_word_missing"),
            ("extra", "train_word_extra"),
        ):
            ctk.CTkLabel(
                legend,
                text=app.t(key),
                text_color=_WORD_COLORS[status],
                font=ctk.CTkFont("Segoe UI", 11, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 14))
        self.practice_score_var = tk.StringVar(value="")
        ctk.CTkLabel(
            card,
            textvariable=self.practice_score_var,
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=18, pady=(0, 4))
        self.practice_attempt = ctk.CTkLabel(
            card,
            text="",
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=720,
            justify="left",
        )
        self.practice_attempt.pack(anchor="w", padx=18, pady=(0, 6))
        self.practice_words = ctk.CTkFrame(card, fg_color="transparent")
        self.practice_words.pack(fill=tk.X, padx=18, pady=(0, 10))
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill=tk.X, padx=18, pady=(0, 16))
        self.repeat_button = ctk.CTkButton(
            actions,
            text=app.t("train_repeat_start"),
            command=self.start_repeat,
            height=36,
            corner_radius=9,
            fg_color=app.BLUE,
            hover_color=app.BLUE_HOVER,
            text_color=app.INK,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self.repeat_button.pack(side=tk.LEFT, padx=(0, 8))
        self.listen_button = ctk.CTkButton(
            actions,
            text=app.t("train_listen_again"),
            command=self.listen_again,
            height=36,
            corner_radius=9,
            fg_color=app.GHOST,
            hover_color="#1c3845",
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self.listen_button.pack(side=tk.LEFT, padx=(0, 8))
        self.next_button = ctk.CTkButton(
            actions,
            text=app.t("train_next_phrase"),
            command=self.next_phrase,
            height=36,
            corner_radius=9,
            fg_color=app.GHOST,
            hover_color="#1c3845",
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self.next_button.pack(side=tk.LEFT)
        self._practice_buttons = (self.repeat_button, self.listen_button, self.next_button)

    def show_level(self, level: float) -> None:
        self.wave.push(level)

    def show_status(self, text: str) -> None:
        self.status_var.set(text)

    def show_pair(self, spoken: str, translated: str) -> None:
        self.heard.configure(text=spoken or self.app.t("train_empty"))
        self.translated.configure(text=translated or self.app.t("train_empty"))

    def show_score(self, spoken: str, score: PracticeScore) -> None:
        self.last_score = score
        self.practice_score_var.set(self.app._fill("train_score", percent=str(score.percent)))
        heard = spoken.strip() or self.app.t("train_repeat_empty")
        self.practice_attempt.configure(text=f"{self.app.t('train_your_attempt')}: {heard}")
        for child in self.practice_words.winfo_children():
            child.destroy()
        for mark in score.words:
            prefix = {"missing": "−", "extra": "+"}.get(mark.status, "")
            ctk.CTkLabel(
                self.practice_words,
                text=f"{prefix}{mark.text}",
                text_color=_WORD_COLORS.get(mark.status, self.app.TEXT),
                font=ctk.CTkFont("Segoe UI", 13, "bold"),
            ).pack(side=tk.LEFT, padx=(0, 8), pady=2)

    def _language_box(self, parent, column: int, caption: str, variable: tk.StringVar, command=None) -> None:
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
            command=command,
            **app._combo_style(),
        ).grid(row=1, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0), pady=(4, 0))

    def _phrase_card(self, parent, column: int, caption: str, placeholder: str) -> ctk.CTkLabel:
        app = self.app
        frame = ctk.CTkFrame(
            parent,
            fg_color=app.FIELD,
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
        self._set_controls(True)
        self.wave.clear()

    def shutdown(self) -> None:
        return

    def _set_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        if self.speak_button.winfo_exists():
            self.speak_button.configure(state=state)
        for button in self._practice_buttons:
            if button.winfo_exists():
                button.configure(state=state)

    def _devices(self) -> tuple[str, str] | None:
        microphone = self.app.device_maps.get("out_input", {}).get(
            getattr(self.app, "out_input_var").get()
        )
        phones = self.app.device_maps.get("in_output", {}).get(
            getattr(self.app, "in_output_var").get()
        )
        if microphone is None or phones is None:
            self.app._show_error(self.app.t("test_need_mic"))
            return None
        return microphone.uid, phones.uid

    def _begin_job(self) -> bool:
        if self._busy or (self.app.pipeline and self.app.pipeline.running):
            self.app._show_error(self.app.t("test_busy"))
            return False
        self._busy = True
        self._set_controls(False)
        return True

    def practice(self) -> None:
        if self._busy or (self.app.pipeline and self.app.pipeline.running):
            self.app._show_error(self.app.t("test_busy"))
            return
        devices = self._devices()
        if devices is None or not self._begin_job():
            return
        self.heard.configure(text=self.app.t("train_listening"))
        self.translated.configure(text=self.app.t("train_translation_idle"))
        source = self.app._language_key(self.language_var.get())
        target = self.app._language_key(self.target_var.get())
        snapshot = self.app.voice_snapshot()
        threading.Thread(
            target=self._worker,
            args=(*devices, source, target, snapshot),
            daemon=True,
        ).start()

    def start_repeat(self) -> None:
        devices = self._devices()
        if devices is None or not self._begin_job():
            return
        phrase = self._ensure_phrase()
        snapshot = self.app.voice_snapshot()
        threading.Thread(
            target=self._repeat_worker,
            args=(*devices, phrase, snapshot, True),
            daemon=True,
        ).start()

    def listen_again(self) -> None:
        devices = self._devices()
        if devices is None or not self._begin_job():
            return
        phrase = self._ensure_phrase()
        snapshot = self.app.voice_snapshot()
        threading.Thread(
            target=self._repeat_worker,
            args=(*devices, phrase, snapshot, False),
            daemon=True,
        ).start()

    def next_phrase(self) -> None:
        if self._busy:
            return
        self._phrase = self._pick_phrase()
        self._show_phrase()
        self._clear_score()

    def _on_target_change(self, _label: str) -> None:
        if self._busy or not hasattr(self, "practice_phrase"):
            return
        self._used_phrases.clear()
        self._phrase = self._pick_phrase()
        self._show_phrase()
        self._clear_score()

    def _ensure_phrase(self) -> PracticePhrase:
        target = self.app._language_key(self.target_var.get())
        if self._phrase is None or language_code(self._phrase.language) != language_code(target):
            self._phrase = self._pick_phrase()
            self._show_phrase()
        return self._phrase

    def _pick_phrase(self) -> PracticePhrase:
        target = self.app._language_key(self.target_var.get())
        try:
            records = self.app.history_store.recent(200)
        except Exception:
            records = []
        phrase = choose_practice_phrase(target, records, self._used_phrases)
        self._used_phrases.add(phrase.text.casefold())
        return phrase

    def _show_phrase(self) -> None:
        if self._phrase is None:
            self.practice_phrase.configure(text=self.app.t("train_repeat_idle"))
            self.practice_meaning.configure(text="")
            return
        self.practice_phrase.configure(text=self._phrase.text)
        self.practice_meaning.configure(text=self._phrase.meaning)

    def _clear_score(self) -> None:
        self.last_score = None
        self.practice_score_var.set("")
        self.practice_attempt.configure(text="")
        for child in self.practice_words.winfo_children():
            child.destroy()

    def _play_phrase(self, phrase: PracticePhrase, phones_uid: str, snapshot: VoiceSnapshot) -> None:
        voice = self._tts.synthesize(
            phrase.text,
            phrase.language,
            snapshot.gender,
            snapshot.mode,
            snapshot.rate,
            pitch_semitones=snapshot.pitch,
            voice_engine=snapshot.engine,
        )
        AudioPlayer.play(
            voice,
            phones_uid,
            pitch_semitones=0.0 if snapshot.engine != "windows" else snapshot.pitch,
        )

    def _worker(
        self,
        microphone_uid: str,
        phones_uid: str,
        source_name: str,
        target_name: str,
        snapshot: VoiceSnapshot,
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
            stt = WhisperEngine(snapshot.stt)
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
                snapshot.mode,
                context=self.app.settings.conversation_context,
            )
            self.app._post_event(-1, "train_pair", spoken, translated.text)
            self.app._post_event(-1, "train_status", self.app.t("train_replay"))
            AudioPlayer.play_samples(audio, rate, phones_uid)
            self.app._post_event(-1, "train_status", self.app.t("train_translated_voice"))
            voice = self._tts.synthesize(
                translated.text,
                target_name,
                snapshot.gender,
                snapshot.mode,
                snapshot.rate,
                pitch_semitones=snapshot.pitch,
                voice_engine=snapshot.engine,
            )
            AudioPlayer.play(
                voice,
                phones_uid,
                pitch_semitones=0.0 if snapshot.engine != "windows" else snapshot.pitch,
            )
            self.app._post_event(-1, "train_status", self.app.t("train_done"))
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "train_idle")

    def _repeat_worker(
        self,
        microphone_uid: str,
        phones_uid: str,
        phrase: PracticePhrase,
        snapshot: VoiceSnapshot,
        record: bool,
    ) -> None:
        try:
            self.app._post_event(-1, "train_status", self.app.t("train_repeat_listen"))
            self._play_phrase(phrase, phones_uid, snapshot)
            if not record:
                self.app._post_event(-1, "train_status", self.app.t("train_done"))
                return
            self.app._post_event(-1, "train_status", self.app.t("train_repeat_record"))
            audio, rate = TestsView._capture(
                microphone_uid,
                4.0,
                on_level=lambda level: self.app._post_event(-1, "train_level", level),
            )
            self.app._post_event(-1, "train_status", self.app.t("status_transcribing"))
            code = language_code(phrase.language)
            if code == "auto":
                code = "en"
            mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
            if rate != 16_000 and mono.size > 1:
                size = max(1, round(mono.size * 16_000 / rate))
                mono = np.interp(
                    np.linspace(0.0, 1.0, size, endpoint=False),
                    np.linspace(0.0, 1.0, mono.size, endpoint=False),
                    mono,
                ).astype(np.float32)
            spoken = WhisperEngine(snapshot.stt).transcribe(mono.astype(np.float32), code).text
            score = score_repetition(phrase.text, spoken)
            self.app._post_event(-1, "train_score", spoken, score)
            self.app._post_event(-1, "train_status", self.app.t("train_done"))
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "train_idle")
