from __future__ import annotations

import threading
import time
import tkinter as tk

import customtkinter as ctk
import numpy as np
import sounddevice as sd

from services.audio_devices import AudioDevice, AudioDeviceManager
from services.translation_engine import VOICE_MAP
from services.tts_engine import AudioPlayer, TTSEngine
from ui.tips import TipsCard
from ui.waveform import WaveformMeter


class TestsView(ctk.CTkFrame):
    """Ambiente isolado para microfone, fone, tom de voz e saída virtual."""

    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._stop_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._listening = False
        self._busy = False
        self._buttons: list[ctk.CTkButton] = []

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill=tk.BOTH, expand=True)
        scroll.grid_columnconfigure((0, 1), weight=1, uniform="tests")

        ctk.CTkLabel(
            scroll,
            text=app.t("tests_title"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ctk.CTkLabel(
            scroll,
            text=app.t("tests_subtitle"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 13),
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.summary = ctk.CTkLabel(
            scroll,
            text="",
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 12),
            justify="left",
            anchor="w",
            wraplength=760,
        )
        self.summary.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        TipsCard(scroll, app, "tests").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            scroll,
            textvariable=self.status_var,
            text_color=app.GREEN,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

        mic = self._card(scroll, 0, 5, app.t("mic_card_title"), app.t("mic_card_body"))
        ctk.CTkLabel(
            mic,
            text=app.t("test_level"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=18, pady=(0, 6))
        self.level = ctk.CTkProgressBar(
            mic,
            height=12,
            corner_radius=6,
            progress_color=app.BLUE,
            fg_color="#E8EEF8",
        )
        self.level.pack(fill=tk.X, padx=18, pady=(0, 8))
        self.level.set(0)
        self.wave = WaveformMeter(mic, color=app.BLUE)
        self.wave.pack(fill=tk.X, padx=18, pady=(0, 12))
        actions = ctk.CTkFrame(mic, fg_color="transparent")
        actions.pack(fill=tk.X, padx=18, pady=(0, 16))
        self.listen_button = self._button(actions, app.t("mic_listen"), self.toggle_listen)
        self.listen_button.pack(side=tk.LEFT, padx=(0, 8))
        self._button(actions, app.t("mic_record"), self.record_microphone).pack(side=tk.LEFT)

        phones = self._card(scroll, 1, 5, app.t("phones_card_title"), app.t("phones_card_body"))
        self._button(phones, app.t("phones_play"), self.play_headphones).pack(
            anchor="w", padx=18, pady=(0, 16)
        )

        voice = self._card(scroll, 0, 6, app.t("voice_card_title"), app.t("voice_card_body"))
        voice_actions = ctk.CTkFrame(voice, fg_color="transparent")
        voice_actions.pack(fill=tk.X, padx=18, pady=(0, 16))
        self._button(voice_actions, app.t("voice_preview"), self.preview_pitch).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self._button(voice_actions, app.t("voice_sample"), self.play_voice_sample).pack(side=tk.LEFT)

        cable = self._card(scroll, 1, 6, app.t("cable_card_title"), app.t("cable_card_body"))
        self._button(cable, app.t("cable_play"), self.play_virtual_output).pack(
            anchor="w", padx=18, pady=(0, 16)
        )
        self.refresh_summary()

    def _card(self, parent, column: int, row: int, title: str, body: str) -> ctk.CTkFrame:
        app = self.app
        frame = ctk.CTkFrame(
            parent,
            fg_color=app.PANEL,
            corner_radius=15,
            border_width=1,
            border_color=app.BORDER,
        )
        frame.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(
            frame,
            text=title,
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            frame,
            text=body,
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))
        return frame

    def _button(self, parent, text: str, command) -> ctk.CTkButton:
        app = self.app
        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=36,
            corner_radius=9,
            fg_color="#EEF2F7",
            hover_color="#E2E8F0",
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )
        self._buttons.append(button)
        return button

    def refresh_summary(self) -> None:
        app = self.app
        missing = app.t("devices_none")
        self.summary.configure(
            text=app.t("tests_devices").format(
                mic=app.out_input_var.get() or missing,
                phones=app.in_output_var.get() or missing,
                cable=app.out_output_var.get() or missing,
            )
        )

    def show_level(self, level: float) -> None:
        if self.winfo_exists():
            self.level.set(max(0.0, min(1.0, level)))
            self.wave.push(level)

    def show_status(self, text: str) -> None:
        if self.winfo_exists():
            self.status_var.set(text)

    def show_idle(self) -> None:
        if not self.winfo_exists() or self._busy or self._listening:
            return
        self.listen_button.configure(text=self.app.t("mic_listen"))
        self.level.set(0)

    def shutdown(self) -> None:
        self.stop_monitor()

    def _device(self, key: str) -> AudioDevice | None:
        variable = getattr(self.app, f"{key}_var")
        return self.app.device_maps.get(key, {}).get(variable.get())

    def _blocked(self) -> bool:
        pipeline = self.app.pipeline
        if pipeline and pipeline.running:
            self.app._show_error(self.app.t("test_pipeline_running"))
            return True
        return self._busy

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self._buttons:
            if button.winfo_exists():
                button.configure(state=state)

    def _post_status(self, key: str) -> None:
        self.app._post_event(-1, "test_status", self.app.t(key))

    def toggle_listen(self) -> None:
        if self._listening:
            self.stop_monitor()
            self.listen_button.configure(text=self.app.t("mic_listen"))
            self.level.set(0)
            return
        if self._blocked():
            return
        device = self._device("out_input")
        if device is None:
            self.app._show_error(self.app.t("test_need_mic"))
            return
        self._listening = True
        self.listen_button.configure(text=self.app.t("mic_stop"))
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            args=(device.uid,),
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop_monitor.set()
        thread = self._monitor_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._stop_monitor.clear()
        self._listening = False

    def _monitor_worker(self, device_uid: str) -> None:
        try:
            index = AudioDeviceManager.index_from_uid(device_uid, "sd")

            def callback(indata, _frames, _time_info, _status) -> None:
                rms = float(np.sqrt(np.mean(np.square(indata)) + 1e-12))
                self.app._post_event(-1, "mic_level", max(0.0, min(1.0, rms / 0.08)))

            with sd.InputStream(
                device=index,
                channels=1,
                samplerate=16_000,
                dtype="float32",
                callback=callback,
            ):
                while not self._stop_monitor.wait(0.1):
                    pass
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "mic_idle")

    def record_microphone(self) -> None:
        if self._blocked():
            return
        microphone = self._device("out_input")
        phones = self._device("in_output")
        if microphone is None:
            self.app._show_error(self.app.t("test_need_mic"))
            return
        if phones is None:
            self.app._show_error(self.app.t("test_need_output"))
            return
        self.stop_monitor()
        self.listen_button.configure(text=self.app.t("mic_listen"))
        self._set_busy(True)
        self._post_status("test_recording")
        threading.Thread(
            target=self._record_worker,
            args=(microphone.uid, phones.uid),
            daemon=True,
        ).start()

    def _record_worker(self, microphone_uid: str, phones_uid: str) -> None:
        try:
            audio, rate = self._capture(
                microphone_uid,
                3.0,
                on_level=lambda level: self.app._post_event(-1, "mic_level", level),
            )
            self._post_status("test_playing")
            time.sleep(0.4)
            AudioPlayer.play_samples(audio, rate, phones_uid)
            self._post_status("test_done")
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "test_idle")

    @staticmethod
    def _capture(device_uid: str, seconds: float, on_level=None) -> tuple[np.ndarray, int]:
        index = AudioDeviceManager.index_from_uid(device_uid, "sd")
        native = int(sd.query_devices(index)["default_samplerate"])
        rates = []
        for candidate in (native, 48_000, 44_100, 16_000):
            if candidate not in rates:
                rates.append(candidate)
        last_error: Exception | None = None
        for rate in rates:
            frames: list[np.ndarray] = []

            def callback(indata, _frame_count, _time_info, _status) -> None:
                frames.append(indata.copy())
                if on_level is not None:
                    rms = float(np.sqrt(np.mean(np.square(indata)) + 1e-12))
                    on_level(max(0.0, min(1.0, rms / 0.08)))

            try:
                sd.check_input_settings(
                    device=index,
                    channels=1,
                    dtype="float32",
                    samplerate=rate,
                )
                with sd.InputStream(
                    device=index,
                    channels=1,
                    samplerate=rate,
                    dtype="float32",
                    latency="high",
                    callback=callback,
                ):
                    threading.Event().wait(seconds)
            except Exception as exc:
                last_error = exc
                continue
            if frames:
                audio = np.concatenate(frames, axis=0)
                return audio[: int(seconds * rate)], rate
        if last_error is not None:
            raise last_error
        raise RuntimeError("Nenhuma amostra capturada")

    def play_headphones(self) -> None:
        device = self._require_output()
        if device is None:
            return
        self._play_async(device.uid, self._chime(), 48_000, pitch=0.0)

    def preview_pitch(self) -> None:
        device = self._require_output()
        if device is None:
            return
        self._play_async(device.uid, self._vowel(), 22_050, pitch=self.app._pitch_value())

    def play_virtual_output(self) -> None:
        if self._blocked():
            return
        device = self._device("out_output")
        if device is None:
            self.app._show_error(self.app.t("test_need_cable"))
            return
        self._play_async(device.uid, self._chime(520), 48_000, pitch=0.0)

    def play_voice_sample(self) -> None:
        device = self._require_output()
        if device is None:
            return
        self._set_busy(True)
        self._post_status("test_generating")
        threading.Thread(
            target=self._voice_worker,
            args=(device.uid,),
            daemon=True,
        ).start()

    def _voice_worker(self, device_uid: str) -> None:
        try:
            language = self.app.settings.native_language
            if language not in VOICE_MAP:
                language = "Inglês"
            audio_file = TTSEngine().synthesize(
                self.app.t("voice_sample_text"),
                language,
                self.app._gender_key(),
                self.app._mode_key(),
                self.app._rate_value(),
                pitch_semitones=self.app._pitch_value(),
                voice_engine=self.app._voice_engine_key(),
            )
            self._post_status("test_playing")
            AudioPlayer.play(
                audio_file,
                device_uid,
                pitch_semitones=0.0 if self.app._voice_engine_key() != "windows" else self.app._pitch_value(),
            )
            self._post_status("test_done")
        except Exception as exc:
            self.app._post_event(-1, "error", str(exc))
        finally:
            self.app._post_event(-1, "test_idle")

    def _require_output(self) -> AudioDevice | None:
        if self._blocked():
            return None
        device = self._device("in_output")
        if device is None:
            self.app._show_error(self.app.t("test_need_output"))
            return None
        return device

    def _play_async(self, device_uid: str, samples: np.ndarray, rate: int, pitch: float) -> None:
        self._set_busy(True)
        self._post_status("test_playing")

        def worker() -> None:
            try:
                AudioPlayer.play_samples(
                    samples,
                    rate,
                    device_uid,
                    pitch_semitones=pitch,
                )
                self._post_status("test_done")
            except Exception as exc:
                self.app._post_event(-1, "error", str(exc))
            finally:
                self.app._post_event(-1, "test_idle")

        threading.Thread(target=worker, daemon=True).start()

    def finish_busy(self) -> None:
        if self.winfo_exists():
            self._set_busy(False)

    @staticmethod
    def _chime(frequency: float = 660) -> np.ndarray:
        first = TestsView._tone(frequency, 0.18)
        gap = np.zeros(int(48_000 * 0.06), dtype=np.float32)
        second = TestsView._tone(frequency * 1.25, 0.28)
        return np.concatenate([first, gap, second])

    @staticmethod
    def _tone(frequency: float, seconds: float, rate: int = 48_000) -> np.ndarray:
        times = np.arange(int(rate * seconds), dtype=np.float32) / rate
        fade = np.minimum(1.0, times * 40) * np.minimum(1.0, (seconds - times) * 40)
        return (0.12 * np.sin(2 * np.pi * frequency * times) * fade).astype(np.float32)

    @staticmethod
    def _vowel() -> np.ndarray:
        rate = 22_050
        times = np.arange(int(rate * 0.7), dtype=np.float32) / rate
        tone = np.zeros_like(times)
        for harmonic, amplitude in ((1, 0.16), (2, 0.07), (3, 0.035)):
            tone += amplitude * np.sin(2 * np.pi * 196 * harmonic * times)
        envelope = np.minimum(1.0, times * 18) * np.minimum(1.0, (0.7 - times) * 8)
        return (tone * envelope).astype(np.float32)
