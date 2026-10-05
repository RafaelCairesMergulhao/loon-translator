from __future__ import annotations

import tkinter as tk
from collections import deque

import customtkinter as ctk


class WaveformMeter(ctk.CTkFrame):
    """Barras que acompanham a voz enquanto a pessoa fala."""

    def __init__(self, master, color: str = "#2563EB", bars: int = 42) -> None:
        super().__init__(master, fg_color="transparent", height=72)
        self._color = color
        self._levels: deque[float] = deque([0.0] * bars, maxlen=bars)
        self.canvas = tk.Canvas(self, height=72, highlightthickness=0, bg="#F8FAFC")
        self.canvas.pack(fill=tk.X, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def push(self, level: float) -> None:
        self._levels.append(max(0.0, min(1.0, level)))
        self.redraw()

    def clear(self) -> None:
        for index in range(len(self._levels)):
            self._levels[index] = 0.0
        self.redraw()

    def redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("wave")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        count = len(self._levels)
        gap = 3
        bar_width = max(2, (width - gap * (count + 1)) / count)
        mid = height / 2
        for index, level in enumerate(self._levels):
            bar_height = max(2, level * (height - 8))
            x0 = gap + index * (bar_width + gap)
            canvas.create_rectangle(
                x0,
                mid - bar_height / 2,
                x0 + bar_width,
                mid + bar_height / 2,
                fill=self._color,
                width=0,
                tags="wave",
            )
