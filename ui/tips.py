from __future__ import annotations

import tkinter as tk

import customtkinter as ctk


SECTIONS = ("translate", "tests", "train", "settings")


class TipsCard(ctk.CTkFrame):
    def __init__(self, master, app, section: str) -> None:
        super().__init__(
            master,
            fg_color="#F8FAFC",
            corner_radius=12,
            border_width=1,
            border_color=app.BORDER,
        )
        self.app = app
        ctk.CTkLabel(
            self,
            text=app.t(f"tips_{section}_title"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            self,
            text=app.t(f"tips_{section}_body"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            wraplength=260 if section == "settings" else 720,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 10))


def open_tutorial(app) -> None:
    window = ctk.CTkToplevel(app.root)
    window.title(app.t("tutorial_title"))
    window.geometry("720x640")
    window.transient(app.root)
    window.lift()
    ctk.CTkLabel(
        window,
        text=app.t("tutorial_title"),
        text_color=app.TEXT,
        font=ctk.CTkFont("Segoe UI", 22, "bold"),
    ).pack(anchor="w", padx=24, pady=(20, 4))
    ctk.CTkLabel(
        window,
        text=app.t("tutorial_intro"),
        text_color=app.MUTED,
        font=ctk.CTkFont("Segoe UI", 13),
        wraplength=660,
        justify="left",
    ).pack(anchor="w", padx=24, pady=(0, 12))
    scroll = ctk.CTkScrollableFrame(window, fg_color="transparent")
    scroll.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
    for section in SECTIONS:
        card = ctk.CTkFrame(
            scroll,
            fg_color=app.PANEL,
            corner_radius=14,
            border_width=1,
            border_color=app.BORDER,
        )
        card.pack(fill=tk.X, pady=6)
        ctk.CTkLabel(
            card,
            text=app.t(f"tutorial_{section}_title"),
            text_color=app.TEXT,
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            card,
            text=app.t(f"tutorial_{section}_body"),
            text_color=app.MUTED,
            font=ctk.CTkFont("Segoe UI", 13),
            wraplength=640,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 14))
