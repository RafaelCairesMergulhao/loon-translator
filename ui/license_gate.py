"""Tela que só deixa o programa abrir com a chave desta cópia."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from services.licensing import ActivationLedger, LicenseError, license_server_url
from ui.i18n import ui_text
from ui.win_chrome import bind_opaque

BG = "#080d14"
PANEL = "#0d1620"
FIELD = "#101a25"
TEXT = "#eaf1f8"
MUTED = "#7f8e9e"
BORDER = "#1c2b38"
BLUE = "#39d7e4"
BLUE_HOVER = "#6de4f1"
INK = "#031216"


def _t(key: str) -> str:
    return ui_text("Português", key)


def prompt_license() -> bool:
    accepted = {"ok": False}
    root = ctk.CTk()
    root.title("Loon Translator")
    root.geometry("560x460")
    root.minsize(520, 430)
    root.configure(fg_color=BG)
    bind_opaque(root)
    ctk.CTkLabel(
        root,
        text=_t("license_title"),
        text_color=TEXT,
        font=ctk.CTkFont("Segoe UI", 22, "bold"),
    ).pack(anchor="w", padx=28, pady=(24, 8))
    ctk.CTkLabel(
        root,
        text=_t("license_body"),
        text_color=MUTED,
        font=ctk.CTkFont("Segoe UI", 13),
        wraplength=500,
        justify="left",
    ).pack(anchor="w", padx=28)
    key = ctk.CTkTextbox(
        root,
        height=90,
        fg_color=FIELD,
        text_color=TEXT,
        border_width=1,
        border_color=BORDER,
        corner_radius=10,
        font=ctk.CTkFont("Consolas", 12),
        wrap="word",
    )
    key.pack(fill=tk.X, padx=28, pady=(16, 10))
    accepted_terms = tk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        root,
        text=_t("license_accept"),
        variable=accepted_terms,
        text_color=TEXT,
        fg_color=BLUE,
        hover_color=BLUE_HOVER,
        border_color=BORDER,
        font=ctk.CTkFont("Segoe UI", 12),
    ).pack(anchor="w", padx=28)
    ctk.CTkLabel(
        root,
        text=_t("license_terms"),
        text_color=MUTED,
        font=ctk.CTkFont("Segoe UI", 12),
        wraplength=500,
        justify="left",
    ).pack(anchor="w", padx=28, pady=(6, 8))
    status = tk.StringVar(value="")
    ctk.CTkLabel(
        root,
        textvariable=status,
        text_color="#ff8b95",
        font=ctk.CTkFont("Segoe UI", 12),
        wraplength=500,
        justify="left",
    ).pack(anchor="w", padx=28)

    def activate() -> None:
        if not accepted_terms.get():
            status.set(_t("license_need_terms"))
            return
        token = key.get("1.0", tk.END).strip()
        try:
            ActivationLedger(server_url=license_server_url()).submit(token)
        except LicenseError as exc:
            status.set(_t(f"license_{exc.code}" if exc.code in {"invalid", "used", "server"} else "license_invalid"))
            return
        accepted["ok"] = True
        root.destroy()

    ctk.CTkButton(
        root,
        text=_t("license_activate"),
        command=activate,
        height=42,
        corner_radius=10,
        fg_color=BLUE,
        hover_color=BLUE_HOVER,
        text_color=INK,
        font=ctk.CTkFont("Segoe UI", 14, "bold"),
    ).pack(anchor="w", padx=28, pady=(8, 0))
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return accepted["ok"]
