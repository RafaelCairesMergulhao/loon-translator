import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import customtkinter as ctk
from core.config import APP_DIR, load_environment
from ui.gui_app import LoonApp


def configure_logging() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        APP_DIR / "loon.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
    if os.getenv("LOON_LOG_TRANSCRIPTS", "false").lower() not in {"1", "true", "yes"}:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    load_environment()
    configure_logging()
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    icon = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "assets" / "loon.ico"
    if icon.exists():
        # O CustomTkinter reaplica o ícone padrão logo após criar a janela.
        root.after(250, lambda: root.iconbitmap(str(icon)))
    LoonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()