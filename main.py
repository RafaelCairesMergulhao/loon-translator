import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import customtkinter as ctk
from core.config import APP_DIR, load_environment
from services.licensing import activated_license
from ui.gui_app import LoonApp
from ui.license_gate import prompt_license


def configure_logging() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        APP_DIR / "loon.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [file_handler]
    # Aberto por outro programa, a saída fica num cano. Escrever nele trava a janela
    # quando o cano enche. No terminal normal o texto continua aparecendo.
    if sys.stderr is not None and sys.stderr.isatty():
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        handlers.insert(0, console)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    threading.excepthook = _log_thread_exception
    if os.getenv("LOON_LOG_TRANSCRIPTS", "false").lower() not in {"1", "true", "yes"}:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    logging.getLogger(__name__).error(
        "Exceção na thread %s",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def main() -> None:
    load_environment()
    configure_logging()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    if activated_license() is None and not prompt_license():
        return
    root = ctk.CTk()
    icon = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "assets" / "loon.ico"
    if icon.exists() and sys.platform.startswith("win"):
        # O CustomTkinter reaplica o ícone padrão logo após criar a janela.
        # iconbitmap é do Windows; no Mac e no Linux o Tk usa o tema do sistema.
        root.after(250, lambda: root.iconbitmap(str(icon)))
    LoonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()