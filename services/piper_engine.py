"""Piper: voz neural local (ONNX), a voz padrão do Loon.

O binário oficial do Windows (~21 MB) e as vozes (~60 MB cada) são baixados
uma vez para ``%APPDATA%\\LoonTranslator\\piper``, em segundo plano: até a voz
ficar pronta, cada frase vai para a próxima voz da cadeia. No app instalado o
binário já vem na pasta do programa; só as vozes faltantes são baixadas.

Abrir o piper.exe carrega o modelo de novo (~300 ms); um processo aberto com
``--json-input`` atende cada frase em 100 a 160 ms.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import httpx

from core.config import APP_DIR
from services.pitch_shift import shift_pitch

logger = logging.getLogger(__name__)

BINARY_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
HF_VOICE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Caminho relativo no repositório Hugging Face → nome do arquivo .onnx
_VOICES: dict[str, dict[str, str]] = {
    "Português": {
        "Feminina": "pt/pt_BR/faber/medium/pt_BR-faber-medium",
        "Masculina": "pt/pt_BR/faber/medium/pt_BR-faber-medium",
    },
    "Inglês": {
        "Feminina": "en/en_US/amy/medium/en_US-amy-medium",
        "Masculina": "en/en_US/lessac/medium/en_US-lessac-medium",
    },
    "Espanhol": {
        "Feminina": "es/es_ES/sharvard/medium/es_ES-sharvard-medium",
        "Masculina": "es/es_ES/davefx/medium/es_ES-davefx-medium",
    },
    "Espanhol (Latino)": {
        "Feminina": "es/es_MX/ald/medium/es_MX-ald-medium",
        "Masculina": "es/es_MX/ald/medium/es_MX-ald-medium",
    },
    "Francês": {
        "Feminina": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
        "Masculina": "fr/fr_FR/upmc/medium/fr_FR-upmc-medium",
    },
    "Alemão": {
        "Feminina": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
        "Masculina": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    },
    "Italiano": {
        "Feminina": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
        "Masculina": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
    },
    "Hindi": {
        "Feminina": "hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium",
        "Masculina": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium",
    },
    "Árabe": {
        "Feminina": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
        "Masculina": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
    },
    "Chinês (Mandarim)": {
        "Feminina": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
        "Masculina": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
    },
    "Russo": {
        "Feminina": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
        "Masculina": "ru/ru_RU/denis/medium/ru_RU-denis-medium",
    },
}


_VOICE_FLAGS = ("--sentence_silence", "0.18", "--noise_scale", "0.667", "--noise_w", "0.8")


class PiperError(RuntimeError):
    pass


class PiperWorker:
    """Um piper.exe aberto com ``--json-input``; a velocidade é fixa por processo."""

    def __init__(
        self,
        command: Sequence[str],
        model: Path,
        length_scale: str,
        cwd: Path | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.timeout_s = timeout_s
        self._args = [*command, "--model", str(model), "--json-input", "--length_scale", length_scale, *_VOICE_FLAGS]
        self._cwd = str(cwd) if cwd else None
        self._process: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    def ensure_started(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()

    def synthesize(self, text: str, output: Path) -> None:
        # O Piper de 2023 lê o caminho do JSON na página de código ANSI: só ASCII é seguro.
        request = (json.dumps({"text": text, "output_file": str(output)}) + "\n").encode("ascii")
        with self._lock:
            for attempt in range(2):
                if self._process is None or self._process.poll() is not None:
                    self._start()
                assert self._process is not None and self._process.stdin is not None
                try:
                    self._process.stdin.write(request)
                    self._process.stdin.flush()
                    break
                except OSError as exc:
                    self._kill()
                    if attempt:
                        raise PiperError(f"Piper fechou a entrada: {exc}") from exc
            try:
                reply = self._lines.get(timeout=self.timeout_s)
            except queue.Empty as exc:
                self._kill()
                raise PiperError("Piper não respondeu a tempo") from exc
            if reply is None:
                self._kill()
                raise PiperError("Piper foi encerrado")

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    def _start(self) -> None:
        self._lines = queue.Queue()
        self._process = subprocess.Popen(
            self._args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._cwd,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(
            target=self._pump, args=(self._process, self._lines), name="loon-piper-reader", daemon=True
        ).start()

    @staticmethod
    def _pump(process: subprocess.Popen[bytes], lines: queue.Queue[str | None]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.decode("utf-8", errors="replace").strip())
        lines.put(None)

    def _kill(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass


class _WorkerPool:
    """Ida, volta e uma velocidade de recuperar atraso; a menos usada fecha."""

    def __init__(self, capacity: int = 3) -> None:
        self.capacity = capacity
        self._workers: OrderedDict[tuple[str, str, str], PiperWorker] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, executable: Path, model: Path, length_scale: str) -> PiperWorker:
        key = (str(executable), str(model), length_scale)
        with self._lock:
            worker = self._workers.pop(key, None) or PiperWorker(
                [str(executable)], model, length_scale, cwd=executable.parent
            )
            self._workers[key] = worker
            evicted = [self._workers.popitem(last=False)[1] for _ in range(len(self._workers) - self.capacity)]
        for old in evicted:
            threading.Thread(target=old.close, name="loon-piper-close", daemon=True).start()
        return worker

    def close(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.close()


_POOL = _WorkerPool()
atexit.register(_POOL.close)

# Um download ou cópia por vez, valendo para todas as instâncias de PiperEngine.
_INSTALL_LOCK = threading.Lock()
# Instalações em segundo plano: em andamento (inf) ou falharam e só voltam a tentar no instante guardado.
_INSTALLS: dict[tuple[str, str, str], float] = {}
_INSTALLS_LOCK = threading.Lock()
_RETRY_S = 60.0


def bundled_root() -> Path | None:
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "piper")
    candidates.append(Path(__file__).resolve().parent.parent / "vendor" / "piper")
    for folder in candidates:
        if find_executable(folder) is not None:
            return folder
    return None


def find_executable(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    direct = folder / "piper.exe" if os.name == "nt" else folder / "piper"
    if direct.exists():
        return direct
    matches = list(folder.rglob("piper.exe" if os.name == "nt" else "piper"))
    return matches[0] if matches else None


class PiperEngine:
    """Garante o binário e a voz, depois sintetiza WAV."""

    def __init__(self, root: Path | None = None) -> None:
        self.user_root = root or APP_DIR / "piper"

    def available_for(self, language_name: str) -> bool:
        return language_name in _VOICES or bool(os.getenv("PIPER_MODEL", "").strip())

    def synthesize(
        self,
        text: str,
        language_name: str,
        gender: str,
        output: Path,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> Path:
        if output.exists() and output.stat().st_size > 0:
            return output
        model = self.find_model(language_name, gender)
        executable = self.find_binary()
        if model is None or executable is None:
            self._install_later(language_name, gender, speed)
            raise PiperError(f"A voz Piper de {language_name} ainda está sendo instalada")
        length_scale = _length_scale(speed)
        target = output.resolve()
        if str(target).isascii():
            try:
                _POOL.get(executable, model, length_scale).synthesize(text, target)
            except PiperError as exc:
                logger.info("Piper aberto falhou; usando um processo só para esta frase: %s", exc)
                self._run_once(executable, model, length_scale, text, target)
        else:
            self._run_once(executable, model, length_scale, text, target)
        if not output.exists() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise PiperError("Piper não gerou áudio")
        if abs(pitch_semitones) >= 0.05:
            self._apply_pitch(output, pitch_semitones)
        return output

    def warm_up(self, language_name: str, gender: str, speed: float = 1.0) -> None:
        """Abre o processo da voz antes da primeira frase, que senão pagaria a carga do modelo."""
        model = self.find_model(language_name, gender)
        executable = self.find_binary()
        if model is None or executable is None:
            self._install_later(language_name, gender, speed)
            return
        _POOL.get(executable, model, _length_scale(speed)).ensure_started()

    def _install_later(self, language_name: str, gender: str, speed: float) -> None:
        """Baixar ~80 MB no meio da conversa atrasaria cada frase em dezenas de segundos."""
        key = (str(self.user_root), language_name, gender)
        with _INSTALLS_LOCK:
            if _INSTALLS.get(key, 0.0) > time.monotonic():
                return
            _INSTALLS[key] = math.inf

        def install() -> None:
            retry_at = 0.0
            try:
                executable = self.ensure_binary()
                model = self.ensure_model(language_name, gender)
                _POOL.get(executable, model, _length_scale(speed)).ensure_started()
                logger.info("Voz Piper pronta: %s", model.name)
            except Exception as exc:
                retry_at = time.monotonic() + _RETRY_S
                logger.warning("Não deu para instalar a voz Piper de %s: %s", language_name, exc)
            finally:
                with _INSTALLS_LOCK:
                    if retry_at:
                        _INSTALLS[key] = retry_at
                    else:
                        _INSTALLS.pop(key, None)

        threading.Thread(target=install, name="loon-piper-install", daemon=True).start()

    @staticmethod
    def _run_once(executable: Path, model: Path, length_scale: str, text: str, output: Path) -> None:
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    "--model",
                    str(model),
                    "--output_file",
                    str(output),
                    "--length_scale",
                    length_scale,
                    *_VOICE_FLAGS,
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=45,
                check=False,
                cwd=str(executable.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output.unlink(missing_ok=True)
            raise PiperError(f"Piper não respondeu: {exc}") from exc
        if completed.returncode != 0:
            output.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace").strip()
            raise PiperError(f"Piper falhou: {detail[:240] or completed.returncode}")

    def find_binary(self) -> Path | None:
        """Só o que já está no disco; não baixa nem espera outra instalação."""
        override = shutil.which("piper")
        if override:
            return Path(override)
        return find_executable(self.user_root / "runtime")

    def find_model(self, language_name: str, gender: str) -> Path | None:
        custom = os.getenv("PIPER_MODEL", "").strip()
        if custom:
            path = Path(custom)
            if not path.exists():
                raise PiperError(f"PIPER_MODEL não encontrado: {path}")
            return path
        _relative, onnx = self._voice_file(language_name, gender)
        return onnx if _complete(onnx) else None

    def ensure_binary(self) -> Path:
        found = self.find_binary()
        if found is not None:
            return found
        runtime = self.user_root / "runtime"
        staging = self.user_root / "runtime.part"
        with _INSTALL_LOCK:
            found = find_executable(runtime)
            if found is not None:
                return found
            shutil.rmtree(staging, ignore_errors=True)
            bundled = bundled_root()
            if bundled is not None:
                # O Piper 2023 quebra em caminhos com acento (OneDrive «Área de Trabalho»).
                shutil.copytree(find_executable(bundled).parent, staging)
            else:
                self.user_root.mkdir(parents=True, exist_ok=True)
                archive = self.user_root / "piper_windows_amd64.zip"
                logger.info("Baixando o Piper para voz local...")
                _download(BINARY_URL, archive)
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(staging)
                archive.unlink(missing_ok=True)
            if find_executable(staging) is None:
                raise PiperError("O pacote do Piper não trouxe piper.exe")
            # Quem procura sem a trava só enxerga a pasta pronta, nunca um piper.exe sem as DLLs.
            shutil.rmtree(runtime, ignore_errors=True)
            staging.replace(runtime)
            found = find_executable(runtime)
            if found is None:
                raise PiperError("O Piper não ficou em runtime")
            return found

    def ensure_model(self, language_name: str, gender: str) -> Path:
        found = self.find_model(language_name, gender)
        if found is not None:
            return found
        relative, onnx = self._voice_file(language_name, gender)
        config = onnx.with_suffix(".onnx.json")
        with _INSTALL_LOCK:
            if _complete(onnx):
                return onnx
            onnx.parent.mkdir(parents=True, exist_ok=True)
            bundled = bundled_root()
            packaged = bundled / "voices" / onnx.name if bundled else None
            # A configuração vai por último: com ela no disco, o modelo já está inteiro.
            if packaged is not None and _complete(packaged):
                shutil.copy2(packaged, onnx)
                shutil.copy2(packaged.with_suffix(".onnx.json"), config)
            else:
                logger.info("Baixando voz Piper %s...", onnx.stem)
                _download(f"{HF_VOICE}/{relative}.onnx", onnx)
                _download(f"{HF_VOICE}/{relative}.onnx.json", config)
            return onnx

    def _voice_file(self, language_name: str, gender: str) -> tuple[str, Path]:
        voices = _VOICES.get(language_name)
        if not voices:
            raise PiperError(f"Piper ainda não tem voz para {language_name}")
        relative = voices.get(gender) or next(iter(voices.values()))
        return relative, self.user_root / "voices" / f"{relative.rsplit('/', 1)[-1]}.onnx"

    @staticmethod
    def _apply_pitch(path: Path, semitones: float) -> None:
        import numpy as np
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        shifted = shift_pitch(data, semitones)
        sf.write(path, np.asarray(shifted, dtype="float32"), rate)


def _length_scale(speed: float) -> str:
    return f"{1.0 / max(0.5, min(2.0, float(speed))):.3f}"


def _complete(onnx: Path) -> bool:
    return onnx.exists() and onnx.with_suffix(".onnx.json").exists()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 256):
                        handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
