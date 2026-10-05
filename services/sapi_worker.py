"""Voz do Windows (System.Speech) servida por um PowerShell que fica aberto.

Abrir um PowerShell e carregar System.Speech a cada frase custa de 0,3 a 1,5 s,
conforme a máquina. O processo persistente paga esse custo uma vez e depois
atende cada pedido em cerca de 100 ms.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import math
import os
import queue
import subprocess
import threading
from pathlib import Path

from services.speech_prosody import sapi_ssml

logger = logging.getLogger(__name__)

_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
function Decode([string] $value) { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($value)) }
[Console]::Out.WriteLine('READY')
[Console]::Out.Flush()
while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    try {
        $request = $line | ConvertFrom-Json
        $prefix = $request.culture.Substring(0, 2)
        $candidates = @($voices | Where-Object { $_.VoiceInfo.Culture.Name.StartsWith($prefix) })
        if ($candidates.Count -eq 0) { throw "Nenhuma voz instalada para $($request.culture)" }
        $preferred = $candidates | Where-Object { $_.VoiceInfo.Gender.ToString() -eq $request.gender } |
            Select-Object -First 1
        if ($null -eq $preferred) { $preferred = $candidates[0] }
        $synth.SelectVoice($preferred.VoiceInfo.Name)
        $synth.Rate = 0
        $synth.SetOutputToWaveFile((Decode $request.output))
        if ($request.ssml) {
            $synth.SpeakSsml((Decode $request.ssml))
        } else {
            $synth.Speak((Decode $request.text))
        }
        $synth.SetOutputToNull()
        [Console]::Out.WriteLine('OK')
    } catch {
        try { $synth.SetOutputToNull() } catch {}
        [Console]::Out.WriteLine('ERR ' + ($_.Exception.Message -replace '\s+', ' '))
    }
    [Console]::Out.Flush()
}
"""


class SapiError(RuntimeError):
    pass


def sapi_rate(speed: float) -> int:
    """SAPI usa -10..10; +10 fica perto de 3x, então a escala é logarítmica de base 3."""
    speed = max(0.5, min(2.0, float(speed)))
    return int(max(-10, min(10, round(10 * math.log(speed) / math.log(3)))))


def _encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


class SapiWorker:
    _instance: SapiWorker | None = None
    _instance_lock = threading.Lock()

    def __init__(self, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    @classmethod
    def shared(cls) -> SapiWorker:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
                atexit.register(cls._instance.close)
            return cls._instance

    def _start(self) -> None:
        if os.name != "nt":
            raise SapiError("SAPI está disponível apenas no Windows")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._lines = queue.Queue()
        self._process = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        threading.Thread(
            target=self._pump,
            args=(self._process, self._lines),
            name="loon-sapi-reader",
            daemon=True,
        ).start()
        if self._read() != "READY":
            self._kill()
            raise SapiError("O serviço de voz do Windows não iniciou")

    @staticmethod
    def _pump(process: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.strip())
        lines.put(None)

    def _read(self) -> str:
        try:
            line = self._lines.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            self._kill()
            raise SapiError("A voz do Windows não respondeu a tempo") from exc
        if line is None:
            self._kill()
            raise SapiError("O serviço de voz do Windows foi encerrado")
        return line

    def ensure_started(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()

    def synthesize(self, text: str, culture: str, gender: str, output: Path, speed: float = 1.0) -> Path:
        request = json.dumps(
            {
                "culture": culture,
                "gender": gender,
                "rate": sapi_rate(speed),
                "output": _encode(str(output.resolve())),
                "text": _encode(text),
                "ssml": _encode(sapi_ssml(text, culture, speed)),
            }
        )
        with self._lock:
            for attempt in range(2):
                if self._process is None or self._process.poll() is not None:
                    self._start()
                assert self._process is not None and self._process.stdin is not None
                try:
                    self._process.stdin.write(request + "\n")
                    self._process.stdin.flush()
                except OSError:
                    self._kill()
                    if attempt == 0:
                        continue
                    raise
                reply = self._read()
                if reply == "OK":
                    break
                raise SapiError(reply.removeprefix("ERR ").strip() or "falha desconhecida do Windows SAPI")
        if not output.exists() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise SapiError("A voz do Windows não gerou áudio")
        return output

    def _kill(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.kill()
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
