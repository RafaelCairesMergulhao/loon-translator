"""Detecção de voz quadro a quadro para a captura em tempo real.

O Silero VAD v6 já vem dentro do faster-whisper (ONNX, ~2 MB): não precisa de
torch nem de pacote novo. Cada fluxo guarda o próprio estado da LSTM e as
últimas 64 amostras do quadro anterior, então microfone e retorno da chamada
dividem a mesma sessão ONNX sem misturar contexto.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
_CONTEXT_SAMPLES = 64


class VoiceDetector(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroVAD:
    """Probabilidade de fala de cada quadro de 32 ms; ~0,1 ms por quadro na CPU."""

    _session = None
    _session_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = self._shared_session()
        self.reset()

    @classmethod
    def _shared_session(cls):
        with cls._session_lock:
            if cls._session is None:
                import onnxruntime
                from faster_whisper.utils import get_assets_path

                options = onnxruntime.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.log_severity_level = 4
                cls._session = onnxruntime.InferenceSession(
                    os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
                    providers=["CPUExecutionProvider"],
                    sess_options=options,
                )
            return cls._session

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        window = np.concatenate((self._context, frame)).reshape(1, -1)
        output, self._h, self._c = self._model.run(None, {"input": window, "h": self._h, "c": self._c})
        self._context = frame[-_CONTEXT_SAMPLES:].copy()
        return float(output.reshape(-1)[0])


class EnergyVAD:
    """Reserva sem ONNX: energia acima de um piso de ruído que se adapta no silêncio."""

    def __init__(self, noise_floor: float = 0.006) -> None:
        self.noise_floor = noise_floor

    def reset(self) -> None:
        """O piso de ruído descreve o ambiente, não a frase: sobrevive ao reinício."""

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if rms >= max(0.012, self.noise_floor * 3.2):
            return 1.0
        self.noise_floor = self.noise_floor * 0.97 + rms * 0.03
        return 0.0


def create_vad() -> VoiceDetector:
    try:
        return SileroVAD()
    except Exception as exc:
        logger.warning("Silero VAD indisponível; usando detecção por energia: %s", exc)
        return EnergyVAD()
