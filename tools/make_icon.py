"""Gera ``assets/loon.ico`` (16 a 256 px) só com numpy: formas por campos de distância com
supersampling 4x para bordas suaves; cada tamanho vira um PNG dentro do ICO.

Uso:
    python -m tools.make_icon
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 4
TOP = np.array([14, 165, 233], dtype=np.float64)  # azul céu
BOTTOM = np.array([79, 70, 229], dtype=np.float64)  # índigo
BARS = ((-0.30, 0.16), (-0.15, 0.30), (0.0, 0.42), (0.15, 0.30), (0.30, 0.16))


def _rounded_box(x: np.ndarray, y: np.ndarray, half: float, radius: float) -> np.ndarray:
    qx = np.abs(x) - (half - radius)
    qy = np.abs(y) - (half - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - radius


def _capsule(x: np.ndarray, y: np.ndarray, cx: float, half_height: float, radius: float) -> np.ndarray:
    dy = np.maximum(np.abs(y) - half_height, 0)
    return np.hypot(x - cx, dy) - radius


def render(size: int) -> np.ndarray:
    n = size * SUPERSAMPLE
    coords = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(coords, coords)
    pixel = 2 / n

    background = np.clip(0.5 - _rounded_box(x, y, 0.94, 0.42) / pixel, 0, 1)
    bar_width = 0.052 if size >= 32 else 0.075
    bars = np.full_like(x, np.inf)
    for cx, half_height in BARS:
        bars = np.minimum(bars, _capsule(x, y, cx, half_height, bar_width))
    foreground = np.clip(0.5 - bars / pixel, 0, 1)

    t = ((y + 1) / 2)[..., None]
    color = TOP * (1 - t) + BOTTOM * t
    color = color * (1 - foreground[..., None]) + 255 * foreground[..., None]
    rgba = np.concatenate([color, (background * 255)[..., None]], axis=-1)
    rgba = rgba.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    return np.clip(np.round(rgba), 0, 255).astype(np.uint8)


def png(rgba: np.ndarray) -> bytes:
    height, width, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def ico(images: list[bytes], sizes: tuple[int, ...]) -> bytes:
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for size, data in zip(sizes, images):
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return header + entries + b"".join(images)


def main() -> None:
    target = Path(__file__).resolve().parent.parent / "assets"
    target.mkdir(exist_ok=True)
    images = [png(render(size)) for size in SIZES]
    (target / "loon.ico").write_bytes(ico(images, SIZES))
    (target / "loon.png").write_bytes(images[-1])
    print(f"Ícone salvo em {target / 'loon.ico'}")


if __name__ == "__main__":
    main()
