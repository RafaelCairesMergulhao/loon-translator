import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import services.piper_engine as piper_module
from services.piper_engine import PiperEngine, PiperError, PiperWorker, find_executable
from services.voice_likeness import is_neural_voice

# Fala o protocolo do ``piper --json-input``: grava o WAV e devolve o caminho numa linha.
FAKE_PIPER = r"""
import json, sys, wave
scale = sys.argv[sys.argv.index("--length_scale") + 1]
for line in sys.stdin:
    request = json.loads(line)
    if request["text"] == "cai":
        sys.exit(3)
    with wave.open(request["output_file"], "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(b"\x00\x10" * 2205)
    print(request["output_file"], flush=True)
"""


@pytest.fixture
def fake_piper(tmp_path) -> list[str]:
    script = tmp_path / "fake_piper.py"
    script.write_text(FAKE_PIPER, encoding="utf-8")
    return [sys.executable, str(script)]


def test_piper_worker_reuses_one_process_and_recovers(tmp_path, fake_piper) -> None:
    worker = PiperWorker(fake_piper, tmp_path / "voz.onnx", "1.000", timeout_s=10)
    try:
        worker.synthesize("olá, tudo bem?", tmp_path / "a.wav")
        first = worker._process.pid
        worker.synthesize("pode começar sem mim", tmp_path / "b.wav")
        assert worker._process.pid == first, "a segunda frase usa o mesmo processo"
        assert (tmp_path / "b.wav").stat().st_size > 0
        with pytest.raises(PiperError):
            worker.synthesize("cai", tmp_path / "c.wav")
        worker.synthesize("de volta", tmp_path / "d.wav")
        assert (tmp_path / "d.wav").stat().st_size > 0
    finally:
        worker.close()


def test_piper_pool_keeps_the_recent_voices(tmp_path, monkeypatch) -> None:
    started: list[tuple] = []

    class FakeWorker:
        def __init__(self, command, model, length_scale, cwd=None):
            started.append((model.name, length_scale))
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(piper_module, "PiperWorker", FakeWorker)
    pool = piper_module._WorkerPool(capacity=2)
    exe = tmp_path / "piper.exe"
    english = pool.get(exe, tmp_path / "en.onnx", "1.000")
    portuguese = pool.get(exe, tmp_path / "pt.onnx", "1.000")
    assert pool.get(exe, tmp_path / "en.onnx", "1.000") is english
    pool.get(exe, tmp_path / "en.onnx", "0.800")
    assert started == [("en.onnx", "1.000"), ("pt.onnx", "1.000"), ("en.onnx", "0.800")]
    assert pool.get(exe, tmp_path / "en.onnx", "1.000") is english
    assert pool.get(exe, tmp_path / "pt.onnx", "1.000") is not portuguese, "a menos usada foi fechada"


def test_piper_engine_uses_the_open_process(tmp_path, monkeypatch, fake_piper) -> None:
    if not str(tmp_path.resolve()).isascii():
        pytest.skip("caminho com acento usa o processo avulso")
    model = tmp_path / "voz.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setenv("PIPER_MODEL", str(model))
    pool = piper_module._WorkerPool()
    real_worker = piper_module.PiperWorker
    monkeypatch.setattr(piper_module, "_POOL", pool)
    monkeypatch.setattr(
        piper_module,
        "PiperWorker",
        lambda _command, model, length_scale, cwd=None: real_worker(fake_piper, model, length_scale, timeout_s=10),
    )
    monkeypatch.setattr(PiperEngine, "find_binary", lambda self: tmp_path / "piper.exe")
    monkeypatch.setattr(PiperEngine, "_run_once", staticmethod(lambda *_args: pytest.fail("abriu um processo avulso")))
    try:
        engine = PiperEngine(root=tmp_path)
        output = engine.synthesize("bom dia", "Português", "Feminina", tmp_path / "bom.piper.wav")
        assert output.stat().st_size > 0
    finally:
        pool.close()


@pytest.fixture
def offline_piper(tmp_path, monkeypatch):
    """Nada instalado, sem piper no PATH nem pacote embutido; os downloads passam por ``serve``."""
    monkeypatch.delenv("PIPER_MODEL", raising=False)
    monkeypatch.setattr(piper_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(piper_module, "bundled_root", lambda: None)
    monkeypatch.setattr(piper_module, "_INSTALLS", {})
    warmed: list[str] = []
    monkeypatch.setattr(
        piper_module,
        "_POOL",
        SimpleNamespace(get=lambda _exe, model, _scale: SimpleNamespace(ensure_started=lambda: warmed.append(model.name))),
    )
    state = SimpleNamespace(downloads=[], warmed=warmed, serve=None)

    def download(url: str, destination: Path) -> None:
        state.downloads.append(url.rsplit("/", 1)[-1])
        state.serve(url, destination)

    monkeypatch.setattr(piper_module, "_download", download)
    return state


def _wait_installs(timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while any(value == float("inf") for value in piper_module._INSTALLS.values()):
        assert time.monotonic() < deadline, "a instalação não terminou"
        time.sleep(0.01)


def test_missing_voice_installs_in_the_background(tmp_path, offline_piper) -> None:
    release = threading.Event()

    def serve(url: str, destination: Path) -> None:
        assert release.wait(10)
        if url == piper_module.BINARY_URL:
            with zipfile.ZipFile(destination, "w") as bundle:
                bundle.writestr("piper/piper.exe", b"")
        else:
            destination.write_bytes(b"{}")

    offline_piper.serve = serve
    engine = PiperEngine(root=tmp_path)
    started = time.perf_counter()
    for text in ("bom dia", "tudo bem?"):
        with pytest.raises(PiperError, match="instalada"):
            engine.synthesize(text, "Inglês", "Feminina", tmp_path / f"{text}.piper.wav")
    assert time.perf_counter() - started < 1.0, "a frase não espera o download"
    release.set()
    _wait_installs()
    assert offline_piper.downloads == [
        "piper_windows_amd64.zip",
        "en_US-amy-medium.onnx",
        "en_US-amy-medium.onnx.json",
    ], "duas frases, uma instalação"
    assert engine.find_binary() == tmp_path / "runtime" / "piper" / "piper.exe"
    assert engine.find_model("Inglês", "Feminina") == tmp_path / "voices" / "en_US-amy-medium.onnx"
    assert not (tmp_path / "runtime.part").exists()
    assert offline_piper.warmed == ["en_US-amy-medium.onnx"], "o processo já abre com a voz nova"


def test_failed_install_waits_before_trying_again(tmp_path, offline_piper) -> None:
    def serve(_url: str, _destination: Path) -> None:
        raise OSError("sem rede")

    offline_piper.serve = serve
    engine = PiperEngine(root=tmp_path)
    with pytest.raises(PiperError):
        engine.synthesize("bom dia", "Português", "Feminina", tmp_path / "a.piper.wav")
    _wait_installs()
    with pytest.raises(PiperError):
        engine.synthesize("boa tarde", "Português", "Feminina", tmp_path / "b.piper.wav")
    assert offline_piper.downloads == ["piper_windows_amd64.zip"]
    assert not (tmp_path / "runtime").exists()


def test_find_executable_sees_piper_exe(tmp_path) -> None:
    exe = tmp_path / "piper.exe"
    exe.write_bytes(b"")
    assert find_executable(tmp_path) == exe
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "piper.exe").write_bytes(b"")
    assert find_executable(tmp_path) is not None


def test_piper_honours_custom_model_path(tmp_path, monkeypatch) -> None:
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setenv("PIPER_MODEL", str(model))
    engine = PiperEngine(root=tmp_path / "piper")
    assert engine.ensure_model("Klingon", "Feminina") == model


def test_piper_rejects_unknown_language(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PIPER_MODEL", raising=False)
    engine = PiperEngine(root=tmp_path)
    try:
        engine.ensure_model("Klingon", "Feminina")
    except Exception as exc:
        assert "Klingon" in str(exc)
    else:
        raise AssertionError("esperava PiperError")


def test_piper_wav_is_treated_as_neural() -> None:
    assert is_neural_voice(Path("abc.piper.wav"))
    assert is_neural_voice(Path("abc.neural.mp3"))
    assert not is_neural_voice(Path("abc.sapi.wav"))
