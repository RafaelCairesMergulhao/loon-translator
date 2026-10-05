from pathlib import Path

from services.piper_engine import PiperEngine, find_executable
from services.voice_likeness import is_neural_voice


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
