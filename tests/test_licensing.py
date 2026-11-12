from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from services.licensing import (
    ActivationLedger,
    LicenseError,
    export_public_pem,
    issue_token,
    load_public_key,
    verify_token,
)
from tools.license_server import activate
import sqlite3


def _pair(tmp_path: Path) -> tuple[Ed25519PrivateKey, Path]:
    private = Ed25519PrivateKey.generate()
    public = tmp_path / "public.pem"
    public.write_bytes(export_public_pem(private))
    return private, public


def test_only_a_signed_key_for_this_product_opens(tmp_path: Path) -> None:
    private, public = _pair(tmp_path)
    token = issue_token("Maria Souza", private)
    license_info = verify_token(token, load_public_key(public))
    assert license_info.name == "Maria Souza"
    other = Ed25519PrivateKey.generate()
    foreign = issue_token("Maria Souza", other)
    with pytest.raises(LicenseError) as error:
        verify_token(foreign, load_public_key(public))
    assert error.value.code == "invalid"
    with pytest.raises(LicenseError):
        verify_token(token[:-4] + "AAAA", load_public_key(public))


def test_the_same_key_does_not_open_a_second_computer(tmp_path: Path) -> None:
    private, public = _pair(tmp_path)
    key = load_public_key(public)
    token = issue_token("Rafael", private)
    path = tmp_path / "license.json"
    first = ActivationLedger(path, machine="pc-1", public_key=key)
    assert first.submit(token).name == "Rafael"
    assert ActivationLedger(path, machine="pc-1", public_key=key).current().name == "Rafael"
    assert ActivationLedger(path, machine="pc-2", public_key=key).current() is None
    with pytest.raises(LicenseError) as error:
        ActivationLedger(path, machine="pc-2", public_key=key).submit(token)
    assert error.value.code == "used"


def test_server_keeps_the_key_on_the_first_computer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    private, public = _pair(tmp_path)
    token = issue_token("Ana", private)
    monkeypatch.setattr("tools.license_server.verify_token", lambda raw: verify_token(raw, load_public_key(public)))
    connection = sqlite3.connect(tmp_path / "activations.db")
    connection.execute(
        "CREATE TABLE activations (license_id TEXT PRIMARY KEY, machine TEXT NOT NULL, name TEXT NOT NULL)"
    )
    first, body = activate(connection, token, "pc-1")
    second, other = activate(connection, token, "pc-1")
    blocked, _payload = activate(connection, token, "pc-2")
    assert first == second == 200
    assert body["name"] == "Ana"
    assert blocked == 409
    assert other["ok"] is True


def test_a_dead_server_does_not_count_as_a_valid_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    private, public = _pair(tmp_path)
    token = issue_token("Ana", private)

    def fail(*_args, **_kwargs):
        raise httpx.ConnectError("sem rede")

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(LicenseError) as error:
        ActivationLedger(
            tmp_path / "license.json",
            server_url="http://127.0.0.1:9",
            machine="pc-1",
            public_key=load_public_key(public),
        ).submit(token)
    assert error.value.code == "server"
    assert not (tmp_path / "license.json").exists()
