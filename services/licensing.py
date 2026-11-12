"""Licença assinada: uma chave por venda, um computador por chave.

A chave privada que emite licenças fica fora do programa, em secrets/.
O aplicativo só carrega a chave pública. Uma chave inventada ou de outro
produto não passa da verificação. O servidor de ativação recusa a mesma
chave num segundo computador.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

from core.config import APP_DIR
from services.platform_host import machine_identity

PRODUCT = "loon-translator"
_ACTIVATION = APP_DIR / "license.json"


def public_key_path() -> Path:
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "services" / "license_public.pem"
    if bundled.is_file():
        return bundled
    return Path(__file__).with_name("license_public.pem")


class LicenseError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class License:
    license_id: str
    name: str
    token: str


def machine_fingerprint() -> str:
    """Código deste computador. Não é o nome da máquina e não volta a ser o GUID."""
    guid = machine_identity()
    return hashlib.sha256(f"{PRODUCT}\n{guid}".encode()).hexdigest()


def license_server_url() -> str:
    return os.getenv("LOON_LICENSE_URL", "").strip().rstrip("/")


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise LicenseError("invalid")
    return key


def load_public_key(path: Path | None = None) -> Ed25519PublicKey:
    key = load_pem_public_key((path or public_key_path()).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise LicenseError("invalid")
    return key


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def issue_token(name: str, private_key: Ed25519PrivateKey, license_id: str | None = None) -> str:
    cleaned = " ".join(name.split())
    if not cleaned:
        raise LicenseError("invalid")
    payload = {
        "v": 1,
        "product": PRODUCT,
        "lic": license_id or uuid.uuid4().hex,
        "name": cleaned,
        "seats": 1,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    signature = private_key.sign(body)
    return f"loon1.{_b64(body)}.{_b64(signature)}"


def verify_token(token: str, public_key: Ed25519PublicKey | None = None) -> License:
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != "loon1":
        raise LicenseError("invalid")
    try:
        body = _b64_decode(parts[1])
        signature = _b64_decode(parts[2])
        (public_key or load_public_key()).verify(signature, body)
        payload = json.loads(body)
    except (InvalidSignature, ValueError, json.JSONDecodeError, LicenseError):
        raise LicenseError("invalid") from None
    if payload.get("product") != PRODUCT or payload.get("seats") != 1:
        raise LicenseError("invalid")
    license_id = str(payload.get("lic") or "")
    name = str(payload.get("name") or "").strip()
    if not license_id or not name:
        raise LicenseError("invalid")
    return License(license_id=license_id, name=name, token=token.strip())


def export_public_pem(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


def export_private_pem(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())


class ActivationLedger:
    """Uma chave, um computador. A cópia local cobre o teste sem servidor."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        server_url: str = "",
        machine: str = "",
        public_key: Ed25519PublicKey | None = None,
    ) -> None:
        self.path = path or _ACTIVATION
        self.server_url = server_url.rstrip("/")
        self.machine = machine or machine_fingerprint()
        self.public_key = public_key

    def current(self) -> License | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            license_info = verify_token(str(raw.get("token") or ""), self.public_key)
        except (OSError, ValueError, json.JSONDecodeError, LicenseError):
            return None
        if raw.get("machine") != self.machine or raw.get("license_id") != license_info.license_id:
            return None
        return license_info

    def submit(self, token: str) -> License:
        license_info = verify_token(token, self.public_key)
        if self.server_url:
            self._activate_remote(license_info)
        else:
            self._activate_local(license_info)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "token": license_info.token,
                    "license_id": license_info.license_id,
                    "name": license_info.name,
                    "machine": self.machine,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return license_info

    def _activate_local(self, license_info: License) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if raw.get("license_id") == license_info.license_id and raw.get("machine") not in {self.machine, None, ""}:
            raise LicenseError("used")

    def _activate_remote(self, license_info: License) -> None:
        try:
            response = httpx.post(
                f"{self.server_url}/v1/activate",
                json={"token": license_info.token, "machine": self.machine},
                timeout=8,
            )
        except httpx.HTTPError as exc:
            raise LicenseError("server") from exc
        if response.status_code == 409:
            raise LicenseError("used")
        if response.status_code == 401:
            raise LicenseError("invalid")
        if response.status_code != 200:
            raise LicenseError("server")


def activated_license(path: Path | None = None) -> License | None:
    return ActivationLedger(path or _ACTIVATION, server_url=license_server_url()).current()


def licensee_name(path: Path | None = None) -> str:
    license_info = activated_license(path)
    return license_info.name if license_info else ""
