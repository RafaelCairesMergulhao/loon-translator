"""Emite uma chave de venda. A chave privada não entra no instalador.

Exemplo:
    python -m tools.issue_license --name "Maria Souza"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from services.licensing import export_private_pem, export_public_pem, issue_token, load_private_key

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "secrets" / "license_ed25519.pem"
PUBLIC = ROOT / "services" / "license_public.pem"


def ensure_keys() -> Ed25519PrivateKey:
    if PRIVATE.exists():
        return load_private_key(PRIVATE)
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    PRIVATE.write_bytes(export_private_pem(private))
    PUBLIC.write_bytes(export_public_pem(private))
    return private


def main() -> None:
    parser = argparse.ArgumentParser(description="Emite uma chave única do Loon Translator")
    parser.add_argument("--name", required=True, help="Nome de quem comprou")
    args = parser.parse_args()
    token = issue_token(args.name, ensure_keys())
    print(token)


if __name__ == "__main__":
    main()
