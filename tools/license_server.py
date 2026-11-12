"""Servidor de ativação: a mesma chave só abre um computador.

A chave privada de emissão não fica aqui. Este processo só confere a
assinatura com a chave pública e grava qual computador já usou a licença.

    python -m tools.license_server --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from services.licensing import LicenseError, verify_token

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "secrets" / "activations.db"


def activate(connection: sqlite3.Connection, token: str, machine: str) -> tuple[int, dict]:
    try:
        license_info = verify_token(token)
    except LicenseError:
        return 401, {"error": "invalid"}
    if not machine.strip():
        return 401, {"error": "invalid"}
    row = connection.execute(
        "SELECT machine FROM activations WHERE license_id = ?",
        (license_info.license_id,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO activations (license_id, machine, name) VALUES (?, ?, ?)",
            (license_info.license_id, machine, license_info.name),
        )
        connection.commit()
        return 200, {"ok": True, "name": license_info.name, "license_id": license_info.license_id}
    if row[0] != machine:
        return 409, {"error": "used"}
    return 200, {"ok": True, "name": license_info.name, "license_id": license_info.license_id}


def serve(host: str, port: int, database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS activations (
            license_id TEXT PRIMARY KEY,
            machine TEXT NOT NULL,
            name TEXT NOT NULL
        )
        """
    )
    connection.commit()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/activate":
                self._send(404, {"error": "missing"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(401, {"error": "invalid"})
                return
            status, payload = activate(connection, str(body.get("token") or ""), str(body.get("machine") or ""))
            self._send(status, payload)

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ativa chaves do Loon Translator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    serve(args.host, args.port, args.db)


if __name__ == "__main__":
    main()
