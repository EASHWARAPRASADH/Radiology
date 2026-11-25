"""Append-only, hash-chained audit log (prototype stand-in for the ``audit_events`` table).

Each event stores the SHA-256 of its canonical payload and of the previous event, so any edit or
deletion breaks ``verify()``. Payloads are stored as hashes only by default (no PHI in the log).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    def __init__(self, path: str | Path, store_payload: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_payload = store_payload
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with open(self.path, "rb") as f:
            last = f.read().strip().split(b"\n")[-1]
        return json.loads(last)["hash"]

    def append(self, event: str, *, actor: str, subject: str, payload: dict | None = None) -> dict:
        with self._lock:
            rec = {
                "ts": datetime.now(UTC).isoformat(),
                "event": event,
                "actor": actor,
                "subject": subject,
                "payload_sha256": hashlib.sha256(_canon(payload or {}).encode()).hexdigest(),
                "prev": self._last_hash(),
            }
            if self.store_payload:
                rec["payload"] = payload
            rec["hash"] = hashlib.sha256(_canon({k: v for k, v in rec.items() if k != "hash"}).encode()).hexdigest()
            with open(self.path, "a") as f:
                f.write(_canon(rec) + "\n")
            return rec

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify(self) -> bool:
        prev = GENESIS
        for rec in self.events():
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec["prev"] != prev or hashlib.sha256(_canon(body).encode()).hexdigest() != rec["hash"]:
                return False
            prev = rec["hash"]
        return True
