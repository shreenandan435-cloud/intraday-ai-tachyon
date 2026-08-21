"""Append-only JSONL journal — CLAUDE.md §6.4, §9.

``data/journal/<prefix>_<date>.jsonl`` is the post-mortem record of record. One class serves
every journal in the system (orders, sentinel prompts and responses, and whatever Phase 10
adds), because a second implementation would eventually disagree with the first about
redaction — and the one that drifted would be the one that leaked a credential to disk.

Two properties are load-bearing:

**Written before the system acts.** If the process dies between sending a request and recording
its outcome, the journal still shows the attempt. A record written afterwards is silent about
exactly the case it exists to survive.

**A write failure never aborts the caller.** A full disk must not become "we cannot cancel this
order". The failure is logged at ``CRITICAL`` and the operation proceeds, because being unable
to record what we did is strictly better than being unable to act.

Payloads are scrubbed through :func:`tachyon.core.logger.scrub_secrets` — the same rules the
log sink uses — so no API key or JWT reaches a file that outlives the session.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Any, Final

import msgspec

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist, today_ist
from tachyon.core.constants import JOURNAL_DIR
from tachyon.core.logger import get_logger, scrub_secrets

_log = get_logger(__name__)

_ENCODER: Final[msgspec.json.Encoder] = msgspec.json.Encoder()

#: Journal entry kinds. Requests and responses are separate records rather than one paired
#: object, because the failure mode worth surviving is precisely the one where the response
#: never arrives.
KIND_REQUEST: Final[str] = "REQUEST"
KIND_RESPONSE: Final[str] = "RESPONSE"
KIND_ERROR: Final[str] = "ERROR"
KIND_DECISION: Final[str] = "DECISION"


class JsonlJournal:
    """Append-only JSONL record of everything a subsystem did and was told.

    Args:
        directory: journal root. Defaults to ``data/journal``.
        prefix: filename stem — ``orders``, ``sentinel``, ...
        clock: injected for testing; decides which dated file is written.
        enabled: set False only in unit tests that assert on the caller, never in production.

    Thread-safe: the square-off watchdog thread and the strategy loop both write.
    """

    __slots__ = ("_clock", "_directory", "_enabled", "_failures", "_lock", "_prefix", "_written")

    def __init__(
        self,
        directory: Path = JOURNAL_DIR,
        *,
        prefix: str = "journal",
        clock: Clock = SYSTEM_CLOCK,
        enabled: bool = True,
    ) -> None:
        self._directory = directory
        self._prefix = prefix
        self._clock = clock
        self._enabled = enabled
        self._lock = threading.Lock()
        self._written = 0
        self._failures = 0

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def records_written(self) -> int:
        return self._written

    @property
    def write_failures(self) -> int:
        """Records that could not be persisted. Non-zero means the audit trail has holes."""
        return self._failures

    def path_for(self, day: date | None = None) -> Path:
        """Journal file for ``day`` (default: today in IST)."""
        session_date = day if day is not None else today_ist(self._clock)
        return self._directory / f"{self._prefix}_{session_date:%Y-%m-%d}.jsonl"

    # ── writing ──────────────────────────────────────────────────────────────

    def record(self, kind: str, event: str, **fields: Any) -> None:
        """Append one record. Never raises.

        Args:
            kind: one of :data:`KIND_REQUEST`, :data:`KIND_RESPONSE`, :data:`KIND_ERROR`,
                :data:`KIND_DECISION`.
            event: short machine-readable name, e.g. ``"place_order"``.
            **fields: arbitrary payload. Scrubbed of credentials before encoding.
        """
        if not self._enabled:
            return

        payload: dict[str, Any] = {
            "ts_ist": now_ist(self._clock).isoformat(timespec="milliseconds"),
            "kind": kind,
            "event": event,
            **fields,
        }

        try:
            scrubbed = scrub_secrets(payload)
            line = _ENCODER.encode(scrubbed) + b"\n"
        except Exception as exc:  # noqa: BLE001 - an unencodable field must not stop the caller
            self._failures += 1
            _log.error(
                "journal.encode_failed",
                journal_event=event,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return

        with self._lock:
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                # Opened per record rather than held open: an append to a closed-and-reopened
                # file is atomic enough for our rate (tens of records a day), and it means an
                # operator can rotate or copy the journal mid-session without breaking us.
                with self.path_for().open("ab") as handle:
                    handle.write(line)
                    handle.flush()
                self._written += 1
            except OSError as exc:
                self._failures += 1
                _log.critical(
                    "journal.write_failed",
                    journal_event=event,
                    path=str(self.path_for()),
                    error=str(exc),
                    impact="operation proceeding UNJOURNALLED — the audit trail has a hole",
                )

    # ── convenience wrappers ─────────────────────────────────────────────────

    def request(self, endpoint: str, payload: dict[str, Any], *, order_tag: str = "") -> None:
        """Record an outbound call **before** it is made."""
        self.record(KIND_REQUEST, endpoint, order_tag=order_tag, payload=payload)

    def response(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        order_tag: str = "",
        latency_ms: float | None = None,
    ) -> None:
        """Record a response before acting on it."""
        self.record(
            KIND_RESPONSE,
            endpoint,
            order_tag=order_tag,
            latency_ms=latency_ms,
            payload=payload,
        )

    def error(self, endpoint: str, error: str, *, order_tag: str = "", **fields: Any) -> None:
        """Record a failure — including the ones whose outcome we could not determine."""
        self.record(KIND_ERROR, endpoint, order_tag=order_tag, error=error, **fields)

    def decision(self, event: str, **fields: Any) -> None:
        """Record a local decision (a rejection, a veto, a square-off) that placed no call."""
        self.record(KIND_DECISION, event, **fields)

    # ── reading ──────────────────────────────────────────────────────────────

    def read(self, day: date | None = None) -> tuple[dict[str, Any], ...]:
        """Parse a day's journal. For reconciliation, tests and post-mortems.

        Malformed lines are skipped rather than raising: a truncated final record from a crash
        is expected, and it must not make the rest of the day unreadable.
        """
        path = self.path_for(day)
        if not path.is_file():
            return ()
        records: list[dict[str, Any]] = []
        try:
            raw = path.read_bytes()
        except OSError as exc:
            _log.error("journal.read_failed", path=str(path), error=str(exc))
            return ()
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                decoded = msgspec.json.decode(line)
            except msgspec.DecodeError:
                continue
            if isinstance(decoded, dict):
                records.append(decoded)
        return tuple(records)
