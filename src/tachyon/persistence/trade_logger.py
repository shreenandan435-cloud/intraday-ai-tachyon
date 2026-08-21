"""Trade persistence — the operator-facing record of what actually happened.

``data/journal/`` already holds the machine record: every broker request and response, written
before we act on it (CLAUDE.md §6.4). This module writes the *human* record beside it —
``data/trades/trades_<date>.csv``, one row per fill, openable in a spreadsheet without a JSON
parser — plus ``vetoes_<date>.csv`` and an end-of-session ``summary_<date>.json``.

Vetoes go in their own file deliberately. A veto is not a trade, and a trades file containing
non-trades makes every win-rate and P&L sum computed over it wrong. Two files that each mean
one thing beat one file that means two.

Non-blocking by construction
----------------------------
Callers hand a row to a bounded queue and return; a single daemon thread does the open, write
and flush. Nothing on the tick path, the fill path or the event loop ever waits on a disk.

Two failure rules follow ``persistence.journal``, for the same reason:

* **A write failure never reaches the caller.** It is logged at ``CRITICAL`` and dropped. A full
  disk must not become "we cannot record this fill", and it certainly must not become an
  exception on the path that books P&L against the ₹500 limit.
* **A full queue drops the row rather than blocking.** Blocking would couple disk latency to
  the trading loop, which is the one thing this module exists to avoid. The queue holds
  :data:`DEFAULT_QUEUE_SIZE` rows against a session that realistically produces tens, so a full
  queue means the writer thread is wedged — logged loudly, never waited on.

The writer is a **daemon** thread, unlike the square-off watchdog. The watchdog is non-daemon
because it must outlive shutdown to flatten; this one must not, because a non-daemon writer
that nobody closed would hang the process forever — and CPython joins non-daemon threads
*before* running ``atexit`` handlers, so the usual "flush at exit" safety net cannot fire.
:meth:`TradeLogger.close` drains and joins explicitly, and is registered with ``atexit`` as the
backstop for a path that forgets.

``realized_pnl`` is **net of estimated charges**, matching the number §1.2 enforces the daily
limit against. On a ₹500 budget a round trip costs ₹10–40, so a gross figure here would read
consistently better than the session actually was. The summary carries gross, charges and net
separately.
"""

from __future__ import annotations

import atexit
import csv
import json
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import DATA_DIR
from tachyon.core.logger import get_logger

if TYPE_CHECKING:
    from tachyon.risk.engine import RiskDecision
    from tachyon.ui.postback import OrderUpdate

_log = get_logger(__name__)

TRADES_DIR: Final[Path] = DATA_DIR / "trades"

#: Columns of ``trades_<date>.csv``, in order. Append only — a reordering silently corrupts
#: every file already on disk that a later reader assumes it understands.
TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp_ist",
    "symbol",
    "side",
    "quantity",
    "fill_price",
    "order_id",
    "latency_ms",
    "trigger_reason",
    "realized_pnl",
)

VETO_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp_ist",
    "symbol",
    "reason",
    "detail",
    "failed_check",
)

TRADES_PREFIX: Final[str] = "trades"
VETOES_PREFIX: Final[str] = "vetoes"
SUMMARY_PREFIX: Final[str] = "summary"

#: Deep enough that filling it means the writer is wedged, not that the session was busy.
DEFAULT_QUEUE_SIZE: Final[int] = 10_000

#: How long :meth:`TradeLogger.close` waits for the queue to drain before giving up. Bounded so
#: a wedged writer cannot hold up the 15:15 shutdown.
DRAIN_TIMEOUT_SECONDS: Final[float] = 5.0

#: Cap on remembered order intents. Bounds memory on a long session; evicts oldest first.
MAX_TRACKED_ORDERS: Final[int] = 5_000

# ── trigger reasons ──────────────────────────────────────────────────────────
#: Why an order existed. Entries carry the signal that produced them; exits carry the thing
#: that closed them. Free-form strings are accepted — these are the ones the system emits.
TRIGGER_VWAP_CONFLUENCE: Final[str] = "VWAP_CONFLUENCE"
TRIGGER_RISK_STOP: Final[str] = "RISK_STOP"
TRIGGER_TARGET: Final[str] = "TARGET"
TRIGGER_SENTINEL_EXIT: Final[str] = "SENTINEL_EXIT"
TRIGGER_SQUARE_OFF: Final[str] = "SQUARE_OFF"
TRIGGER_UNKNOWN: Final[str] = "UNKNOWN"

_SENTINEL: Final[object] = object()


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    """One row bound for one dated CSV.

    The date is resolved when the row is *created*, not when it is written: a row generated at
    15:14:59 must not land in tomorrow's file because the writer thread was briefly behind.
    """

    prefix: str
    columns: tuple[str, ...]
    row: dict[str, str]
    on_date: date


@dataclass(slots=True)
class _Intent:
    """What we knew when the order went out, kept until its fill arrives."""

    placed_monotonic: float
    trigger_reason: str


@dataclass(slots=True)
class SessionSummary:
    """Running session aggregates. Closed positions only — an open one has realised nothing."""

    total_fills: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    stop_outs: int = 0
    gross_pnl: Decimal = Decimal("0")
    total_charges: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    best_trade: Decimal | None = None
    worst_trade: Decimal | None = None
    vetoes: int = 0
    rows_dropped: int = 0
    write_failures: int = 0
    peak_equity: Decimal = field(default=Decimal("0"), repr=False)

    def book(self, realised: Decimal, charges: Decimal, *, was_stop_out: bool) -> Decimal:
        """Fold one closed trade in and return its net. Drawdown is tracked on the net equity
        curve, peak to trough, in the order trades actually closed."""
        net = realised - charges
        self.total_trades += 1
        self.gross_pnl += realised
        self.total_charges += charges
        self.net_pnl += net
        if was_stop_out:
            self.stop_outs += 1

        if net > 0:
            self.wins += 1
        elif net < 0:
            self.losses += 1
        else:
            self.scratches += 1

        self.best_trade = net if self.best_trade is None else max(self.best_trade, net)
        self.worst_trade = net if self.worst_trade is None else min(self.worst_trade, net)

        self.peak_equity = max(self.peak_equity, self.net_pnl)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.net_pnl)
        return net

    @property
    def win_rate(self) -> float:
        """Wins as a fraction of *decided* trades. Scratches are excluded from the denominator
        rather than counted as losses — a flat is not a loss, and folding it in understates a
        strategy that is genuinely break-even."""
        decided = self.wins + self.losses
        return round(self.wins / decided, 4) if decided else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "total_fills": self.total_fills,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "scratches": self.scratches,
            "stop_outs": self.stop_outs,
            "win_rate": self.win_rate,
            "gross_pnl": str(self.gross_pnl),
            "total_charges": str(self.total_charges),
            "net_pnl": str(self.net_pnl),
            "max_drawdown": str(self.max_drawdown),
            "best_trade": None if self.best_trade is None else str(self.best_trade),
            "worst_trade": None if self.worst_trade is None else str(self.worst_trade),
            "vetoes": self.vetoes,
            "rows_dropped": self.rows_dropped,
            "write_failures": self.write_failures,
        }


class TradeLogger:
    """Appends fills and vetoes to dated CSVs, and writes a session summary at shutdown.

    Args:
        directory: output root. Created on construction. ``None`` resolves :data:`TRADES_DIR`
            *at construction time* rather than at import, so a test fixture can redirect the
            module constant and no test can write into the operator's real ``data/trades``.
        clock: injected for tests; decides the IST timestamp and which dated file is written.
        queue_size: bounded write queue. A full queue drops rows, never blocks.
        enabled: set False in tests that assert on the caller rather than on disk.
        start: start the writer thread immediately. False keeps construction inert for tests
            that drive :meth:`drain_for_test` by hand.
    """

    __slots__ = (
        "_clock",
        "_directory",
        "_enabled",
        "_intents",
        "_lock",
        "_queue",
        "_started",
        "_stopping",
        "_thread",
        "summary",
    )

    def __init__(
        self,
        *,
        directory: Path | None = None,
        clock: Clock = SYSTEM_CLOCK,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        enabled: bool = True,
        start: bool = True,
    ) -> None:
        self._directory = directory if directory is not None else TRADES_DIR
        self._clock = clock
        self._enabled = enabled
        self._queue: queue.Queue[_PendingWrite | object] = queue.Queue(maxsize=queue_size)
        self._intents: OrderedDict[str, _Intent] = OrderedDict()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self.summary = SessionSummary()

        if self._enabled:
            self._ensure_directory()
        if start:
            self.start()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the writer thread. Idempotent."""
        if self._started or not self._enabled:
            self._started = True
            return
        self._thread = threading.Thread(target=self._run, name="trade-logger", daemon=True)
        self._thread.start()
        self._started = True
        atexit.register(self.close)

    def close(self, *, write_summary: bool = True) -> None:
        """Drain, write the session summary, and stop the writer. Idempotent and never raises.

        Called from the orchestrator's graceful shutdown and, as a backstop, from ``atexit``.
        """
        if self._stopping.is_set():
            return
        if write_summary and self._enabled:
            self.write_summary()

        self._stopping.set()
        thread = self._thread
        if thread is None:
            return
        self._queue.put(_SENTINEL)
        thread.join(timeout=DRAIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            _log.critical(
                "trade_logger.writer_stuck",
                seconds=DRAIN_TIMEOUT_SECONDS,
                queued=self._queue.qsize(),
                impact="trade rows still queued were not written to disk",
            )
        self._thread = None

    def _ensure_directory(self) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.critical(
                "trade_logger.directory_unavailable",
                directory=str(self._directory),
                error=str(exc),
                impact="no trade CSV will be written this session",
            )

    # ── recording ────────────────────────────────────────────────────────────

    def note_order_placed(self, order_id: str, *, trigger_reason: str = TRIGGER_UNKNOWN) -> None:
        """Remember why an order went out and when, so its fill can report latency.

        Timed on ``time.monotonic`` via the injected clock — a wall-clock delta would be
        corrupted by an NTP correction landing between placement and fill (CLAUDE.md §8).
        """
        if not order_id:
            return
        with self._lock:
            self._intents[order_id] = _Intent(
                placed_monotonic=self._clock.monotonic(), trigger_reason=trigger_reason
            )
            while len(self._intents) > MAX_TRACKED_ORDERS:
                self._intents.popitem(last=False)

    def record_fill(self, update: OrderUpdate, realized_pnl: Decimal | None = None) -> None:
        """Append one fill. ``realized_pnl`` is set only on the fill that flattens the symbol.

        Never raises: this runs on the fill path, and an exception here would propagate into
        the code that books P&L against the daily limit.
        """
        if not self._enabled:
            return
        try:
            intent = self._take_intent(update.order_id)
            row = {
                "timestamp_ist": now_ist(self._clock).isoformat(timespec="milliseconds"),
                "symbol": update.symbol,
                "side": update.side.strip().upper(),
                "quantity": str(update.filled_quantity),
                "fill_price": str(update.average_price),
                "order_id": update.order_id,
                "latency_ms": self._latency_ms(intent),
                "trigger_reason": self._trigger_for(update, intent),
                "realized_pnl": "" if realized_pnl is None else str(realized_pnl),
            }
            self.summary.total_fills += 1
            self._enqueue(_PendingWrite(TRADES_PREFIX, TRADE_COLUMNS, row, self._today()))
        except Exception as exc:  # noqa: BLE001 - recording must never break the fill path
            # Deliberately does not read anything off `update`: whatever broke above may well
            # be the update itself, and a handler that re-raises is not a handler.
            _log.error(
                "trade_logger.record_fill_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )

    def record_veto(self, decision: RiskDecision) -> None:
        """Append one risk veto. Allowed decisions are not recorded — they are not events."""
        if not self._enabled or decision.allowed:
            return
        try:
            row = {
                "timestamp_ist": decision.at_ist.isoformat(timespec="milliseconds"),
                "symbol": decision.symbol,
                "reason": str(decision.reason) if decision.reason is not None else "",
                "detail": decision.detail,
                "failed_check": decision.failed_check,
            }
            self.summary.vetoes += 1
            self._enqueue(_PendingWrite(VETOES_PREFIX, VETO_COLUMNS, row, self._today()))
        except Exception as exc:  # noqa: BLE001 - the gate must never fail because of logging
            _log.error("trade_logger.record_veto_failed", error=str(exc), exc_info=True)

    def record_close(
        self,
        symbol: str,
        realised: Decimal,
        charges: Decimal,
        was_stop_out: bool,
    ) -> None:
        """Fold one closed position into the session aggregates.

        The fill that closed it is written by :meth:`record_fill`; this only updates the
        numbers that reach the summary. Drawdown is tracked on the *net* equity curve, peak to
        trough, in the order trades closed.
        """
        if not self._enabled:
            return
        try:
            summary = self.summary
            net = summary.book(realised, charges, was_stop_out=was_stop_out)
            _log.info(
                "trade_logger.trade_closed",
                symbol=symbol,
                net_inr=str(net),
                session_net_inr=str(summary.net_pnl),
                max_drawdown_inr=str(summary.max_drawdown),
            )
        except Exception as exc:  # noqa: BLE001 - never break the close path
            _log.error("trade_logger.record_close_failed", symbol=symbol, error=str(exc))

    # ── summary ──────────────────────────────────────────────────────────────

    def summary_path(self, on_date: date | None = None) -> Path:
        return self._directory / f"{SUMMARY_PREFIX}_{(on_date or self._today()).isoformat()}.json"

    def write_summary(self) -> Path | None:
        """Write ``summary_<date>.json``. Returns the path, or None if it could not be written.

        Written synchronously rather than through the queue: this runs once, at shutdown, after
        the point where the writer thread may already have been asked to stop.
        """
        if not self._enabled:
            return None
        path = self.summary_path()
        payload: dict[str, object] = {
            "date": self._today().isoformat(),
            "generated_at_ist": now_ist(self._clock).isoformat(timespec="seconds"),
            "trades_csv": str(self._trade_path(self._today()).name),
            **self.summary.as_dict(),
        }
        try:
            self._ensure_directory()
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            _log.critical(
                "trade_logger.summary_write_failed",
                path=str(path),
                error=str(exc),
                impact="the session summary was not persisted; the trade CSV still is",
            )
            return None
        _log.info("trade_logger.summary_written", path=str(path), **self.summary.as_dict())
        return path

    # ── paths ────────────────────────────────────────────────────────────────

    def _today(self) -> date:
        return now_ist(self._clock).date()

    def _trade_path(self, on_date: date) -> Path:
        return self._path_for(TRADES_PREFIX, on_date)

    def _path_for(self, prefix: str, on_date: date) -> Path:
        return self._directory / f"{prefix}_{on_date.isoformat()}.csv"

    # ── internals ────────────────────────────────────────────────────────────

    def _take_intent(self, order_id: str) -> _Intent | None:
        """Pop what we recorded at placement. Popped, not read: a bracket leg fills once."""
        with self._lock:
            return self._intents.pop(order_id, None)

    def _latency_ms(self, intent: _Intent | None) -> str:
        if intent is None:
            # We never saw this order go out — a restart, or an exit placed by the broker's own
            # bracket. Blank is honest; zero would read as an instant fill.
            return ""
        elapsed = (self._clock.monotonic() - intent.placed_monotonic) * 1000.0
        return f"{max(elapsed, 0.0):.1f}"

    @staticmethod
    def _trigger_for(update: OrderUpdate, intent: _Intent | None) -> str:
        """A stop-out is identified by the order that closed the position, never by whether the
        trade lost money (CLAUDE.md §7.4)."""
        if update.is_stop_order:
            return TRIGGER_RISK_STOP
        if intent is not None:
            return intent.trigger_reason
        return TRIGGER_UNKNOWN

    def _enqueue(self, pending: _PendingWrite) -> None:
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            self.summary.rows_dropped += 1
            _log.critical(
                "trade_logger.queue_full",
                prefix=pending.prefix,
                dropped_total=self.summary.rows_dropped,
                impact="this row was not written; the writer thread is not draining",
            )

    def _run(self) -> None:
        """Writer thread. Absorbs everything — a logging thread that dies takes the record with
        it, and does so silently."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            if not isinstance(item, _PendingWrite):
                continue
            self._write(item)

    def _write(self, pending: _PendingWrite) -> None:
        path = self._path_for(pending.prefix, pending.on_date)
        try:
            is_new = not path.exists()
            if is_new:
                self._directory.mkdir(parents=True, exist_ok=True)
            # newline="" is required by the csv module on Windows; without it every row is
            # separated by a blank line and the file reads as half-empty in a spreadsheet.
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pending.columns))
                if is_new:
                    writer.writeheader()
                writer.writerow(pending.row)
                handle.flush()
        except (OSError, csv.Error, ValueError) as exc:
            self.summary.write_failures += 1
            _log.critical(
                "trade_logger.write_failed",
                path=str(path),
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this row is lost; trading is unaffected",
            )

    # ── testing ──────────────────────────────────────────────────────────────

    def drain_for_test(self) -> int:
        """Write everything queued, synchronously. Tests only — never call in production.

        Lets a test assert on file contents without sleeping on a background thread.
        """
        written = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return written
            if isinstance(item, _PendingWrite):
                self._write(item)
                written += 1


def session_summary_from_disk(path: Path) -> dict[str, object]:
    """Read a summary file back. Raises ``OSError``/``ValueError`` — this is an operator tool,
    not part of the trading path, so it reports failure rather than swallowing it."""
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
