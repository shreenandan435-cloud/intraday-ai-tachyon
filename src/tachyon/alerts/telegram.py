"""Telegram alert system for autonomous paper trading deployment.

Provides real-time formatted messages for:
- Engine boot / heartbeat
- Order execution (entry, exit, trail)
- Position closed / trailed
- Daily market close summary
- Critical errors / exceptions

All configuration via environment variables:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- TELEGRAM_ALERTS_ENABLED (default: true)
"""

from __future__ import annotations

import atexit
import contextlib
import html
import logging
import os
import queue
import threading
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings

if TYPE_CHECKING:
    from tachyon.core.config import Settings

_log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TELEGRAM_API_BASE: Final[str] = "https://api.telegram.org"
MAX_MESSAGE_CHARS: Final[int] = 4096
DEFAULT_QUEUE_SIZE: Final[int] = 256
SEND_TIMEOUT_SECONDS: Final[float] = 5.0
DRAIN_TIMEOUT_SECONDS: Final[float] = 5.0
MAX_SESSION_SENDS: Final[int] = 500

_RETRY_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
_SENTINEL: Final[object] = object()


class AlertType(StrEnum):
    """Alert type classification for routing and formatting."""

    BOOT = "BOOT"
    HEARTBEAT = "HEARTBEAT"
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    TRAIL = "TRAIL"
    STOP_LOSS = "STOP_LOSS"
    TARGET_HIT = "TARGET_HIT"
    SQUARE_OFF = "SQUARE_OFF"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    KILL_SWITCH = "KILL_SWITCH"
    ROGUE_FILL = "ROGUE_FILL"
    SESSION_LOCKED = "SESSION_LOCKED"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


# Icons for visual scanning on lock screen
_ICON: Final[dict[AlertType, str]] = {
    AlertType.BOOT: "🚀",
    AlertType.HEARTBEAT: "💓",
    AlertType.ENTRY: "📈",
    AlertType.EXIT: "📉",
    AlertType.TRAIL: "🔄",
    AlertType.STOP_LOSS: "🛑",
    AlertType.TARGET_HIT: "🎯",
    AlertType.SQUARE_OFF: "🔔",
    AlertType.DAILY_SUMMARY: "📊",
    AlertType.KILL_SWITCH: "🔴",
    AlertType.ROGUE_FILL: "☠️",
    AlertType.SESSION_LOCKED: "🔒",
    AlertType.ERROR: "🚨",
    AlertType.WARNING: "⚠️",
    AlertType.INFO: "ℹ️",
}

# Kill switch guidance by reason
_KILL_SWITCH_GUIDANCE: Final[dict[str, str]] = {
    "DAILY_LOSS_LIMIT_REALISED": (
        "Daily loss limit hit on booked P&L. No action needed — "
        "the day is over and a restart will not reopen it."
    ),
    "DAILY_LOSS_LIMIT_TOTAL": (
        "Daily loss limit hit including open P&L. No action needed — "
        "the day is over and a restart will not reopen it."
    ),
    "DAILY_DRAWDOWN_BREACHED": (
        "Drawdown limit hit on the watchdog tick. No action needed — "
        "the day is over and a restart will not reopen it."
    ),
    "ROGUE_FILL": (
        "A fill arrived for an order this system never placed. Something else is "
        "trading this account, or our record of what we placed is wrong. "
        "CHECK THE BROKER TERMINAL NOW — restarting fixes neither."
    ),
    "PNL_UNDEFINED": (
        "P&L became undefined (NaN/Inf), so the limit could not be evaluated. "
        "Treated as breached. Check the broker terminal and reconcile before restarting."
    ),
    "PNL_UNCOMPARABLE": (
        "P&L comparison failed, so the limit could not be evaluated. "
        "Treated as breached. Check the broker terminal and reconcile before restarting."
    ),
}


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AlertStats:
    """Alert counters for observability."""

    queued: int = 0
    sent: int = 0
    failed: int = 0
    dropped: int = 0
    truncated: int = 0
    suppressed: int = 0


@dataclass(slots=True)
class EntryAlert:
    """Order entry alert data."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    quantity: int
    entry_price: Decimal
    stop_loss: Decimal
    target: Decimal
    simulated: bool = True
    order_tag: str = ""


@dataclass(slots=True)
class ExitAlert:
    """Position exit alert data."""

    symbol: str
    realised_pnl: Decimal
    charges: Decimal
    was_stop_out: bool
    session_total: Decimal
    headroom: Decimal
    exit_reason: str = ""


@dataclass(slots=True)
class TrailAlert:
    """Trailing stop update alert data."""

    symbol: str
    new_stop: Decimal
    direction: Literal["LONG", "SHORT"]
    unrealised_pnl: Decimal


@dataclass(slots=True)
class DailySummaryAlert:
    """End-of-day summary alert data."""

    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    net_pnl: Decimal
    charges: Decimal
    win_rate: Decimal
    virtual_balance: Decimal
    max_drawdown: Decimal
    best_trade: Decimal
    worst_trade: Decimal


@dataclass(slots=True)
class ErrorAlert:
    """Error/exception alert data."""

    error_type: str
    message: str
    context: str = ""
    severity: Literal["ERROR", "WARNING", "CRITICAL"] = "ERROR"


# ── Core Alert Classes ───────────────────────────────────────────────────────


def _esc(value: object) -> str:
    """Escape for HTML parse_mode."""
    return html.escape(str(value))


def _money(value: Decimal) -> str:
    """Render rupees with explicit sign."""
    return f"{'+' if value >= 0 else '-'}Rs.{abs(value):,.2f}"


def _endpoint(token: str, method: str) -> str:
    """Build API URL — only place token is interpolated."""
    return f"{TELEGRAM_API_BASE}/bot{token}/{method}"


def _describe(response: httpx.Response) -> str:
    """Telegram's error text for logging."""
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase
    return str(body.get("description", response.reason_phrase)) if isinstance(body, dict) else ""


def _hint_for(response: httpx.Response) -> str:
    """Turn common failures into actionable instructions."""
    detail = _describe(response).lower()
    if "chat not found" in detail:
        return (
            "Open Telegram, message the bot once (press Start), then retry — a bot cannot "
            "initiate a chat, so TELEGRAM_CHAT_ID is unreachable until you do"
        )
    if "unauthorized" in detail or response.status_code == httpx.codes.UNAUTHORIZED:
        return "TELEGRAM_BOT_TOKEN is wrong or was revoked — reissue it via @BotFather"
    if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.NOT_FOUND):
        return "check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
    return ""


class TelegramAlerter:
    """Thread-safe, non-blocking Telegram alert delivery.

    - Enqueues alerts on caller thread (event loop, webhook, watchdog)
    - Daemon thread drains queue and POSTs to Telegram
    - Never blocks trading path; failures are logged and swallowed
    - Session send ceiling prevents runaway alert loops
    """

    __slots__ = (
        "_token",
        "_chat_id",
        "_enabled",
        "_max_sends",
        "_queue",
        "_lock",
        "_stopping",
        "_thread",
        "_client",
        "_transport",
        "_clock",
        "_started",
        "stats",
    )

    def __init__(
        self,
        *,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = True,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_sends: int = MAX_SESSION_SENDS,
        transport: httpx.BaseTransport | None = None,
        clock: Clock = SYSTEM_CLOCK,
        start: bool = True,
    ) -> None:
        self._token = bot_token.strip()
        self._chat_id = chat_id.strip()
        self._enabled = enabled and bool(self._token) and bool(self._chat_id)
        self._max_sends = max_sends
        self._transport = transport
        self._clock = clock
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None
        self._started = False
        self.stats = AlertStats()

        if not self._enabled:
            _log.info(
                "telegram.disabled",
                extra={
                    "configured": bool(self._token and self._chat_id),
                    "impact": "no operator alerts this session; trading is unaffected",
                },
            )
        elif start:
            self.start()

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> TelegramAlerter:
        """Build from resolved config. Credentials come from .env only."""
        return cls(
            bot_token=settings.telegram_bot_token.get_secret_value(),
            chat_id=settings.telegram_chat_id,
            enabled=settings.telegram_alerts_enabled,
            **kwargs,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the delivery daemon thread. Idempotent."""
        if self._started or not self._enabled:
            self._started = True
            return
        self._thread = threading.Thread(target=self._run, name="telegram-alerts", daemon=True)
        self._thread.start()
        self._started = True
        atexit.register(self.close)

    def close(self) -> None:
        """Drain and stop. Idempotent, never raises."""
        if self._stopping.is_set():
            return
        self._stopping.set()

        thread = self._thread
        if thread is not None:
            self._queue.put(_SENTINEL)
            thread.join(timeout=DRAIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                _log.warning(
                    "telegram.sender_stuck",
                    extra={
                        "seconds": DRAIN_TIMEOUT_SECONDS,
                        "queued": self._queue.qsize(),
                        "impact": "queued alerts were not delivered",
                    },
                )
            self._thread = None

        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

    # ── Public Alert Methods ───────────────────────────────────────────────────

    def send(self, message: str, kind: AlertType = AlertType.INFO) -> bool:
        """Queue one alert. Returns True if accepted for delivery."""
        if not self._enabled:
            self.stats.suppressed += 1
            return False

        text = message.strip()
        if not text:
            return False

        with self._lock:
            if self.stats.queued >= self._max_sends:
                self.stats.suppressed += 1
                if self.stats.queued == self._max_sends:
                    self.stats.queued += 1
                    _log.warning(
                        "telegram.session_ceiling_reached",
                        extra={
                            "limit": self._max_sends,
                            "impact": "alerts disabled for the session; trading is unaffected",
                        },
                    )
                return False
            self.stats.queued += 1

        stamped = (
            f"{_ICON.get(kind, '')} <b>{kind.value}</b> {now_ist(self._clock):%H:%M:%S}\n{text}"
        )
        if len(stamped) > MAX_MESSAGE_CHARS:
            stamped = stamped[:MAX_MESSAGE_CHARS - 1] + "…"
            self.stats.truncated += 1

        try:
            self._queue.put_nowait(stamped)
        except queue.Full:
            self.stats.dropped += 1
            _log.warning(
                "telegram.queue_full",
                extra={
                    "depth": self._queue.qsize(),
                    "kind": kind.value,
                    "impact": (
                        "this alert was dropped; a backlog this deep means something is looping"
                    ),
                },
            )
            return False
        return True

    # ── Structured Alert Helpers ───────────────────────────────────────────────

    def alert_boot(
        self,
        mode: str,
        budget: Decimal,
        symbols: list[str],
        capital: Decimal,
    ) -> bool:
        """Engine boot alert with config summary."""
        symbols_str = ", ".join(symbols) if symbols else "none"
        return self.send(
            f"Mode: <b>{_esc(mode)}</b>\n"
            f"Virtual Capital: <b>{_money(capital)}</b>\n"
            f"Budget: <b>Rs.{budget:,.2f}</b>\n"
            f"Symbols: <b>{_esc(symbols_str)}</b>",
            AlertType.BOOT,
        )

    def alert_heartbeat(self, mode: str, state: str, open_positions: int) -> bool:
        """Periodic heartbeat with session state."""
        return self.send(
            f"Mode: <b>{_esc(mode)}</b>  State: <b>{_esc(state)}</b>\n"
            f"Open positions: <b>{open_positions}</b>",
            AlertType.HEARTBEAT,
        )

    def alert_entry(self, data: EntryAlert) -> bool:
        """Order entry filled."""
        mode = " [PAPER]" if data.simulated else ""
        return self.send(
            f"<b>{_esc(data.symbol)}</b> {_esc(data.direction)} x{data.quantity}{mode}\n"
            f"Entry: <b>Rs.{data.entry_price:,.2f}</b>\n"
            f"Stop: <b>Rs.{data.stop_loss:,.2f}</b>  Target: <b>Rs.{data.target:,.2f}</b>"
            + (f"\nTag: {_esc(data.order_tag)}" if data.order_tag else ""),
            AlertType.ENTRY,
        )

    def alert_exit(self, data: ExitAlert) -> bool:
        """Position closed (stop loss, target, or square-off)."""
        kind = AlertType.STOP_LOSS if data.was_stop_out else AlertType.TARGET_HIT
        if "square_off" in data.exit_reason.lower():
            kind = AlertType.SQUARE_OFF

        gross = data.realised_pnl
        net = gross - data.charges

        return self.send(
            f"<b>{_esc(data.symbol)}</b> closed\n"
            f"Net: <b>{_money(net)}</b>  (gross {_money(gross)}  charges Rs.{data.charges:,.2f})\n"
            f"Session: <b>{_money(data.session_total)}</b>  Headroom: Rs.{data.headroom:,.2f}"
            + (f"\nReason: {_esc(data.exit_reason)}" if data.exit_reason else ""),
            kind,
        )

    def alert_trail(self, data: TrailAlert) -> bool:
        """Trailing stop updated."""
        return self.send(
            f"<b>{_esc(data.symbol)}</b> {_esc(data.direction)} trail update\n"
            f"New stop: <b>Rs.{data.new_stop:,.2f}</b>\n"
            f"Unrealised: {_money(data.unrealised_pnl)}",
            AlertType.TRAIL,
        )

    def alert_daily_summary(self, data: DailySummaryAlert) -> bool:
        """End-of-day summary."""
        win_rate_pct = data.win_rate * 100
        return self.send(
            f"📊 <b>DAILY SUMMARY — {_esc(data.date)}</b>\n"
            f"Total trades: <b>{data.total_trades}</b>  "
            f"Wins: <b>{data.winning_trades}</b>  Losses: <b>{data.losing_trades}</b>\n"
            f"Win rate: <b>{win_rate_pct:.1f}%</b>\n"
            f"Net P&L: <b>{_money(data.net_pnl)}</b>  Charges: Rs.{data.charges:,.2f}\n"
            f"Virtual balance: <b>{_money(data.virtual_balance)}</b>\n"
            f"Max drawdown: <b>{_money(data.max_drawdown)}</b>\n"
            f"Best: {_money(data.best_trade)}  Worst: {_money(data.worst_trade)}",
            AlertType.DAILY_SUMMARY,
        )

    def alert_kill_switch(
        self,
        reason: str,
        detail: str,
        realised: Decimal,
        total: Decimal,
        limit: Decimal,
        lock_durable: bool,
    ) -> bool:
        """Session kill switch triggered — most critical alert."""
        guidance = _KILL_SWITCH_GUIDANCE.get(
            reason, "Session locked. Check the broker terminal before restarting."
        )
        warning = (
            ""
            if lock_durable
            else "\n\n⚠ THE ON-DISK LOCK COULD NOT BE WRITTEN. This session is locked "
            "but a RESTART WOULD NOT BE. Do not restart today."
        )
        return self.send(
            f"<b>TRADING HALTED</b> — {_esc(reason)}\n"
            f"Session: <b>{_money(total)}</b> (booked {_money(realised)})"
            f" vs limit Rs.{limit:,.2f}\n"
            f"{_esc(detail)}\n\n{guidance}{warning}",
            AlertType.KILL_SWITCH,
        )

    def alert_rogue_fill(
        self,
        symbol: str,
        order_id: str,
        quantity: int,
        side: str,
        price: Decimal,
    ) -> bool:
        """Fill for unknown order — potential account compromise."""
        return self.send(
            f"<b>{_esc(symbol)}</b> — fill for an order we never placed\n"
            f"{_esc(side)} {quantity} @ Rs.{price:,.2f}  order {_esc(order_id)}\n\n"
            f"NOT booked into P&L. Session locked on disk, so a restart stays locked.\n"
            f"Something else is trading this account, or our record of what we placed is "
            f"wrong. <b>OPEN THE BROKER TERMINAL NOW.</b>",
            AlertType.ROGUE_FILL,
        )

    def alert_session_locked(self, reason: str) -> bool:
        """Non-P&L session lock (unknown entry, math failure, etc.)."""
        return self.send(
            f"<b>TRADING HALTED</b> — {_esc(reason)}\n\n"
            f"No new entries this session. This is not a P&L breach — check the broker "
            f"terminal for orders or positions this process may not know about.",
            AlertType.SESSION_LOCKED,
        )

    def alert_square_off_started(self, open_positions: int) -> bool:
        """15:15 auto square-off initiated."""
        return self.send(
            f"15:15 IST auto square-off firing.\n{open_positions} position(s) to flatten.",
            AlertType.SQUARE_OFF,
        )

    def alert_square_off_completed(self, session_total: Decimal) -> bool:
        """Square-off confirmed flat."""
        return self.send(
            f"Square-off complete — account is flat.\nSession P&L {_money(session_total)}",
            AlertType.SQUARE_OFF,
        )

    def alert_square_off_failed(self, detail: str) -> bool:
        """Square-off did not confirm flat; watchdog retrying."""
        return self.send(
            f"SQUARE-OFF DID NOT CONFIRM FLAT — retrying.\n{_esc(detail)}\n"
            f"Check the broker terminal now.",
            AlertType.ERROR,
        )

    def alert_error(self, data: ErrorAlert) -> bool:
        """Generic error/exception alert."""
        kind = {
            "ERROR": AlertType.ERROR,
            "WARNING": AlertType.WARNING,
            "CRITICAL": AlertType.ERROR,
        }.get(data.severity.upper(), AlertType.ERROR)

        return self.send(
            f"<b>{_esc(data.error_type)}</b>\n{_esc(data.message)}"
            + (f"\nContext: {_esc(data.context)}" if data.context else ""),
            kind,
        )

    # ── Internal Delivery ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Drain queue until stopped. Absorbs everything."""
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                if isinstance(item, str):
                    self._deliver(item)
            except Exception as exc:  # noqa: BLE001
                self.stats.failed += 1
                _log.error(
                    "telegram.sender_error",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
            finally:
                self._queue.task_done()

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=SEND_TIMEOUT_SECONDS, transport=self._transport)
        return self._client

    def _deliver(self, text: str) -> None:
        """POST one message, retry once on 429/5xx."""
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        url = _endpoint(self._token, "sendMessage")

        for attempt in (1, 2):
            try:
                response = self._ensure_client().post(url, json=payload)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    self.stats.failed += 1
                    _log.warning(
                        "telegram.send_failed",
                        extra={"error_type": type(exc).__name__},
                    )
                    return
                continue

            if response.status_code == httpx.codes.OK:
                self.stats.sent += 1
                return
            if response.status_code in _RETRY_STATUS and attempt == 1:
                continue

            self.stats.failed += 1
            _log.warning(
                "telegram.send_rejected",
                extra={
                    "status": response.status_code,
                    "description": _describe(response),
                    "hint": _hint_for(response),
                },
            )
            return

    def drain_for_test(self) -> int:
        """Synchronous drain for tests only."""
        delivered = 0
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, str):
                self._deliver(item)
                delivered += 1
            self._queue.task_done()
        return delivered


# ── Convenience Functions ────────────────────────────────────────────────────

def create_alerter_from_env(clock: Clock = SYSTEM_CLOCK) -> TelegramAlerter:
    """Create alerter from environment variables."""
    return TelegramAlerter(
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        enabled=os.environ.get("TELEGRAM_ALERTS_ENABLED", "true").lower() == "true",
        clock=clock,
    )


def send_telegram_alert(message: str, kind: AlertType = AlertType.INFO) -> bool:
    """Module-level convenience for simple alerts."""
    try:
        return create_alerter_from_env().send(message, kind)
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "telegram.alert_failed",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        return False
