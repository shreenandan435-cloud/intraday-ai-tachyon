"""Telegram operator alerts — advisory only, CLAUDE.md §5.1, §8, §9.2.

The operator is not watching the terminal at 15:15. This pushes the three events that
actually need a human to know about them — an entry filled, a stop-out, and the square-off —
to a phone.

Three rules govern this module, and all three are inherited rather than invented:

**It may only ever observe.** Like the Sentinel (§5.1), this is an *advisor*. It cannot
create a signal, size a position, or veto one. There is deliberately no inbound path: the bot
never reads updates, so a Telegram message can never reach the trading loop. A messaging
integration that could accept commands would be a second order path around §4's risk gate,
and §7.3 already refuses that for the UI, which at least runs on this machine.

**It may never block the trading path.** Callers enqueue onto a bounded queue and return; one
daemon thread does the HTTP. This is `persistence/trade_logger.py`'s shape (§9.2) for
`trade_logger`'s reason, and here the stakes are higher than a disk: the square-off hook is
called *from the watchdog thread* (§1.1), and a five-second socket timeout inside it would
delay the retry of the flatten. §2.2 also bans blocking calls in async paths, and the entry
hook runs on the Brain's event loop. Enqueueing costs a lock and a deque append from both.

**Its failure is silent.** A dropped alert must never become an exception on the path that
books P&L or flattens a position. Every failure is logged and swallowed; nothing here raises.

Not `requests`
--------------
The obvious implementation uses `requests`, and `requirements.txt` bans it by name for §2.2's
reason. `httpx` is already a dependency and its *sync* client is correct here, because this
thread has one job and blocking it is the design.

On the token in the URL
-----------------------
§5 requires the Gemini key in a header, never a query string, because query strings reach
proxy logs. Telegram gives no such choice — the token is a **path** segment
(`/bot<token>/sendMessage`) and there is no header form. So: the URL is built at send time,
never logged, never journaled, and never carried in an exception message. `_endpoint` is the
only function that formats it, and it is the only place the secret is interpolated.

Configuration
-------------
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` (§8 — secrets never in `settings.yaml`,
never in source). Absent either one, the alerter constructs successfully and does nothing:
the same shape as a Sentinel with no API key, because an operator who has not configured
alerts must still be able to trade.

Example::

    send_telegram_alert("TACHYON: manual test")

    alerts = TelegramAlerter.from_settings(settings)
    alerts.entry_placed(symbol="RELIANCE", direction="LONG", quantity=12, ...)
    alerts.close(...)
"""

from __future__ import annotations

import atexit
import queue
import threading
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import httpx

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.logger import get_logger

if TYPE_CHECKING:
    from tachyon.core.config import Settings

_log = get_logger(__name__)

#: Telegram Bot API root. The bot token is appended as a path segment by :func:`_endpoint`.
TELEGRAM_API_BASE: Final[str] = "https://api.telegram.org"

#: Telegram rejects anything longer with HTTP 400. Truncated rather than dropped: a clipped
#: alert still tells the operator a stop was hit.
MAX_MESSAGE_CHARS: Final[int] = 4096

#: Queue depth. Deliberately small — this carries a handful of messages a session, so a
#: backlog means something is looping, and dropping is the correct response to that.
DEFAULT_QUEUE_SIZE: Final[int] = 256

#: Per-request socket budget. Short because nothing waits on it and a slow Telegram must not
#: keep the drain thread from the next alert.
SEND_TIMEOUT_SECONDS: Final[float] = 5.0

#: How long :meth:`TelegramAlerter.close` waits for the queue to drain at shutdown.
DRAIN_TIMEOUT_SECONDS: Final[float] = 5.0

#: Session send ceiling. Mirrors §5's token budget: exceeding it disables *alerts*, never
#: trading. A bug that alerts in a loop should cost silence, not a rate-limited bot.
MAX_SESSION_SENDS: Final[int] = 500

#: Retried exactly once, and only these — §5's rule. A 400/401/404 is a bad token or a bad
#: chat id, which a retry cannot fix.
_RETRY_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

_SENTINEL: Final[object] = object()


class AlertKind(StrEnum):
    """What happened. Prefixed onto the message so a phone glance is enough."""

    ENTRY = "ENTRY"
    STOP_LOSS = "STOP-LOSS"
    TARGET = "TARGET"
    SQUARE_OFF = "SQUARE-OFF"
    KILL_SWITCH = "KILL SWITCH"
    ROGUE_FILL = "ROGUE FILL"
    LOCKED = "SESSION LOCKED"
    ERROR = "ERROR"
    INFO = "INFO"


#: Leading glyph per kind. Colour carries meaning here for the same reason it does in §7.2:
#: the operator is reading this on a lock screen, not studying it.
_ICON: Final[dict[AlertKind, str]] = {
    AlertKind.ENTRY: "\U0001f4c8",  # chart increasing
    AlertKind.STOP_LOSS: "\U0001f6d1",  # stop sign
    AlertKind.TARGET: "\U0001f3af",  # direct hit
    AlertKind.SQUARE_OFF: "\U0001f514",  # bell
    AlertKind.KILL_SWITCH: "\U0001f534",  # red circle
    AlertKind.ROGUE_FILL: "☠",  # skull and crossbones
    AlertKind.LOCKED: "\U0001f512",  # lock
    AlertKind.ERROR: "\U0001f6a8",  # rotating light
    AlertKind.INFO: "ℹ",  # information
}

#: What the operator should actually *do*, per trip reason. A kill-switch buzz that says only
#: "locked" leaves them opening the laptop to find out which failure it was — and these have
#: very different answers: a loss limit is the system working, a rogue fill means something
#: else is trading the account right now.
_KILL_SWITCH_GUIDANCE: Final[dict[str, str]] = {
    "DAILY_LOSS_LIMIT_REALISED": "Daily loss limit hit on booked P&amp;L. No action needed — "
    "the day is over and a restart will not reopen it.",
    "DAILY_LOSS_LIMIT_TOTAL": "Daily loss limit hit including open P&amp;L. No action needed — "
    "the day is over and a restart will not reopen it.",
    "DAILY_DRAWDOWN_BREACHED": "Drawdown limit hit on the watchdog tick. No action needed — "
    "the day is over and a restart will not reopen it.",
    "ROGUE_FILL": "A fill arrived for an order this system never placed. Something else is "
    "trading this account, or our record of what we placed is wrong. CHECK THE BROKER "
    "TERMINAL NOW — restarting fixes neither.",
    "PNL_UNDEFINED": "P&amp;L became undefined (NaN/Inf), so the limit could not be evaluated. "
    "Treated as breached. Check the broker terminal and reconcile before restarting.",
    "PNL_UNCOMPARABLE": "P&amp;L comparison failed, so the limit could not be evaluated. "
    "Treated as breached. Check the broker terminal and reconcile before restarting.",
}


@dataclass(slots=True)
class AlertStats:
    """Counters for observability. Never used for a trading decision."""

    queued: int = 0
    sent: int = 0
    failed: int = 0
    dropped: int = 0
    truncated: int = 0
    suppressed: int = 0
    """Alerts refused because the session ceiling was reached, or the alerter is disabled."""


def _endpoint(token: str, method: str) -> str:
    """Build the API URL. **The only place the bot token is interpolated.**

    Never log the result; it contains the credential (see the module docstring).
    """
    return f"{TELEGRAM_API_BASE}/bot{token}/{method}"


def _describe(response: httpx.Response) -> str:
    """Telegram's own error text, or the status phrase if the body is not JSON.

    Safe to log: the body carries a description, never the token — unlike the request URL.
    """
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase
    return str(body.get("description", response.reason_phrase)) if isinstance(body, dict) else ""


def _hint_for(response: httpx.Response) -> str:
    """Turn the common setup failures into the instruction that actually fixes them.

    ``chat not found`` is the dominant one and reads as a bad chat id, which it usually is
    not: **a bot cannot open a conversation.** Until someone sends the bot ``/start``, every
    send to that chat is refused however correct the id is.
    """
    detail = _describe(response).lower()
    if "chat not found" in detail:
        return (
            "open Telegram, message the bot once (press Start), then retry — a bot cannot "
            "initiate a chat, so TELEGRAM_CHAT_ID is unreachable until you do"
        )
    if "unauthorized" in detail or response.status_code == httpx.codes.UNAUTHORIZED:
        return "TELEGRAM_BOT_TOKEN is wrong or was revoked — reissue it via @BotFather"
    if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.NOT_FOUND):
        return "check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
    return ""


def _esc(value: object) -> str:
    """Escape a value for ``parse_mode=HTML``.

    Telegram rejects the **whole message** with a 400 when it hits a bare ``&``, ``<`` or
    ``>`` — so an unescaped broker symbol does not garble the alert, it deletes it. That
    matters most for the rogue-fill and kill-switch messages, which carry broker-supplied
    strings and are the two alerts that cannot afford to go missing.
    """
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _money(value: Decimal) -> str:
    """Render rupees with an explicit sign, so a loss is never mistaken for a gain."""
    return f"{'+' if value >= 0 else '-'}Rs.{abs(value):,.2f}"


class TelegramAlerter:
    """Queues operator alerts and delivers them on a daemon thread. Never raises.

    Args:
        bot_token: from ``TELEGRAM_BOT_TOKEN``. Empty disables delivery.
        chat_id: from ``TELEGRAM_CHAT_ID``. Empty disables delivery.
        enabled: master switch; False disables regardless of credentials.
        queue_size: bounded depth; a full queue drops the newest and logs.
        max_sends: session ceiling. Exceeding it disables alerts, not trading.
        transport: injected for tests, so the suite never reaches the network.
        clock: injected for tests.
        start: False constructs without a thread (tests use :meth:`drain_for_test`).
    """

    __slots__ = (
        "_chat_id",
        "_clock",
        "_client",
        "_enabled",
        "_lock",
        "_max_sends",
        "_queue",
        "_started",
        "_stopping",
        "_thread",
        "_token",
        "_transport",
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
                configured=bool(self._token and self._chat_id),
                impact="no operator alerts this session; trading is unaffected",
            )
        elif start:
            self.start()

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> TelegramAlerter:
        """Build from resolved config. Credentials come from ``.env`` only (§8)."""
        return cls(
            bot_token=settings.telegram_bot_token.get_secret_value(),
            chat_id=settings.telegram_chat_id,
            enabled=settings.telegram_alerts_enabled,
            **kwargs,
        )

    @property
    def enabled(self) -> bool:
        """True when a token and a chat id are configured and alerts are switched on."""
        return self._enabled

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the delivery thread. Idempotent.

        A **daemon** thread, unlike the square-off watchdog (§1.1). The watchdog is non-daemon
        because it must outlive shutdown to flatten; an undrained non-daemon sender would hang
        the process on exit, and no alert is worth that.
        """
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
                    seconds=DRAIN_TIMEOUT_SECONDS,
                    queued=self._queue.qsize(),
                    impact="queued alerts were not delivered",
                )
            self._thread = None

        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                _log.warning("telegram.client_close_failed", error=str(exc))

    # ── the public surface ───────────────────────────────────────────────────

    def send(self, message: str, kind: AlertKind = AlertKind.INFO) -> bool:
        """Queue one alert. Returns True if it was **accepted for delivery**, not delivered.

        Never blocks and never raises — safe to call from the event loop, from the postback
        webhook thread, and from the square-off watchdog thread.
        """
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
                    self.stats.queued += 1  # log the ceiling exactly once
                    _log.warning(
                        "telegram.session_ceiling_reached",
                        limit=self._max_sends,
                        impact="alerts disabled for the session; trading is unaffected",
                    )
                return False
            self.stats.queued += 1

        stamped = (
            f"{_ICON.get(kind, '')} <b>{kind.value}</b> {now_ist(self._clock):%H:%M:%S}\n{text}"
        )
        if len(stamped) > MAX_MESSAGE_CHARS:
            stamped = stamped[: MAX_MESSAGE_CHARS - 1] + "…"
            self.stats.truncated += 1

        try:
            self._queue.put_nowait(stamped)
        except queue.Full:
            self.stats.dropped += 1
            _log.warning(
                "telegram.queue_full",
                depth=self._queue.qsize(),
                kind=kind.value,
                impact="this alert was dropped; a backlog this deep means something is looping",
            )
            return False
        return True

    # ── the three events CLAUDE.md cares about ───────────────────────────────

    def entry_placed(
        self,
        *,
        symbol: str,
        direction: str,
        quantity: int,
        entry: Decimal,
        stop: Decimal,
        target: Decimal,
        simulated: bool,
    ) -> bool:
        """The bot opened a position (§6.1). Carries the geometry, not just the fact."""
        mode = " [PAPER]" if simulated else ""
        return self.send(
            f"<b>{_esc(symbol)}</b> {_esc(direction)} x{quantity}{mode}\n"
            f"entry Rs.{entry:,.2f}  stop Rs.{stop:,.2f}  target Rs.{target:,.2f}",
            AlertKind.ENTRY,
        )

    def position_closed(
        self,
        *,
        symbol: str,
        realised: Decimal,
        charges: Decimal,
        was_stop_out: bool,
        session_total: Decimal,
        headroom: Decimal,
    ) -> bool:
        """A position went flat (§7.4).

        ``was_stop_out`` comes from the *order that closed the position*, never from whether
        the trade lost money — §7.4 is explicit, and a T2 fill that nets negative on charges
        is not a stop-out.
        """
        return self.send(
            f"<b>{_esc(symbol)}</b> closed  net {_money(realised - charges)}\n"
            f"gross {_money(realised)}  charges Rs.{charges:,.2f}\n"
            f"session {_money(session_total)}  headroom Rs.{headroom:,.2f}",
            AlertKind.STOP_LOSS if was_stop_out else AlertKind.TARGET,
        )

    def kill_switch(
        self,
        *,
        reason: str,
        detail: str,
        realised: Decimal,
        total: Decimal,
        limit: Decimal,
        lock_durable: bool,
    ) -> bool:
        """The session latched (CLAUDE.md §1.2). **The most important message this bot sends.**

        Fired from ``PnLTracker.trip``, which every lockdown path funnels through — the
        realised limit, the total limit, the watchdog's drawdown check, an undefined P&L, and
        a rogue fill. Hooking the funnel rather than each call site is what makes it
        impossible to add a sixth breach path that forgets to tell the operator.

        ``lock_durable`` is not a detail. When ``daily_lock.txt`` could not be written, this
        session is locked but a **restart would not be** — the one case where the operator
        must be told to stay away from the process, and the one thing they cannot see from
        the P&L number alone.
        """
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
            f"<b>TRADING HALTED</b> - {_esc(reason)}\n"
            f"session {_money(total)} (booked {_money(realised)}) vs limit Rs.{limit:,.2f}\n"
            f"{_esc(detail)}\n\n{guidance}{warning}",
            AlertKind.KILL_SWITCH,
        )

    def rogue_fill(
        self, *, symbol: str, order_id: str, quantity: int, side: str, price: Decimal
    ) -> bool:
        """A fill for an order this system never placed (CLAUDE.md §7.4, §9).

        Sent *in addition to* the kill-switch alert, and deliberately so: this is the only
        event in the system where the correct response is to open the broker terminal
        immediately, and it is the one alert worth buzzing twice for. The fill is **not**
        booked — folding in a position we did not open would corrupt the number the loss
        limit is enforced against — so the quantity below is what arrived, not what we hold.
        """
        return self.send(
            f"<b>{_esc(symbol)}</b> - fill for an order we never placed\n"
            f"{_esc(side)} {quantity} @ Rs.{price:,.2f}  order {_esc(order_id)}\n\n"
            f"NOT booked into P&amp;L. Session locked on disk, so a restart stays locked.\n"
            f"Something else is trading this account, or our record of what we placed is "
            f"wrong. OPEN THE BROKER TERMINAL NOW.",
            AlertKind.ROGUE_FILL,
        )

    def session_locked(self, *, reason: str) -> bool:
        """The session latched for a reason that is not a P&L breach.

        Covers the paths that go through ``StrategyBrain._lock`` rather than
        ``PnLTracker.trip``: an entry whose outcome is unknown, an entry that raised, a math
        kernel failing its self-test. An unknown entry outcome means a leg may be live at the
        broker with no local record of it, which needs the operator as much as a breach does
        — and without this it was the one lockdown that happened in silence.
        """
        return self.send(
            f"<b>TRADING HALTED</b> - {_esc(reason)}\n\n"
            f"No new entries this session. This is not a P&amp;L breach - check the broker "
            f"terminal for orders or positions this process may not know about.",
            AlertKind.LOCKED,
        )

    def square_off_started(self, *, open_positions: int) -> bool:
        """15:15 IST — the flatten has begun (§1.1)."""
        return self.send(
            f"15:15 IST auto square-off firing.\n{open_positions} position(s) to flatten.",
            AlertKind.SQUARE_OFF,
        )

    def square_off_completed(self, *, session_total: Decimal) -> bool:
        """The broker confirms zero working orders and zero net positions (§6.5)."""
        return self.send(
            f"Square-off complete — account is flat.\nsession P&amp;L {_money(session_total)}",
            AlertKind.SQUARE_OFF,
        )

    def square_off_failed(self, *, detail: str) -> bool:
        """The flatten did not confirm flat. **The watchdog is still retrying.**

        Sent once per session, not once per retry — see ``StrategyBrain._square_off_action``.
        This is the one alert that may need the operator to open the broker terminal.
        """
        return self.send(
            f"SQUARE-OFF DID NOT CONFIRM FLAT — retrying.\n{_esc(detail)}\n"
            f"Check the broker terminal now.",
            AlertKind.ERROR,
        )

    # ── delivery ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Drain the queue until stopped. Absorbs everything; this thread must not die."""
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                if isinstance(item, str):
                    self._deliver(item)
            except Exception as exc:  # noqa: BLE001 - a sender that dies is worse than silence
                self.stats.failed += 1
                _log.error("telegram.sender_error", error=str(exc), error_type=type(exc).__name__)
            finally:
                self._queue.task_done()

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=SEND_TIMEOUT_SECONDS,
                transport=self._transport,
            )
        return self._client

    def _deliver(self, text: str) -> None:
        """POST one message, retrying exactly once on 429/5xx.

        The URL is never logged: it carries the bot token as a path segment.
        """
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
                # str(exc) can embed the request URL, and that URL carries the token.
                if attempt == 2:
                    self.stats.failed += 1
                    _log.warning("telegram.send_failed", error_type=type(exc).__name__)
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
                status=response.status_code,
                # Telegram's own diagnosis. Safe to log — the body is an error description
                # and never echoes the token, unlike the request URL.
                description=_describe(response),
                hint=_hint_for(response),
            )
            return

    def drain_for_test(self) -> int:
        """Deliver everything queued, synchronously. Tests only — never call in production."""
        delivered = 0
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, str):
                self._deliver(item)
                delivered += 1
            self._queue.task_done()
        return delivered


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────────────

_default: TelegramAlerter | None = None
_default_lock: Final[threading.Lock] = threading.Lock()


def get_alerter() -> TelegramAlerter:
    """The process-wide alerter, built from `.env` on first use.

    The Brain owns its own instance (injectable, so tests never reach the network). This one
    exists for scripts and for :func:`send_telegram_alert`.
    """
    global _default  # noqa: PLW0603 - one lazily-built process-wide sender, guarded by a lock
    with _default_lock:
        if _default is None:
            from tachyon.core.config import get_settings

            _default = TelegramAlerter.from_settings(get_settings())
        return _default


def send_telegram_alert(message: str) -> bool:
    """Send one message to the configured Telegram chat.

    Returns True if the alert was **accepted for delivery** — the actual POST happens on a
    background thread, because this is called from the event loop and from the square-off
    watchdog, and neither may block on a socket.

    Never raises. An unconfigured, unreachable or rate-limited Telegram returns False and
    changes nothing about the trading session.
    """
    try:
        return get_alerter().send(message)
    except Exception as exc:  # noqa: BLE001 - an advisor must never break its caller
        _log.warning("telegram.alert_failed", error=str(exc), error_type=type(exc).__name__)
        return False


def reset_default_alerter() -> None:
    """Drop the process-wide alerter. Tests only."""
    global _default  # noqa: PLW0603 - test-only teardown of the module singleton
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = None
