"""FastAPI observability server — CLAUDE.md §7.3.

A **separate process** that watches the trading system and cannot participate in it. It holds
no broker client, no order builder and no risk engine; it subscribes to two ZeroMQ sockets and
serves a browser. If this process dies, trading is unaffected. If it hangs, trading is
unaffected. That independence is the design, not a side effect.

Read-only, with one exception
-----------------------------
Every endpoint is a read except ``POST /api/panic``, and even that cannot place, modify or
cancel an order. It **engages the on-disk daily lock** (``data/journal/daily_lock.txt``) — the
same latch the ₹500 kill switch writes. The Brain's risk gate checks it on every entry
evaluation (§4, check 3), so new entries stop within one evaluation, and the lock survives a
restart.

Be precise about what that button does, because §7.2 requires the UI to be truthful:

* it **blocks all new entries**, immediately and durably;
* it does **not** flatten open positions.

The UI cannot flatten — it has no order capability by constitutional rule, and giving it one to
make a button feel better would be exactly the wrong trade. Open positions are closed by their
broker-side bracket stops, by the 15:15 watchdog, or by the operator at the broker terminal.
The button is labelled accordingly.

Panic is also **not reversible from here**. There is no un-panic endpoint. Clearing the lock is
a deliberate act on the filesystem, which is the correct amount of friction for a decision to
resume trading after someone hit the emergency stop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import DAILY_LOSS_LIMIT_INR
from tachyon.core.logger import get_logger
from tachyon.core.state import DailyLock
from tachyon.persistence.journal import JsonlJournal
from tachyon.ui.postback import OrderStatusListener, watchlist_resolver
from tachyon.ui.telemetry import TelemetryBridge

_log = get_logger(__name__)

STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"

#: Reason recorded when the operator hits the panic button.
PANIC_REASON: Final[str] = "OPERATOR_PANIC"


def create_app(
    *,
    settings: Settings | None = None,
    bridge: TelemetryBridge | None = None,
    clock: Clock = SYSTEM_CLOCK,
    start_bridge: bool = True,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: resolved config.
        bridge: injected telemetry bridge, for tests.
        start_bridge: when False the poll loop is not started — tests drive ``poll_once``
            themselves rather than racing a background task.
    """
    resolved = settings if settings is not None else get_settings()
    telemetry = bridge if bridge is not None else TelemetryBridge(settings=resolved, clock=clock)
    daily_lock = DailyLock(clock=clock)
    journal = JsonlJournal(prefix="ui", clock=clock)

    listener = OrderStatusListener(
        on_closed=_log_only_closure,
        symbol_resolver=watchlist_resolver(
            {item.token: item.symbol for item in resolved.watchlist},
            {item.symbol: item.symbol for item in resolved.watchlist},
        ),
        on_fill=lambda update: telemetry.record_event(
            "FILL",
            symbol=update.symbol,
            order_id=update.order_id,
            status=update.status,
            side=update.side,
            filled=update.filled_quantity,
            price=str(update.average_price),
        ),
        daily_lock=daily_lock,
        journal=journal,
        mode=resolved.trading_mode,
        clock=clock,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if start_bridge:
            task = asyncio.create_task(telemetry.run(), name="ui-telemetry")
        _log.info("ui.started", host=resolved.ui_host, port=resolved.ui_port)
        try:
            yield
        finally:
            await telemetry.stop()
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            _log.info("ui.stopped")

    app = FastAPI(
        title="TACHYON",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.telemetry = telemetry
    app.state.listener = listener
    app.state.daily_lock = daily_lock
    app.state.settings = resolved
    app.state.clock = clock

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ── the terminal ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the shell. Loads fully offline — no CDN, no external font (CLAUDE.md §7)."""
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>TACHYON: static/index.html missing</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # ── read-only API ────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "at_ist": now_ist(clock).isoformat(timespec="milliseconds"),
            "clients": telemetry.client_count,
            "mode": resolved.trading_mode.value,
        }

    @app.get("/api/snapshot")
    async def snapshot() -> dict[str, Any]:
        """The same payload the WebSocket pushes. For curl and for a browser without WS."""
        return telemetry.snapshot()

    @app.get("/api/limits")
    async def limits() -> dict[str, Any]:
        """The §1 constants, read from the frozen module so the UI cannot drift from them."""
        return {
            "daily_loss_limit_inr": str(DAILY_LOSS_LIMIT_INR),
            "daily_lock_engaged": daily_lock.is_engaged(),
            "mode": resolved.trading_mode.value,
        }

    # ── the one write ────────────────────────────────────────────────────────

    @app.post("/api/panic")
    async def panic(request: Request) -> JSONResponse:
        """Engage the daily lock. Blocks all new entries; does **not** flatten.

        Idempotent, irreversible from here, and it never touches the broker. A failure to write
        the lock is reported as a failure — the operator must not be told the market is closed
        to them when it is not.
        """
        client = request.client.host if request.client else "unknown"
        try:
            record = daily_lock.engage(
                reason=PANIC_REASON,
                realised_pnl_inr=None,
                mode=resolved.trading_mode,
            )
        except OSError as exc:
            _log.critical(
                "ui.panic_failed",
                client=client,
                error=str(exc),
                impact="THE LOCK WAS NOT WRITTEN — new entries are still permitted",
            )
            return JSONResponse(
                status_code=500,
                content={
                    "engaged": False,
                    "error": str(exc),
                    "impact": "lock not written — new entries are STILL PERMITTED",
                },
            )

        _log.critical(
            "ui.panic",
            client=client,
            action="daily lock engaged — new entries blocked for the rest of the session",
            note="open positions are NOT flattened by this endpoint",
        )
        journal.decision("operator_panic", client=client, at_ist=record.engaged_at_ist)
        telemetry.record_event(
            "RISK",
            kind="OPERATOR_PANIC",
            severity="CRITICAL",
            reason="Operator panic — new entries blocked",
            detail="Open positions are not flattened by this action",
        )
        return JSONResponse(
            {
                "engaged": True,
                "at_ist": record.engaged_at_ist,
                "blocks": "all new entries, for the rest of the session and across restarts",
                "does_not": "flatten open positions",
            }
        )

    # ── broker postback ──────────────────────────────────────────────────────

    @app.post("/api/postback")
    async def postback(payload: dict[str, Any]) -> dict[str, Any]:
        """Angel One order-status webhook.

        Always answers 200, even for a payload it could not use. A broker that receives an
        error retries, and a retry storm against a body we will never parse helps nobody —
        the rejection is logged and counted instead.
        """
        accepted = listener.ingest_raw(payload)
        return {"accepted": accepted}

    # ── telemetry socket ─────────────────────────────────────────────────────

    @app.websocket("/ws/telemetry")
    async def telemetry_socket(websocket: WebSocket) -> None:
        """Push snapshots at ``ui.ws_max_hz``.

        The client is registered, then drained by a loop that only ever awaits *its own*
        queue. A browser that stops reading fills that queue and starts losing its own frames;
        it cannot slow the bridge, the other clients, or the Brain (CLAUDE.md §7.3).
        """
        await websocket.accept()
        peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "ws"
        channel = telemetry.register(peer)
        try:
            while True:
                frame = await channel.get()
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - one bad socket must not affect the others
            _log.warning(
                "ui.client_error", client=peer, error=str(exc), error_type=type(exc).__name__
            )
        finally:
            telemetry.unregister(channel)
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def _log_only_closure(
    symbol: str,
    realised: Decimal,
    charges: Decimal,
    was_stop_out: bool,
    at: Any,
) -> None:
    """Default closure handler when the listener runs inside the UI process.

    The UI holds no ``PnLTracker`` and no ``ReentryManager`` — those live in the Brain, and
    duplicating them here would create a second, divergent answer to "what is today's P&L".
    So this records the close and nothing more.

    In the deployed topology the Brain owns the listener and passes its own
    ``on_position_closed``; this exists so a UI started standalone still journals fills rather
    than dropping them on the floor.
    """
    _log.info(
        "ui.position_closed_observed",
        symbol=symbol,
        realised_inr=str(realised),
        charges_inr=str(charges),
        was_stop_out=was_stop_out,
        at_ist=str(at),
        note="observed only — the Brain owns P&L and the cooldown",
    )
