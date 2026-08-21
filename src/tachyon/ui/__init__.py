"""Cyberpunk glass terminal — FastAPI + WebSocket, CLAUDE.md §7.

A **separate process** that watches the trading system and cannot participate in it. It holds
no broker client, no order builder and no risk engine. If it dies, trading is unaffected; if it
hangs, trading is unaffected.

Read-only by design. The single mutating endpoint is ``POST /api/panic``, which engages the
on-disk daily lock — the same latch the ₹500 kill switch writes. It blocks all new entries
immediately and across restarts. It does **not** flatten open positions, and the button says so,
because §7.2 puts truthfulness above aesthetics.

Push is throttled server-side to ``ui.ws_max_hz`` (10 Hz) — never one frame per tick. Each
client owns a bounded queue and loses *its own* frames when it falls behind; it can never slow
the bridge, the other clients, or the Brain.

Styling law (non-negotiable):
  canvas #08091a, glassmorphic panels (blur 18px), primary numerals 30px monospace with
  ``font-variant-numeric: tabular-nums`` and a neon glow. Colour carries meaning:
  lime = profit/long, magenta = loss/short, amber = degraded/stale, cyan = neutral.
  A stale value renders amber with its age — never as if it were fresh.

Modules:
  app.py        FastAPI app, /ws/telemetry, read-only REST, /api/panic, /api/postback
  telemetry.py  TelemetryBridge — conflating ZMQ reader + non-blocking WebSocket fan-out
  postback.py   OrderStatusListener — the fill source; closes positions and starts cooldowns
  static/       vanilla CSS/JS only; no React, no Tailwind, no CDN — loads fully offline
"""

from __future__ import annotations

from tachyon.ui.app import create_app
from tachyon.ui.postback import (
    OrderStatusListener,
    OrderUpdate,
    PositionLedger,
    watchlist_resolver,
)
from tachyon.ui.telemetry import ClientChannel, SymbolView, TelemetryBridge

__all__ = [
    "ClientChannel",
    "OrderStatusListener",
    "OrderUpdate",
    "PositionLedger",
    "SymbolView",
    "TelemetryBridge",
    "create_app",
    "watchlist_resolver",
]
