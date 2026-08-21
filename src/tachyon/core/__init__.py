"""Core primitives — CLAUDE.md §1, §8.

Phase 2 scope:
  constants.py  hard Final values: 15:15 square-off, ₹500 loss limit, 1.5×ATR, 1.5R/2.5R
  config.py     .env secrets + config/settings.yaml tunables (constants live in neither)
  clock.py      Asia/Kolkata aware wall clock + monotonic durations; naive datetimes banned
  logging.py    structlog JSON-line setup; secrets never enter a log line
  state.py      TradingState enum: BOOTING → PREMARKET → ACTIVE → NO_NEW_ENTRIES
                → SQUAREOFF → HALTED (one-way; HALTED/KILLED terminal)
  shutdown.py   graceful SIGTERM/SIGINT handling; ordered teardown; systemd integration
"""

from __future__ import annotations
