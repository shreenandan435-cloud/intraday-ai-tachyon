"""Shared helpers — CLAUDE.md §8.

Small, pure, dependency-light utilities: Decimal tick rounding (ROUND_HALF_UP), IST datetime
helpers, token-bucket rate limiter, backoff schedules, safe formatters that redact secrets.

Nothing with business logic belongs here. If it makes a trading decision, it lives in
``strategy/``, ``risk/`` or ``execution/``.
"""

from __future__ import annotations
