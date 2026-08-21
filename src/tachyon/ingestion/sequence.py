"""Per-token publish sequence numbering — CLAUDE.md §2.1, §3.3.

The transport drops messages by design when a subscriber falls behind, and without a sequence
number that loss is undetectable: VWAP and OBI would simply become quietly wrong. This
assigns a strictly monotonic counter so
:class:`~tachyon.math_engine.core.TickAggregator` can prove it saw every print.

**Numbering is per token, not global.** A gap in RELIANCE must not invalidate INFY's
accumulators, and a global counter would implicate every symbol whenever any one of them
dropped a packet.

Counters start at 1 and are never reset mid-session. A restart therefore restarts at 1, which
the Brain correctly reads as a gap — a restarted ingester genuinely did miss data, and
pretending otherwise would leave stale accumulators running against an incomplete series.
"""

from __future__ import annotations

from typing import Final

FIRST_SEQUENCE: Final[int] = 1


class SequenceManager:
    """Monotonic publish counters keyed by instrument token.

    Single-threaded by design: it is driven from the WebSocket receive path only. It takes no
    lock, because the cost of one on the hottest path is not worth guarding against a sharing
    pattern the architecture already forbids (CLAUDE.md §2.2 — sockets and their handlers stay
    on one thread).

    Example::

        sequences = SequenceManager()
        tick = Tick(..., seq=sequences.next_for("2885"))
    """

    __slots__ = ("_counters",)

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next_for(self, token: str) -> int:
        """Return the next sequence number for ``token``, starting at 1."""
        nxt = self._counters.get(token, 0) + 1
        self._counters[token] = nxt
        return nxt

    def current(self, token: str) -> int:
        """Last issued sequence for ``token``, or 0 if none has been issued."""
        return self._counters.get(token, 0)

    def issued(self) -> int:
        """Total sequence numbers issued across all tokens."""
        return sum(self._counters.values())

    def tokens(self) -> tuple[str, ...]:
        """Tokens that have been issued at least one sequence number."""
        return tuple(self._counters)

    def reset(self, token: str) -> None:
        """Forget one token's counter, so its next value is 1 again.

        Only for tests and explicit resubscription. Calling this on a live token manufactures
        a gap that the Brain will act on.
        """
        self._counters.pop(token, None)

    def reset_all(self) -> None:
        """Forget every counter. Tests and session rollover only."""
        self._counters.clear()

    def __len__(self) -> int:
        return len(self._counters)

    def __repr__(self) -> str:
        return f"SequenceManager(tokens={len(self._counters)}, issued={self.issued()})"
