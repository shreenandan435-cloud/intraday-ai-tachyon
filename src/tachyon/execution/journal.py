"""Append-only order journal — CLAUDE.md §6.4, §9.

Every broker request and every broker response is written to
``data/journal/orders_<date>.jsonl`` **before it is acted on**. That ordering is the whole
point: if the process dies between sending a placement and recording its outcome, the journal
still shows that a placement was attempted, and
:class:`~tachyon.execution.reconciliation.StateReconciler` can go and find out what happened to
it. A journal written afterwards would be silent about exactly the case it exists to survive.

The mechanics live in :class:`~tachyon.persistence.journal.JsonlJournal`, which the Sentinel
also uses. This module supplies only the filename prefix, so there is one implementation of
"append a scrubbed JSON line to disk" rather than two that can drift apart on redaction.
"""

from __future__ import annotations

from pathlib import Path

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.constants import JOURNAL_DIR
from tachyon.persistence.journal import (
    KIND_DECISION,
    KIND_ERROR,
    KIND_REQUEST,
    KIND_RESPONSE,
    JsonlJournal,
)

__all__ = [
    "KIND_DECISION",
    "KIND_ERROR",
    "KIND_REQUEST",
    "KIND_RESPONSE",
    "OrderJournal",
]


class OrderJournal(JsonlJournal):
    """The broker audit trail: ``data/journal/orders_<date>.jsonl``."""

    __slots__ = ()

    def __init__(
        self,
        directory: Path = JOURNAL_DIR,
        *,
        clock: Clock = SYSTEM_CLOCK,
        enabled: bool = True,
    ) -> None:
        super().__init__(directory, prefix="orders", clock=clock, enabled=enabled)
