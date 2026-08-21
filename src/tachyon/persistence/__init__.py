"""Append-only journal + session state — CLAUDE.md §9.

``data/journal/`` is the post-mortem record of record: orders, fills, risk events, sentinel
prompts/responses, and the kill-switch lock file. Append-only; never rewritten in place.

Journals are written BEFORE the system acts on the corresponding decision, so a crash mid-flight
leaves evidence of intent. Writes on latency-critical paths go through a queue, never
synchronously to disk.
"""

from __future__ import annotations
