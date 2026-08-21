"""Chaos suite — live-fire tests that sabotage the system's dependencies (CLAUDE.md §11).

Every other test asks "does this component do what it says". These ask "what happens when the
thing it depends on dies mid-sentence". They run against real ZeroMQ sockets, real threads and
the real orchestrator components, because a fail-safe that has only ever been exercised through
a mock is a fail-safe nobody has tested.
"""

from __future__ import annotations
