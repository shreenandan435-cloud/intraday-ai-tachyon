"""ZeroMQ IPC backbone — CLAUDE.md §2.

Publisher binds, subscriber connects. Frames are ``[topic: bytes][msgspec payload]``.
``pickle`` over IPC is banned. HWM 10_000, LINGER 0 — a slow subscriber is dropped,
never allowed to back-pressure the tick feed.

Modules:
  schemas.py     frozen msgspec Structs (Tick, OrderBook, Heartbeat), topics, and the codec
  publisher.py   PUB wrapper used by the ingestion process, incl. publish_heartbeat()
  subscriber.py  SUB wrappers (blocking + zmq.asyncio); conflation permitted for
                 UI/telemetry, forbidden for strategy/risk because VWAP and OBI
                 accumulate over every print
  monitor.py     FeedMonitor — 2s staleness watchdog measured on the receiver's monotonic
                 clock, raising FeedStaleException or firing edge-triggered callbacks
"""

from __future__ import annotations
