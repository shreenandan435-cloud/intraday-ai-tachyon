"""Market data ingestion — Process A, CLAUDE.md §2.

SmartAPI WebSocket v2 → binary tick + 5-level L2 depth decode → msgspec → ZMQ PUB :5555.
This process contains ZERO business logic: no indicators, no signals, no orders.
It must keep publishing even when the Brain is dead.

Modules:
  decoder.py     pure struct unpacking of the little-endian v2 packets (51/123/379 bytes)
  sequence.py    SequenceManager — strictly monotonic publish counter per instrument token
  ws_client.py   async client with flap-resistant exponential backoff and app-level keepalive
  service.py     IngestionService — wires the feed to the ZMQ publisher, runs the heartbeat
  instruments.py cached, self-expiring scrip master; verifies watchlist tokens before boot

A token is the only instrument identity that reaches the wire, and the broker accepts a
retired one without complaint — the subscription is acknowledged and no data ever arrives.
``instruments.py`` exists so that failure is caught at boot rather than diagnosed at 09:15.

The heartbeat is published only while the WebSocket is actually subscribed. Heartbeating
through an outage would hold the Brain's FeedMonitor "fresh" while no market data flowed,
which is exactly the failure the 2-second staleness rule exists to catch.
"""

from __future__ import annotations
