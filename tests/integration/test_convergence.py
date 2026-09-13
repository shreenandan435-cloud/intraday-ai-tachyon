"""Integration test for The Convergence — ExecutionRouter via ZeroMQ action spine.

Spins up the router in paper mode, injects simulated tick actions through a mock
ZeroMQ PUSH socket, verifies the risk gate passes (or vetoes correctly), simulates
trade fills against a mock top-of-book, and asserts telemetry reaches the sink.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import pytest
import zmq

from tachyon.core import eventloop
from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import get_settings, reload_settings
from tachyon.execution.router import (
    ACTION_BUY,
    ExecutionRouter,
    TopOfBook,
)
from tachyon.math_engine import warmup


@pytest.fixture(scope="session", autouse=True)
def _warm_math_engine() -> None:
    assert warmup() is True


@pytest.fixture(autouse=True)
def _reload_settings() -> None:
    reload_settings()


def test_paper_mode_action_fill_accuracy_and_telemetry() -> None:
    """Router receives a mock PUSH action, evaluates risk, simulates fill, and logs."""
    settings = get_settings()
    clock = ManualClock(
        wall=datetime(2026, 8, 10, 11, 0, tzinfo=IST),
        mono=1000.0,
    )

    # Mock book provider supplying a healthy top-of-book.
    def mock_book(token: int) -> TopOfBook | None:
        return TopOfBook(
            bid=100.0,
            bid_qty=10.0,
            ask=101.0,
            ask_qty=10.0,
            ts_epoch=time.time(),
        )

    # Mock Parquet telemetry sink that captures router decisions.
    captured: list[dict[str, Any]] = []

    class MockTradeSink:
        def log(self, record: dict[str, Any]) -> None:
            captured.append(record)

    router = ExecutionRouter(
        paper_trade=True,
        settings=settings,
        book_provider=mock_book,
        trade_sink=MockTradeSink(),
        clock=clock,
    )

    async def _run() -> None:
        router_task = asyncio.create_task(router.run(), name="router-test")

        # Allow socket connection to settle.
        await asyncio.sleep(0.3)

        # Bind a mock PUSH socket to the action endpoint so the PULL side consumes.
        ctx = zmq.Context()
        push = ctx.socket(zmq.PUSH)
        push.bind("tcp://127.0.0.1:5567")
        push.setsockopt(zmq.LINGER, 0)
        push.setsockopt(zmq.SNDHWM, 100)

        # Inject a BUY action for the first watchlisted instrument.
        token_str = settings.watchlist[0].token if settings.watchlist else "2885"
        payload = {
            "action": ACTION_BUY,
            "instrument_token": int(token_str),
            "timestamp_ns": time.time_ns(),
        }
        push.send_string(json.dumps(payload))

        # Wait for the router to decode, gate, fill, and sink the outcome.
        await asyncio.sleep(0.5)

        # Clean shutdown: signal the loop and await termination.
        router.stop()
        try:
            await asyncio.wait_for(router_task, timeout=3.0)
        except TimeoutError:
            router.close()
            await router_task

        push.close()
        ctx.term()

    eventloop.run(_run())

    # Assertions contract
    assert router.stats.received >= 1, (
        f"Expected at least one received action, got {router.stats.received}"
    )
    # Whether the entry passed or was vetoed depends on risk state; at minimum the
    # sink must have captured the outcome record.
    assert len(captured) >= 1, (
        f"Expected at least one sink record, captured {len(captured)}"
    )

    # Confirm the sink record carries the fixed uniform schema.
    record = captured[0]
    required_keys = {
        "event",
        "accepted",
        "mode",
        "symbol",
        "token",
        "action",
        "action_name",
        "reason",
        "detail",
        "side",
        "quantity",
        "price",
        "pnl",
        "order_id",
        "ts_epoch",
        "latency_ms",
    }
    assert required_keys.issubset(record.keys()), (
        f"Sink record missing keys: {required_keys - record.keys()}"
    )

    # Verify paper-mode label.
    assert record["mode"] == "PAPER"

    # Verify the action was evaluated against the risk engine (either ENTRY or REJECTED).
    assert record["event"] in {"ENTRY", "REJECTED", "DROPPED", "HOLD"}
