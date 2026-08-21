"""Phase 10 UI and fill-source tests — CLAUDE.md §6, §7.

Three things carry real risk here and get the most attention.

**Double-booking a fill.** The webhook and the order-book poll deliver the same event, in
either order, repeatedly. Booking one twice would double the realised P&L that the ₹500 kill
switch is enforced against — a wrong number on the one value that must not be wrong.

**Blocking.** The UI must not be able to slow the Brain, and one browser must not be able to
slow another. A client that stops reading loses its own frames and nothing else.

**Lying.** §7.2 puts truthfulness above aesthetics: a stale value is amber with its age, a value
that never arrived is an em dash rather than a zero, and a panic that failed to write says so.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import Settings, UiSettings, WatchlistItem
from tachyon.core.state import DailyLock
from tachyon.execution.charges import estimate_charges
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import (
    OrderBook,
    PnLUpdate,
    RiskEvent,
    StateUpdate,
    Tick,
)
from tachyon.ipc.subscriber import Subscriber, SubscriberRole
from tachyon.persistence.journal import JsonlJournal
from tachyon.ui.app import STATIC_DIR, create_app
from tachyon.ui.postback import (
    OrderStatusListener,
    OrderUpdate,
    watchlist_resolver,
)
from tachyon.ui.telemetry import CLIENT_QUEUE_DEPTH, ClientChannel, TelemetryBridge

TICK_ENDPOINT = "tcp://127.0.0.1:5801"
STATE_ENDPOINT = "tcp://127.0.0.1:5802"


def _clock(hh: int = 11, mm: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=mono)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (
            WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),
            WatchlistItem(symbol="HDFCBANK", token="1333", exchange="NSE"),
        ),
        "ui": UiSettings(),
        "zmq_tick_endpoint": TICK_ENDPOINT,
        "zmq_state_endpoint": STATE_ENDPOINT,
    }
    base.update(overrides)
    return Settings(**base)


def _resolver() -> Any:
    return watchlist_resolver({"2885": "RELIANCE", "1333": "HDFCBANK"}, {"RELIANCE-EQ": "RELIANCE"})


def _listener(tmp_path: Path, **kwargs: Any) -> tuple[OrderStatusListener, list[tuple[Any, ...]]]:
    closures: list[tuple[Any, ...]] = []

    def on_closed(
        symbol: str, realised: Decimal, charges: Decimal, stop_out: bool, at: datetime
    ) -> None:
        closures.append((symbol, realised, charges, stop_out, at))

    listener = OrderStatusListener(
        on_closed=kwargs.pop("on_closed", on_closed),
        symbol_resolver=_resolver(),
        journal=JsonlJournal(tmp_path, prefix="fills", clock=_clock()),
        clock=_clock(),
        **kwargs,
    )
    return listener, closures


def _row(
    order_id: str,
    *,
    side: str = "BUY",
    status: str = "complete",
    filled: int = 10,
    price: str = "2500.00",
    order_type: str = "LIMIT",
    token: str = "2885",
) -> dict[str, Any]:
    return {
        "orderid": order_id,
        "symboltoken": token,
        "tradingsymbol": "RELIANCE-EQ",
        "transactiontype": side,
        "orderstatus": status,
        "quantity": filled,
        "filledshares": filled,
        "averageprice": price,
        "ordertype": order_type,
        "ordertag": "TCHYN-20260810-0001",
    }


# ──────────────────────────────────────────────────────────────────────────────
# charges.py
# ──────────────────────────────────────────────────────────────────────────────


class TestCharges:
    def test_a_round_trip_costs_something(self) -> None:
        breakdown = estimate_charges(
            buy_turnover=Decimal("17500"), sell_turnover=Decimal("17600"), orders=2
        )
        assert breakdown.total > 0
        assert breakdown.brokerage > 0
        assert breakdown.stt > 0

    def test_stt_is_sell_side_only(self) -> None:
        buy_only = estimate_charges(buy_turnover=Decimal("10000"), sell_turnover=Decimal("0"))
        assert buy_only.stt == 0

    def test_stamp_duty_is_buy_side_only(self) -> None:
        sell_only = estimate_charges(buy_turnover=Decimal("0"), sell_turnover=Decimal("10000"))
        assert sell_only.stamp == 0

    def test_brokerage_is_capped_per_order(self) -> None:
        """0.03 % of a large turnover exceeds ₹20; the cap must bind."""
        huge = estimate_charges(
            buy_turnover=Decimal("5000000"), sell_turnover=Decimal("5000000"), orders=2
        )
        assert huge.brokerage == Decimal("40")

    def test_four_legs_cost_more_than_two(self) -> None:
        """A two-leg bracket that fills entry and exit on both legs is four orders."""
        two = estimate_charges(
            buy_turnover=Decimal("17500"), sell_turnover=Decimal("17600"), orders=2
        )
        four = estimate_charges(
            buy_turnover=Decimal("17500"), sell_turnover=Decimal("17600"), orders=4
        )
        assert four.total > two.total

    def test_negative_turnover_never_produces_a_credit(self) -> None:
        """A cost that reduces the loss would be a very expensive sign error."""
        breakdown = estimate_charges(
            buy_turnover=Decimal("-10000"), sell_turnover=Decimal("-10000")
        )
        assert breakdown.total >= 0

    def test_charges_are_material_against_the_daily_budget(self) -> None:
        """Documents why §1.2 enforces the limit inclusive of charges."""
        breakdown = estimate_charges(
            buy_turnover=Decimal("17500"), sell_turnover=Decimal("17600"), orders=4
        )
        assert breakdown.total > Decimal("10")  # >2% of the ₹500 daily budget, on one trade


# ──────────────────────────────────────────────────────────────────────────────
# postback.py — the fill source
# ──────────────────────────────────────────────────────────────────────────────


class TestOrderStatusListener:
    def test_a_round_trip_closes_the_position(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("A1", side="BUY", filled=10, price="2500.00"))
        assert listener.net_quantity("RELIANCE") == 10
        assert not closures

        listener.ingest_raw(_row("A2", side="SELL", filled=10, price="2518.90"))
        assert listener.net_quantity("RELIANCE") == 0
        assert len(closures) == 1

        symbol, realised, charges, stop_out, _at = closures[0]
        assert symbol == "RELIANCE"
        assert realised == Decimal("189.00")  # (2518.90 − 2500.00) × 10
        assert charges > 0
        assert not stop_out

    def test_a_short_round_trip_closes_too(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("B1", side="SELL", filled=10, price="2500.00"))
        assert listener.net_quantity("RELIANCE") == -10
        listener.ingest_raw(_row("B2", side="BUY", filled=10, price="2490.00"))
        assert len(closures) == 1
        assert closures[0][1] == Decimal("100.00")

    def test_a_duplicate_delivery_is_ignored(self, tmp_path: Path) -> None:
        """The webhook and the poller deliver the same fill. Booking it twice doubles P&L."""
        listener, closures = _listener(tmp_path)
        row = _row("C1", side="BUY", filled=10)
        assert listener.ingest_raw(row)
        assert not listener.ingest_raw(row)
        assert not listener.ingest_raw(dict(row))  # a fresh dict with identical content

        assert listener.net_quantity("RELIANCE") == 10
        assert listener.stats.duplicates == 2
        assert not closures

    def test_the_poller_can_replay_the_whole_order_book_safely(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        book = [
            _row("D1", side="BUY", filled=10, price="2500.00"),
            _row("D2", side="SELL", filled=10, price="2510.00"),
        ]
        assert listener.ingest_order_book(book) == 2
        assert listener.ingest_order_book(book) == 0
        assert listener.ingest_order_book(book) == 0
        assert len(closures) == 1, "replaying the book must not close the position again"

    def test_partial_fills_are_applied_as_deltas(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        for filled in (3, 7, 10):
            listener.ingest_raw(_row("E1", side="BUY", filled=filled, price="2500.00"))
        assert listener.net_quantity("RELIANCE") == 10, "3 + 4 + 3, not 3 + 7 + 10"

        listener.ingest_raw(_row("E2", side="SELL", filled=10, price="2500.00"))
        assert len(closures) == 1

    def test_an_out_of_order_partial_is_not_unwound(self, tmp_path: Path) -> None:
        """A late-arriving smaller fill must not remove quantity that genuinely traded."""
        listener, _closures = _listener(tmp_path)
        listener.ingest_raw(_row("F1", side="BUY", filled=10, price="2500.00"))
        listener.ingest_raw(_row("F1", side="BUY", filled=4, price="2500.00"))
        assert listener.net_quantity("RELIANCE") == 10

    def test_a_stop_out_is_identified_by_the_closing_order(self, tmp_path: Path) -> None:
        """Not by whether the trade lost money — §8.1's cooldown is about being stopped."""
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("G1", side="BUY", filled=10, price="2500.00"))
        listener.ingest_raw(
            _row("G2", side="SELL", filled=10, price="2487.40", order_type="STOPLOSS_LIMIT")
        )
        assert closures[0][3] is True

    def test_a_losing_trade_closed_at_target_is_not_a_stop_out(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("H1", side="BUY", filled=10, price="2500.00"))
        listener.ingest_raw(_row("H2", side="SELL", filled=10, price="2499.00"))
        assert closures[0][1] < 0
        assert closures[0][3] is False

    @pytest.mark.parametrize("order_type", ["STOPLOSS_LIMIT", "STOPLOSS_MARKET", "SL-M", "sl_m"])
    def test_every_stop_spelling_is_recognised(self, tmp_path: Path, order_type: str) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("I1", side="BUY", filled=10))
        listener.ingest_raw(_row("I2", side="SELL", filled=10, order_type=order_type))
        assert closures[0][3] is True

    def test_a_symbol_off_the_watchlist_is_dropped(self, tmp_path: Path) -> None:
        """§8.1 applied inbound: a fill we may not have traded must not move P&L."""
        listener, closures = _listener(tmp_path)
        stranger = _row("J1", token="99999")
        stranger["tradingsymbol"] = "YESBANK-EQ"
        assert not listener.ingest_raw(stranger)
        assert listener.stats.dropped == 1
        assert not closures

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"orderid": ""},
            {"orderid": "K1"},  # no instrument
            {"orderid": "K1", "symboltoken": "2885", "filledshares": "not-a-number"},
            {"orderid": "K1", "symboltoken": "2885", "averageprice": None},
        ],
    )
    def test_a_malformed_payload_never_raises(self, tmp_path: Path, payload: Any) -> None:
        listener, _closures = _listener(tmp_path)
        listener.ingest_raw(payload)  # must not raise
        assert listener.net_quantity("RELIANCE") == 0

    def test_a_cancelled_order_moves_no_quantity(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("L1", status="cancelled", filled=0))
        listener.ingest_raw(_row("L2", status="rejected", filled=0))
        assert listener.net_quantity("RELIANCE") == 0
        assert not closures

    def test_a_crashing_callback_does_not_lose_the_listener(self, tmp_path: Path) -> None:
        def explode(*_args: Any) -> None:
            raise RuntimeError("consumer bug")

        listener, _ = _listener(tmp_path, on_closed=explode)
        listener.ingest_raw(_row("M1", side="BUY", filled=10, price="2500.00"))
        listener.ingest_raw(_row("M2", side="SELL", filled=10, price="2510.00"))
        assert listener.stats.callback_errors == 1
        # And the listener is still usable afterwards.
        listener.ingest_raw(_row("M3", side="BUY", filled=5, price="2500.00"))
        assert listener.net_quantity("RELIANCE") == 5

    def test_positions_are_tracked_per_symbol(self, tmp_path: Path) -> None:
        listener, closures = _listener(tmp_path)
        listener.ingest_raw(_row("N1", side="BUY", filled=10, token="2885"))
        hdfc = _row("N2", side="BUY", filled=5, token="1333")
        hdfc["tradingsymbol"] = "HDFCBANK-EQ"
        listener.ingest_raw(hdfc)

        assert listener.open_symbols() == frozenset({"RELIANCE", "HDFCBANK"})
        listener.ingest_raw(_row("N3", side="SELL", filled=10, token="2885"))
        assert listener.open_symbols() == frozenset({"HDFCBANK"})
        assert [c[0] for c in closures] == ["RELIANCE"]

    def test_reset_session_clears_everything(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        listener.ingest_raw(_row("O1", side="BUY", filled=10))
        listener.reset_session()
        assert listener.net_quantity("RELIANCE") == 0
        # And the dedupe set is cleared, so the same id may be reused tomorrow.
        assert listener.ingest_raw(_row("O1", side="BUY", filled=10))

    def test_fills_are_journalled(self, tmp_path: Path) -> None:
        journal = JsonlJournal(tmp_path, prefix="fills", clock=_clock())
        listener = OrderStatusListener(
            on_closed=lambda *_: None,
            symbol_resolver=_resolver(),
            journal=journal,
            clock=_clock(),
        )
        listener.ingest_raw(_row("P1", side="BUY", filled=10))
        listener.ingest_raw(_row("P2", side="SELL", filled=10))
        records = journal.read()
        assert any(r["kind"] == "FILL" for r in records)
        assert any(r["event"] == "position_closed" for r in records)

    def test_the_fill_callback_sees_every_update(self, tmp_path: Path) -> None:
        seen: list[OrderUpdate] = []
        listener, _ = _listener(tmp_path, on_fill=seen.append)
        listener.ingest_raw(_row("Q1", side="BUY", filled=10))
        listener.ingest_raw(_row("Q1", status="cancelled", filled=10))
        assert len(seen) == 2

    def test_a_crashing_fill_callback_does_not_affect_bookkeeping(self, tmp_path: Path) -> None:
        def explode(_update: OrderUpdate) -> None:
            raise RuntimeError("telemetry bug")

        listener, closures = _listener(tmp_path, on_fill=explode)
        listener.ingest_raw(_row("R1", side="BUY", filled=10, price="2500.00"))
        listener.ingest_raw(_row("R2", side="SELL", filled=10, price="2510.00"))
        assert len(closures) == 1, "telemetry must never affect the P&L path"


# ──────────────────────────────────────────────────────────────────────────────
# telemetry.py
# ──────────────────────────────────────────────────────────────────────────────


class TestClientChannel:
    async def test_a_slow_client_loses_its_own_frames_only(self) -> None:
        channel = ClientChannel("slow", depth=3)
        assert channel.offer(b"1")
        assert channel.offer(b"2")
        assert channel.offer(b"3")
        assert not channel.offer(b"4"), "the queue is full; the oldest frame is dropped"

        assert channel.dropped == 1
        assert channel.pending == 3
        assert await channel.get() == b"2", "the oldest frame went, not the newest"

    async def test_offer_never_blocks(self) -> None:
        """The fan-out must never await a client. A wedged tab cannot stall the bridge."""
        channel = ClientChannel("wedged", depth=CLIENT_QUEUE_DEPTH)
        for i in range(CLIENT_QUEUE_DEPTH * 10):
            channel.offer(str(i).encode())
        assert channel.pending == CLIENT_QUEUE_DEPTH
        assert channel.dropped == CLIENT_QUEUE_DEPTH * 9


@pytest.fixture
def spine() -> Any:
    """Two bound publishers and a bridge whose subscribers connect to them.

    Bind before connect: ZeroMQ's slow-joiner problem means a SUB connected first silently
    loses the opening messages, which reads exactly like a broken bridge.
    """
    tick_pub = Publisher(TICK_ENDPOINT, role="ingestor")
    state_pub = Publisher(STATE_ENDPOINT, role="brain")
    settings = _settings()
    bridge = TelemetryBridge(settings=settings, clock=_clock())
    Publisher.settle(0.3)
    try:
        yield tick_pub, state_pub, bridge
    finally:
        asyncio.run(bridge.stop())
        tick_pub.close()
        state_pub.close()


class TestTelemetryBridge:
    def test_conflation_keeps_only_the_newest_per_symbol(self, spine: Any) -> None:
        """The whole point: the UI renders the latest value, never a backlog."""
        tick_pub, _state_pub, bridge = spine
        for seq in range(1, 501):
            tick_pub.publish_tick(
                "RELIANCE",
                Tick(token="2885", ltp=2500.0 + seq, volume=1000 + seq, ts_epoch=1.0, seq=seq),
            )
        Publisher.settle(0.3)
        bridge.poll_once()

        symbols = bridge.snapshot()["symbols"]
        assert len(symbols) == 1
        assert symbols[0]["ltp"] == 3000.0, "the newest tick, not the first of 500"
        assert bridge.stats.tick_frames == 1, "500 messages collapsed to one rendered frame"

    def test_depth_and_ticks_merge_into_one_view(self, spine: Any) -> None:
        tick_pub, _state_pub, bridge = spine
        tick_pub.publish_tick(
            "RELIANCE", Tick(token="2885", ltp=2500.0, volume=1, ts_epoch=1.0, seq=1)
        )
        tick_pub.publish_orderbook(
            "RELIANCE",
            OrderBook(
                token="2885",
                bid_price=(2499.95, 0, 0, 0, 0),
                bid_qty=(500, 0, 0, 0, 0),
                ask_price=(2500.05, 0, 0, 0, 0),
                ask_qty=(100, 0, 0, 0, 0),
                ts_epoch=1.0,
            ),
        )
        Publisher.settle(0.3)
        bridge.poll_once()

        view = bridge.snapshot()["symbols"][0]
        assert view["ltp"] == 2500.0
        assert view["best_bid"] == 2499.95
        assert round(view["spread"], 2) == 0.10

    def test_state_and_pnl_reach_the_snapshot(self, spine: Any) -> None:
        _tick_pub, state_pub, bridge = spine
        state_pub.publish_raw(
            b"STATE.SESSION",
            _encode(
                StateUpdate(state="ACTIVE", mode="PAPER", ts_epoch=1.0, open_symbols=("RELIANCE",))
            ),
        )
        state_pub.publish_raw(
            b"PNL.SESSION",
            _encode(
                PnLUpdate(
                    realised="-42.50",
                    floating="0",
                    charges="12.30",
                    total="-54.80",
                    headroom="445.20",
                    limit="500",
                    breached=False,
                    ts_epoch=1.0,
                )
            ),
        )
        Publisher.settle(0.3)
        bridge.poll_once()

        snapshot = bridge.snapshot()
        assert snapshot["session"]["state"] == "ACTIVE"
        assert snapshot["pnl"]["total"] == "-54.80", "money crosses as a string, never a float"

    def test_risk_events_are_kept_not_conflated_away(self, spine: Any) -> None:
        """A dropped tick costs a repaint. A dropped fill is a trade nobody sees."""
        _tick_pub, state_pub, bridge = spine
        for i in range(5):
            state_pub.publish_raw(
                f"RISK.VETO{i}".encode(),
                _encode(
                    RiskEvent(
                        kind="VETO",
                        symbol="RELIANCE",
                        reason=f"reason-{i}",
                        detail="",
                        ts_epoch=1.0,
                    )
                ),
            )
        Publisher.settle(0.3)
        bridge.poll_once()

        reasons = {event["reason"] for event in bridge.snapshot()["events"]}
        assert len(reasons) == 5

    def test_polling_an_empty_socket_is_harmless(self, spine: Any) -> None:
        _tick, _state, bridge = spine
        for _ in range(20):
            bridge.poll_once()
        assert bridge.stats.polls == 20
        assert bridge.snapshot()["symbols"] == []

    async def test_broadcast_never_blocks_on_a_stalled_client(self, spine: Any) -> None:
        tick_pub, _state_pub, bridge = spine
        stalled = bridge.register("stalled")
        healthy = bridge.register("healthy")

        for seq in range(1, 200):
            tick_pub.publish_tick(
                "RELIANCE",
                Tick(token="2885", ltp=2500.0, volume=1, ts_epoch=1.0, seq=seq),
            )
        Publisher.settle(0.2)

        # Nobody drains `stalled`. `healthy` is drained between broadcasts.
        for _ in range(60):
            bridge.poll_once()
            bridge.broadcast()
            healthy.try_get()

        assert stalled.dropped > 0, "the stalled client shed its own frames"
        assert healthy.dropped == 0, "and the healthy client was unaffected"
        assert bridge.stats.frames_dropped > 0
        assert bridge.client_count == 2, "and was never disconnected for it"

    def test_a_new_client_is_primed_with_the_current_snapshot(self, spine: Any) -> None:
        """A browser connecting mid-session must not stare at blank panels."""
        tick_pub, _state_pub, bridge = spine
        tick_pub.publish_tick(
            "RELIANCE", Tick(token="2885", ltp=2500.0, volume=1, ts_epoch=1.0, seq=1)
        )
        Publisher.settle(0.3)
        bridge.poll_once()

        channel = bridge.register("fresh")
        assert channel.pending == 1

    def test_conflation_is_refused_for_the_strategy_role(self) -> None:
        """The guard that keeps a sampled VWAP out of the trading path (§2.1)."""
        from tachyon.ipc.subscriber import ConflationForbiddenError

        with pytest.raises(ConflationForbiddenError):
            Subscriber(
                ("TICK.",),
                role=SubscriberRole.STRATEGY,
                endpoint=TICK_ENDPOINT,
                conflate=True,
            )


def _encode(message: Any) -> bytes:
    from tachyon.ipc.schemas import encode

    return encode(message)


# ──────────────────────────────────────────────────────────────────────────────
# app.py
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A TestClient over the app, with the daily lock redirected into tmp_path."""
    import tachyon.core.state as state_module
    import tachyon.ui.app as app_module

    lock_path = tmp_path / "daily_lock.txt"
    monkeypatch.setattr(state_module, "DAILY_LOCK_FILE", lock_path)
    monkeypatch.setattr(
        app_module, "DailyLock", lambda **kw: DailyLock(path=lock_path, clock=kw.get("clock"))
    )

    settings = _settings()
    bridge = TelemetryBridge(
        settings=settings,
        tick_subscriber=Subscriber(
            ("TICK.",), role=SubscriberRole.UI, endpoint=TICK_ENDPOINT, conflate=True
        ),
        state_subscriber=Subscriber(
            ("STATE.",), role=SubscriberRole.UI, endpoint=STATE_ENDPOINT, conflate=True
        ),
        clock=_clock(),
    )
    app = create_app(settings=settings, bridge=bridge, clock=_clock(), start_bridge=False)
    with TestClient(app) as test_client:
        test_client.lock_path = lock_path  # type: ignore[attr-defined]
        yield test_client


class TestApp:
    def test_health_is_read_only_and_honest_about_the_mode(self, client: Any) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["mode"] == "PAPER"

    def test_snapshot_matches_the_websocket_payload(self, client: Any) -> None:
        body = client.get("/api/snapshot").json()
        assert body["type"] == "snapshot"
        assert "pnl" in body and "session" in body and "symbols" in body

    def test_limits_come_from_the_frozen_constants(self, client: Any) -> None:
        """The UI must not be able to display a limit that differs from the enforced one."""
        body = client.get("/api/limits").json()
        assert body["daily_loss_limit_inr"] == "500"

    def test_the_index_page_loads_offline(self, client: Any) -> None:
        """Every asset must resolve locally — no CDN, no external font (CLAUDE.md §7).

        Asserted on actual references rather than on the substring "cdn", which would also
        match the source comment explaining that there is no CDN.
        """
        html = client.get("/").text
        assert "TACHYON" in html
        assert not re.search(r'(src|href)\s*=\s*"https?://', html)

    def test_static_assets_are_served(self, client: Any) -> None:
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200

    def test_the_websocket_pushes_a_snapshot_immediately(self, client: Any) -> None:
        with client.websocket_connect("/ws/telemetry") as ws:
            frame = ws.receive_bytes()
            assert b'"type":"snapshot"' in frame

    def test_a_client_disconnecting_does_not_affect_the_bridge(self, client: Any) -> None:
        for _ in range(5):
            with client.websocket_connect("/ws/telemetry") as ws:
                ws.receive_bytes()
        assert client.app.state.telemetry.client_count == 0

    # ── the one write ────────────────────────────────────────────────────────

    def test_panic_engages_the_daily_lock(self, client: Any) -> None:
        assert not client.app.state.daily_lock.is_engaged()

        body = client.post("/api/panic").json()
        assert body["engaged"] is True
        assert client.app.state.daily_lock.is_engaged()

    def test_panic_is_honest_about_what_it_does_not_do(self, client: Any) -> None:
        """§7.2 — a button that implied it flattened would be a lie."""
        body = client.post("/api/panic").json()
        assert "flatten" in body["does_not"]
        assert "new entries" in body["blocks"]

    def test_panic_is_idempotent(self, client: Any) -> None:
        assert client.post("/api/panic").json()["engaged"]
        assert client.post("/api/panic").json()["engaged"]

    def test_there_is_no_unpanic_endpoint(self, client: Any) -> None:
        """Resuming after an emergency stop is a deliberate act on the filesystem."""
        routes = {getattr(route, "path", "") for route in client.app.routes}
        assert not any("unpanic" in path or "resume" in path for path in routes)

    def test_the_ui_exposes_no_order_capability(self, client: Any) -> None:
        """CLAUDE.md §7.3 — the UI process must not be able to place an order."""
        routes = {getattr(route, "path", "") for route in client.app.routes}
        for forbidden in ("order", "place", "square", "flatten", "cancel", "modify"):
            offenders = [p for p in routes if forbidden in p.lower() and "postback" not in p]
            assert not offenders, f"{forbidden!r} reachable from the UI: {offenders}"

    def test_a_failed_panic_reports_failure(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator must never be told the market is closed to them when it is not."""

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("read-only filesystem")

        # DailyLock is a frozen, slotted dataclass, so the method is patched on the class.
        monkeypatch.setattr(DailyLock, "engage", explode)
        response = client.post("/api/panic")
        assert response.status_code == 500
        body = response.json()
        assert body["engaged"] is False
        assert "STILL PERMITTED" in body["impact"]

    # ── postback webhook ─────────────────────────────────────────────────────

    def test_the_postback_endpoint_accepts_a_fill(self, client: Any) -> None:
        body = client.post("/api/postback", json=_row("W1", side="BUY", filled=10)).json()
        assert body["accepted"] is True
        assert client.app.state.listener.net_quantity("RELIANCE") == 10

    def test_a_malformed_postback_answers_200(self, client: Any) -> None:
        """A broker that receives an error retries; a retry storm helps nobody."""
        response = client.post("/api/postback", json={"garbage": True})
        assert response.status_code == 200
        assert response.json()["accepted"] is False


# ──────────────────────────────────────────────────────────────────────────────
# The frontend — styling law, CLAUDE.md §7.1, §7.2
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def css() -> str:
    return (STATIC_DIR / "style.css").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def js() -> str:
    return (STATIC_DIR / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


class TestStylingLaw:
    """Asserted mechanically because §7.1 says "mandatory, no substitutions"."""

    @pytest.mark.parametrize(
        ("token", "value"),
        [
            ("--canvas", "#08091a"),
            ("--panel", "rgba(18, 20, 44, 0.55)"),
            ("--panel-brdr", "rgba(120, 160, 255, 0.14)"),
            ("--glass-blur", "18px"),
            ("--neon-cyan", "#00f0ff"),
            ("--neon-lime", "#39ff88"),
            ("--neon-magenta", "#ff2fb9"),
            ("--neon-amber", "#ffb340"),
            ("--text-dim", "#7c86b8"),
            ("--text-bright", "#e8ecff"),
            ("--num-size", "30px"),
        ],
    )
    def test_every_design_token_is_verbatim(self, css: str, token: str, value: str) -> None:
        assert f"{token}:" in css
        assert value in css

    def test_the_canvas_is_the_only_page_background(self, css: str) -> None:
        body_rule = re.search(r"html,\s*body\s*\{[^}]*\}", css)
        assert body_rule is not None
        assert "var(--canvas)" in body_rule.group(0)
        assert "gradient" not in body_rule.group(0)
        assert "url(" not in body_rule.group(0)

    def test_primary_numerals_are_30px_tabular_mono_and_glowing(self, css: str) -> None:
        rule = re.search(r"\.num\s*\{[^}]*\}", css)
        assert rule is not None
        block = rule.group(0)
        assert "var(--num-size)" in block
        assert "tabular-nums" in block
        assert "var(--num-font)" in block
        assert "var(--num-glow)" in block

    def test_panels_are_glassmorphic(self, css: str) -> None:
        rule = re.search(r"\.panel\s*\{[^}]*\}", css)
        assert rule is not None
        block = rule.group(0)
        assert "backdrop-filter: blur(var(--glass-blur))" in block
        assert "var(--panel)" in block
        assert "1px solid var(--panel-brdr)" in block
        assert "border-radius: 14px" in block

    def test_reduced_motion_is_respected(self, css: str) -> None:
        assert "prefers-reduced-motion" in css

    def test_nothing_blinks_faster_than_1hz(self, css: str) -> None:
        durations = [float(d) for d in re.findall(r"animation:[^;]*?([\d.]+)s", css)]
        assert durations, "expected at least one animation to check"
        assert all(d >= 1.0 for d in durations)

    def test_the_stale_threshold_matches_the_risk_engine(self, js: str) -> None:
        assert "const STALE_MS = 2000;" in js

    def test_staleness_is_measured_on_the_browsers_own_clock(self, js: str) -> None:
        """Comparing a server timestamp to Date.now() would conflate skew with staleness."""
        assert "performance.now()" in js
        assert "Date.now() - " not in js

    def test_staleness_is_repainted_on_a_timer_not_only_on_arrival(self, js: str) -> None:
        """The interesting case is when messages have STOPPED — no arrival will fire then."""
        assert "setInterval(repaintStaleness" in js

    def test_a_missing_value_renders_as_a_dash_not_a_zero(self, js: str) -> None:
        """Blank and zero look identical on a P&L display."""
        assert "const EM_DASH = '—';" in js
        assert "return EM_DASH;" in js

    def test_the_frontend_loads_fully_offline(self, html: str, js: str) -> None:
        assert not re.search(r'(src|href)\s*=\s*"https?://', html)
        assert "react" not in html.lower() and "tailwind" not in html.lower()
        # No fetch/import against an external origin.
        assert not re.search(r"""(fetch|import)\s*\(?\s*['"]https?://""", js)

    def test_the_killswitch_gauge_is_never_behind_a_tab(self, css: str, html: str) -> None:
        assert "panel--critical" in html
        rule = re.search(r"\.panel--critical\s*\{[^}]*\}", css)
        assert rule is not None
        assert "position: sticky" in rule.group(0)

    def test_wire_content_is_escaped_before_rendering(self, js: str) -> None:
        """Broker-supplied rejection text reaches this display verbatim."""
        assert "function escapeHtml" in js
        assert "escapeHtml(" in js

    def test_the_panic_button_says_what_it_does(self, html: str) -> None:
        assert "BLOCK NEW ENTRIES" in html
        assert "does NOT flatten" in html or "Does NOT flatten" in html
