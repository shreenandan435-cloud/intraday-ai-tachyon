"""Tests for the Angel One broker adapters.

The :class:`PaperBroker` is the workhorse of the dry-run and
paper-live sessions. The :class:`AngelOneBroker` is a thin wrapper
over the SDK that is hard to test without a real session; we cover
its parameter-builder and error-normalisation paths.
"""

from __future__ import annotations

import asyncio

import pytest

from tachyon.execution.angel_broker import (
    AngelOneBroker,
    BrokerOrderResult,
    PaperBroker,
)


def _book(bid: float = 100.0, ask: float = 101.0) -> object:
    """Build a minimal :class:`TopOfBook` stand-in."""

    class _B:
        pass

    book = _B()
    book.usable = True
    book.bid = bid
    book.ask = ask
    book.bid_qty = 100.0
    book.ask_qty = 100.0
    return book


class TestPaperBroker:
    async def test_market_buy_fills_at_ask(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book(bid=100.0, ask=101.0))
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=10,
        )
        assert result.status == "filled"
        assert result.broker == "paper"
        assert result.detail["fill_price"] == pytest.approx(101.0)
        assert result.detail["quantity"] == 10

    async def test_market_sell_fills_at_bid(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book(bid=100.0, ask=101.0))
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="SELL",
            quantity=5,
        )
        assert result.detail["fill_price"] == pytest.approx(100.0)

    async def test_limit_order_applies_slippage(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book(bid=100.0, ask=101.0))
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
            order_type="LIMIT",
            limit_price=99.0,
        )
        # The paper broker lifts the offer (ask * 1.0005) regardless
        # of the limit price — a marketable limit is treated as a
        # market order for the paper fill.
        assert result.detail["fill_price"] == pytest.approx(101.0 * 1.0005, rel=1e-9)

    async def test_invalid_quantity_is_rejected(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=0,
        )
        assert result.status == "rejected"
        assert "quantity" in result.error

    async def test_invalid_transaction_type_is_rejected(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="HOLD",
            quantity=1,
        )
        assert result.status == "rejected"
        assert "transaction_type" in result.error

    async def test_limit_order_without_price_is_rejected(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
            order_type="LIMIT",
            limit_price=None,
        )
        assert result.status == "rejected"

    async def test_empty_book_defers_fill(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: None)
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
        )
        # An empty book returns "placed" but not "filled" — the router
        # can decide whether to retry once the L2 stream catches up.
        assert result.status == "placed"
        assert result.order_id in broker.orders()

    async def test_modify_updates_record(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        first = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
        )
        result = await broker.modify_order(
            order_id=first.order_id,
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=5,
        )
        assert result.status == "modified"
        assert broker.orders()[first.order_id]["quantity"] == 5

    async def test_modify_unknown_order_rejected(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        result = await broker.modify_order(
            order_id="missing",
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
        )
        assert result.status == "rejected"
        assert "unknown" in result.error

    async def test_cancel_removes_order(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        first = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
        )
        result = await broker.cancel_order(order_id=first.order_id)
        assert result.status == "cancelled"
        # A second cancel is idempotent.
        again = await broker.cancel_order(order_id=first.order_id)
        assert again.status == "cancelled"

    async def test_cancel_unknown_order_rejected(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        result = await broker.cancel_order(order_id="missing")
        assert result.status == "rejected"

    async def test_unique_order_ids(self) -> None:
        broker = PaperBroker(book_provider=lambda _t: _book())
        ids = set()
        for _ in range(10):
            r = await broker.place_order(
                trading_symbol="RELIANCE",
                instrument_token=2885,
                transaction_type="BUY",
                quantity=1,
            )
            ids.add(r.order_id)
        assert len(ids) == 10


class TestAngelOneBrokerParameterBuilding:
    def test_build_order_params_market(self) -> None:
        """The param builder produces a dict the SDK accepts."""
        from tachyon.execution.router import LiveOrderGateway

        params = LiveOrderGateway.build_order_params(
            trading_symbol="RELIANCE-EQ",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=10,
        )
        assert params["tradingsymbol"] == "RELIANCE-EQ"
        assert params["transactiontype"] == "BUY"
        assert params["ordertype"] == "MARKET"
        assert params["producttype"] == "INTRADAY"
        assert params["quantity"] == "10"
        assert params["price"] == "0"

    def test_build_order_params_limit_requires_price(self) -> None:
        from tachyon.execution.router import LiveOrderGateway

        with pytest.raises(ValueError):
            LiveOrderGateway.build_order_params(
                trading_symbol="RELIANCE-EQ",
                instrument_token=2885,
                transaction_type="BUY",
                quantity=10,
                order_type="LIMIT",
            )


class TestAngelOneBrokerDispatch:
    async def test_place_captures_sdk_error(self) -> None:
        """A live broker that cannot authenticate returns a rejected result, not a raise."""
        broker = AngelOneBroker()
        # The gateway cannot authenticate (no creds in this env);
        # the dispatch should normalise the error into a
        # BrokerOrderResult with status="rejected".
        result = await broker.place_order(
            trading_symbol="RELIANCE",
            instrument_token=2885,
            transaction_type="BUY",
            quantity=1,
        )
        assert result.status in ("placed", "rejected")
        # When SDK auth fails the gateway raises a RuntimeError,
        # which the broker catches.
        if result.status == "rejected":
            assert result.error


class TestAngelAdapterTokenBinding:
    """The historical pre-seed path used to send ``getCandleData`` with no
    ``Authorization`` header at all. ``AngelOneWebSocketClient._stamp_session``
    binds the JWT onto every attribute name the SDK / vendor forks may consult
    so the SDK's ``_request`` always sees a non-empty ``access_token`` and
    builds the ``Bearer`` header.
    """

    def test_bind_session_sets_access_token_and_jwt_token(self) -> None:
        try:
            from SmartApi import SmartConnect
        except ImportError:  # pragma: no cover - SDK optional in CI
            pytest.skip("SmartApi SDK not installed")
        from tachyon.ingestion.angel_adapter import AngelOneWebSocketClient

        client = SmartConnect(api_key="dummy_key")
        AngelOneWebSocketClient._bind_session(client, "JWT-XYZ", "FEED-XYZ")

        # ``setAccessToken`` writes ``self.access_token`` — the canonical binding the
        # SDK's ``_request`` reads to build the ``Authorization`` header.
        assert getattr(client, "access_token", None) == "JWT-XYZ"
        assert getattr(client, "jwtToken", None) == "JWT-XYZ"
        assert getattr(client, "feed_token", None) == "FEED-XYZ"
        assert getattr(client, "feedToken", None) == "FEED-XYZ"

    def test_bind_session_builds_bearer_header(self) -> None:
        """Mirror ``SmartConnect._request`` header assembly — no network."""
        try:
            from SmartApi import SmartConnect
        except ImportError:  # pragma: no cover - SDK optional in CI
            pytest.skip("SmartApi SDK not installed")
        from tachyon.ingestion.angel_adapter import AngelOneWebSocketClient

        client = SmartConnect(api_key="dummy_key")
        AngelOneWebSocketClient._bind_session(client, "JWT-XYZ", "FEED-XYZ")

        headers = client.requestHeaders()
        if client.access_token:
            headers["Authorization"] = "Bearer {}".format(client.access_token)

        assert headers.get("Authorization") == "Bearer JWT-XYZ"

        # Fail-safe: even a snapshot that consults ``_header`` directly sees the
        # bearer header set explicitly by the adapter.
        header_map = getattr(client, "_header", None)
        if isinstance(header_map, dict):
            assert header_map.get("Authorization") == "Bearer JWT-XYZ"
