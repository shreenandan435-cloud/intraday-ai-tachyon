"""Live-socket tests for the feed client — CLAUDE.md §2, §5.

These run a real WebSocket server on loopback rather than mocking the transport. The
requirement under test — "a dropped WebSocket must not crash the publisher; it must reconnect
and resume" — is precisely the behaviour a mock cannot demonstrate, because the interesting
part is what the library does when a real connection dies mid-read.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import pytest
from websockets.asyncio.server import Server, ServerConnection, serve

from tachyon.ingestion.decoder import ExchangeType, SubscriptionMode, decode_snap_quote
from tachyon.ingestion.ws_client import (
    FeedCredentials,
    SmartApiFeedClient,
    TokenSubscription,
)

from ..unit.test_ingestion import build_snap_quote

CREDENTIALS = FeedCredentials(api_key="test-key", client_code="TESTCODE", feed_token="test-tok")
SUBSCRIPTIONS = [TokenSubscription(exchange_type=ExchangeType.NSE_CM, tokens=("2885",))]


async def _serve(
    handler: Callable[[ServerConnection], object],
) -> AsyncIterator[tuple[Server, str]]:
    server = await serve(handler, "127.0.0.1", 0)  # type: ignore[arg-type]
    port = server.sockets[0].getsockname()[1]
    try:
        yield server, f"ws://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


async def _wait_for(predicate: Callable[[], bool], limit_seconds: float = 10.0) -> bool:
    """Poll until ``predicate`` holds or the limit expires."""
    deadline = asyncio.get_running_loop().time() + limit_seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


class TestFeedClientAgainstRealSocket:
    async def test_connects_subscribes_and_streams(self) -> None:
        frames: list[bytes] = []
        subscriptions: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            subscriptions.append(json.loads(await connection.recv()))
            await connection.send(build_snap_quote(token="2885", ltp_paise=245_675))
            await connection.wait_closed()

        async for _server, url in _serve(handler):
            client = SmartApiFeedClient(
                CREDENTIALS, SUBSCRIPTIONS, on_binary=frames.append, url=url
            )
            task = asyncio.create_task(client.run())
            try:
                assert await _wait_for(lambda: len(frames) >= 1), "no binary frame received"
            finally:
                await client.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert subscriptions[0]["action"] == 1
        assert subscriptions[0]["params"]["mode"] == int(SubscriptionMode.SNAP_QUOTE)  # type: ignore[index]

        quote = decode_snap_quote(frames[0])
        assert quote.token == "2885"
        assert quote.ltp == pytest.approx(2456.75)

    async def test_sends_authentication_headers(self) -> None:
        captured: dict[str, str] = {}

        async def handler(connection: ServerConnection) -> None:
            # Headers is case-insensitive; dict() of it is not, so read by name.
            headers = connection.request.headers  # type: ignore[union-attr]
            for name in ("Authorization", "x-api-key", "x-client-code", "x-feed-token"):
                value = headers.get(name)
                if value is not None:
                    captured[name] = value
            await connection.recv()
            await connection.wait_closed()

        async for _server, url in _serve(handler):
            client = SmartApiFeedClient(
                CREDENTIALS, SUBSCRIPTIONS, on_binary=lambda _f: None, url=url
            )
            task = asyncio.create_task(client.run())
            try:
                assert await _wait_for(lambda: bool(captured))
            finally:
                await client.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert captured["x-api-key"] == "test-key"
        assert captured["x-client-code"] == "TESTCODE"
        assert captured["x-feed-token"] == "test-tok"
        assert captured["Authorization"] == "test-tok"

    async def test_reconnects_and_resumes_after_a_drop(self) -> None:
        """The core resilience requirement: a dropped socket must not end the process."""
        frames: list[bytes] = []
        connection_count = 0

        async def handler(connection: ServerConnection) -> None:
            nonlocal connection_count
            connection_count += 1
            mine = connection_count
            await connection.recv()  # subscription
            await connection.send(build_snap_quote(token="2885", ltp_paise=100_000 * mine))
            if mine == 1:
                await connection.close()  # yank the first connection
                return
            await connection.wait_closed()

        async for _server, url in _serve(handler):
            client = SmartApiFeedClient(
                CREDENTIALS,
                SUBSCRIPTIONS,
                on_binary=frames.append,
                url=url,
                backoff_seconds=(0.05,),
                max_attempts=10,
            )
            task = asyncio.create_task(client.run())
            try:
                assert await _wait_for(lambda: len(frames) >= 2), (
                    f"expected a frame from each connection, got {len(frames)}"
                )
                assert not task.done(), "run() must survive the drop"
            finally:
                await client.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert connection_count >= 2, "client must have reconnected"
        assert client.stats.reconnects >= 1
        # Data resumed after the reconnect, and the second connection resubscribed.
        assert decode_snap_quote(frames[0]).ltp == pytest.approx(1000.0)
        assert decode_snap_quote(frames[1]).ltp == pytest.approx(2000.0)

    async def test_gives_up_after_max_attempts(self) -> None:
        """A permanently unreachable endpoint must surface, not spin forever."""
        client = SmartApiFeedClient(
            CREDENTIALS,
            SUBSCRIPTIONS,
            on_binary=lambda _f: None,
            url="ws://127.0.0.1:1",  # nothing listens here
            backoff_seconds=(0.01,),
            max_attempts=3,
        )
        with pytest.raises(ConnectionError, match="3 consecutive"):
            await client.run()

    async def test_stop_ends_the_run_loop_promptly(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.recv()
            await connection.wait_closed()

        async for _server, url in _serve(handler):
            client = SmartApiFeedClient(
                CREDENTIALS, SUBSCRIPTIONS, on_binary=lambda _f: None, url=url
            )
            task = asyncio.create_task(client.run())
            assert await _wait_for(lambda c=client: c.subscribed)  # type: ignore[misc]

            await client.stop()
            await asyncio.wait_for(task, timeout=5.0)
            assert task.done()
            assert not client.subscribed

    async def test_handler_exception_does_not_drop_the_subscription(self) -> None:
        """A decoder bug must not cost us market data."""
        calls = 0

        def exploding(_payload: bytes) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("boom")

        async def handler(connection: ServerConnection) -> None:
            await connection.recv()
            for _ in range(3):
                await connection.send(build_snap_quote(token="2885"))
                await asyncio.sleep(0.05)
            await connection.wait_closed()

        async for _server, url in _serve(handler):
            client = SmartApiFeedClient(CREDENTIALS, SUBSCRIPTIONS, on_binary=exploding, url=url)
            task = asyncio.create_task(client.run())
            try:
                assert await _wait_for(lambda: calls >= 3)
                assert client.subscribed, "subscription must survive handler failures"
                assert client.stats.handler_errors == 3
                assert client.stats.reconnects == 0
            finally:
                await client.stop()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
