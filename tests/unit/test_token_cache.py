"""Token cache tests — the rate-limit defence for ``loginByPassword``.

Angel One bans a client that logs in too often, and the supervisor restarts the ingestor
without asking. The cache is what turns a restart from "another login" into "the same
session", so these tests pin every condition under which an entry is adopted or refused:
a stale entry must never be served, a foreign credential pair must never adopt one, and
a corrupt file must degrade to a fresh login rather than block startup.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tachyon.core.clock import IST, ManualClock
from tachyon.core.token_cache import (
    CachedSession,
    credential_hash,
    invalidate_session_cache,
    is_rate_limited_message,
    load_session_cache,
    save_session_cache,
)

API_KEY = "api-key"
CLIENT_CODE = "ABC123"


def _clock(hh: int = 9, mm: int = 15) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 28, hh, mm, tzinfo=IST), mono=1000.0)


def _save(path: Path, *, clock: ManualClock | None = None, **overrides: str) -> None:
    payload = {
        "api_key": API_KEY,
        "client_code": CLIENT_CODE,
        "jwt_token": "jwt-value",
        "refresh_token": "refresh-value",
        "feed_token": "feed-value",
    }
    payload.update(overrides)
    save_session_cache(
        api_key=payload["api_key"],
        client_code=payload["client_code"],
        jwt_token=payload["jwt_token"],
        refresh_token=payload["refresh_token"],
        feed_token=payload["feed_token"],
        path=path,
        clock=clock if clock is not None else _clock(),
    )


class TestRoundTrip:
    def test_a_saved_session_is_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path)

        loaded = load_session_cache(API_KEY, CLIENT_CODE, path=path, clock=_clock())
        assert loaded == CachedSession(
            jwt_token="jwt-value",
            refresh_token="refresh-value",
            feed_token="feed-value",
            client_code=CLIENT_CODE,
            issued_date_ist=_clock().now().date(),
            issued_at_ist=_clock().now(),
        )

    def test_the_file_is_valid_json_with_no_secret_in_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path)
        text = path.read_text(encoding="utf-8")
        assert "jwt-value" in text  # the payload is present...
        assert API_KEY not in text  # ...but the api key is only hashed
        assert credential_hash(API_KEY, CLIENT_CODE) in text


class TestRejection:
    def test_a_missing_file_is_a_miss_not_an_error(self, tmp_path: Path) -> None:
        assert load_session_cache(API_KEY, CLIENT_CODE, path=tmp_path / "absent.json") is None

    def test_a_stale_entry_from_yesterday_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path, clock=_clock())  # issued 2026-08-28

        tomorrow = ManualClock(wall=datetime(2026, 8, 29, 9, 15, tzinfo=IST), mono=2000.0)
        assert load_session_cache(API_KEY, CLIENT_CODE, path=path, clock=tomorrow) is None

    def test_a_different_api_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path)
        assert load_session_cache("other-key", CLIENT_CODE, path=path, clock=_clock()) is None

    def test_a_different_client_code_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path)
        assert load_session_cache(API_KEY, "ZZZ999", path=path, clock=_clock()) is None

    def test_an_empty_jwt_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path, jwt_token="")
        assert load_session_cache(API_KEY, CLIENT_CODE, path=path, clock=_clock()) is None

    def test_a_corrupt_file_degrades_to_a_miss(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_session_cache(API_KEY, CLIENT_CODE, path=path, clock=_clock()) is None

    def test_a_truncated_payload_degrades_to_a_miss(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        path.write_text('{"jwt_token": "only-half"}', encoding="utf-8")
        assert load_session_cache(API_KEY, CLIENT_CODE, path=path, clock=_clock()) is None


class TestInvalidation:
    def test_invalidate_deletes_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "session_token.json"
        _save(path)
        invalidate_session_cache(path=path)
        assert not path.exists()

    def test_invalidate_on_a_missing_file_is_silent(self, tmp_path: Path) -> None:
        invalidate_session_cache(path=tmp_path / "absent.json")  # must not raise


class TestRateLimitDetection:
    def test_the_broker_phrase_matches_case_insensitively(self) -> None:
        assert is_rate_limited_message("Access denied because of exceeding access rate")
        assert is_rate_limited_message("login: EXCEEDING ACCESS RATE [AB1016]")

    def test_a_genuine_rejection_does_not_match(self) -> None:
        assert not is_rate_limited_message("Invalid password")
        assert not is_rate_limited_message("TOTP expired")
        assert not is_rate_limited_message("")
