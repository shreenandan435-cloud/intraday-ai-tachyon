"""Machine edits to ``settings.yaml`` — CLAUDE.md §1.3, §8.1.

Every test writes to ``tmp_path``. Nothing in this file touches the operator's real
``config/settings.yaml``, with one deliberate exception: :class:`TestAgainstTheShippedFile`
*reads* it (never writes) to prove the surgery can still find its landmarks in the file that
actually ships. That mirrors ``TestConfig`` in ``test_core.py`` and exists for the same reason —
a writer validated only against a fixture is a writer that has never met the real file.

The theme running through these tests is that a **failed write must leave the file untouched**.
This module edits the two settings that decide how much a session may lose and which
instruments it may touch; a half-applied edit at 09:10 is worse than no edit at all.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from tachyon.core.config import WatchlistItem
from tachyon.core.config_writer import (
    ConfigWriteError,
    format_decimal,
    render_settings,
    write_settings,
)
from tests.conftest import REAL_SETTINGS_YAML

STAMP = datetime.fromisoformat("2026-08-12T09:09:14+05:30")

#: A pinned stand-in shaped like the real file: comments above and inside every block, a
#: `capital` section with the key we edit, a `watchlist` we replace, and blocks on both sides
#: of it that must survive untouched.
FIXTURE = """\
# INTRADAY AI TACHYON - tunables
#
# WHAT DOES NOT BELONG HERE: anything in CLAUDE.md section 1.

session:
  timezone: Asia/Kolkata
  watchlist_only: true          # trailing comments must survive

capital:
  # The daily loss limit is DERIVED FROM THIS. Read the arithmetic before changing it.
  #   daily loss limit = session_budget_inr x max_daily_drawdown_pct / 100
  session_budget_inr: 50000.0
  max_daily_drawdown_pct: 2.0
  per_trade_risk_pct: 0.4

watchlist:
  # tick_size values are taken from the scrip master, verified 2026-08-11.
  - {symbol: RELIANCE,  token: "2885", exchange: NSE, tick_size: 0.10, lot_size: 1}
  - {symbol: INFY,      token: "1594", exchange: NSE, tick_size: 0.10, lot_size: 1}
  # TATAMOTORS: REMOVED -- token 3456 is now TMPV-EQ in the master.

positions:
  max_concurrent: 2
  max_per_symbol: 1

logging:
  json_lines: true
  directory: logs
"""


def _settings_file(tmp_path: Path, text: str = FIXTURE) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _items(*symbols: str) -> tuple[WatchlistItem, ...]:
    return tuple(
        WatchlistItem(
            symbol=symbol,
            token=str(1000 + index),
            exchange="NSE",
            tick_size=Decimal("0.05"),
            lot_size=1,
        )
        for index, symbol in enumerate(symbols)
    )


# ──────────────────────────────────────────────────────────────────────────────


class TestFormatDecimal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("50000", "50000.00"),
            ("50000.0", "50000.00"),
            ("0.10", "0.10"),
            ("0.05", "0.05"),
            ("1234.5", "1234.50"),
        ],
    )
    def test_it_keeps_two_decimals_where_two_are_exact(self, value: str, expected: str) -> None:
        assert format_decimal(Decimal(value)) == expected

    def test_it_keeps_more_precision_when_paise_would_lose_some(self) -> None:
        assert format_decimal(Decimal("0.001")) == "0.001"

    def test_it_never_emits_exponent_notation(self) -> None:
        """`str(Decimal('1E+5'))` is '1E+5', which YAML reads as the *string* '1E+5', not a
        number — and pydantic would then reject the config at the next boot."""
        assert format_decimal(Decimal("1E+5")) == "100000.00"


class TestBudgetEdit:
    def test_it_replaces_the_value_and_nothing_else(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        write_settings(
            path=path, session_budget_inr=Decimal("25000"), generated_at=STAMP, backup=False
        )
        text = path.read_text(encoding="utf-8")

        assert "session_budget_inr: 25000.00" in text
        assert "50000.0" not in text
        assert yaml.safe_load(text)["capital"]["max_daily_drawdown_pct"] == 2.0

    def test_the_surrounding_comments_survive(self, tmp_path: Path) -> None:
        """The arithmetic in those comments is how an operator knows what the number means.
        A yaml.safe_dump round trip would silently delete every one of them."""
        path = _settings_file(tmp_path)

        write_settings(
            path=path, session_budget_inr=Decimal("25000"), generated_at=STAMP, backup=False
        )
        text = path.read_text(encoding="utf-8")

        assert "# The daily loss limit is DERIVED FROM THIS." in text
        assert "#   daily loss limit = session_budget_inr x max_daily_drawdown_pct / 100" in text
        assert "watchlist_only: true          # trailing comments must survive" in text
        assert "# TATAMOTORS: REMOVED" in text

    def test_the_watchlist_is_untouched_when_only_the_budget_is_given(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        report = write_settings(
            path=path, session_budget_inr=Decimal("25000"), generated_at=STAMP, backup=False
        )

        assert report.symbols is None
        assert not report.touched_watchlist
        assert [e["symbol"] for e in yaml.safe_load(path.read_text())["watchlist"]] == [
            "RELIANCE",
            "INFY",
        ]

    def test_a_missing_capital_section_is_refused_rather_than_invented(
        self, tmp_path: Path
    ) -> None:
        path = _settings_file(tmp_path, "session:\n  timezone: Asia/Kolkata\n")
        before = path.read_text(encoding="utf-8")

        with pytest.raises(ConfigWriteError, match="no top-level 'capital:' key"):
            write_settings(path=path, session_budget_inr=Decimal("1"), generated_at=STAMP)

        assert path.read_text(encoding="utf-8") == before

    def test_a_missing_key_inside_capital_is_refused(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path, "capital:\n  max_daily_drawdown_pct: 2.0\n")

        with pytest.raises(ConfigWriteError, match="no 'session_budget_inr:' line"):
            write_settings(path=path, session_budget_inr=Decimal("1"), generated_at=STAMP)

    def test_a_negative_budget_is_refused(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        with pytest.raises(ConfigWriteError, match="must not be negative"):
            write_settings(path=path, session_budget_inr=Decimal("-1"), generated_at=STAMP)

    def test_zero_is_permitted_because_it_restores_the_constitutional_limits(
        self, tmp_path: Path
    ) -> None:
        """`session_budget_inr: 0` is the shipped default and means "use section 1's Rs.500".
        Refusing it would leave an operator no way to hand the limits back."""
        path = _settings_file(tmp_path)

        write_settings(path=path, session_budget_inr=Decimal("0"), generated_at=STAMP, backup=False)

        assert yaml.safe_load(path.read_text())["capital"]["session_budget_inr"] == 0


class TestWatchlistEdit:
    def test_it_replaces_the_whole_block(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        write_settings(path=path, watchlist=_items("AAA", "BBB"), generated_at=STAMP, backup=False)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert [e["symbol"] for e in loaded["watchlist"]] == ["AAA", "BBB"]
        assert loaded["watchlist"][0]["token"] == "1000"
        assert loaded["watchlist"][0]["tick_size"] == 0.05

    def test_tokens_stay_strings_through_the_round_trip(self, tmp_path: Path) -> None:
        """An unquoted 2885 loads as an int and the WebSocket subscription then carries the
        wrong type — CLAUDE.md 2.3 notes that TokenSubscription rejects a non-string token
        precisely because it fails silently at the broker."""
        path = _settings_file(tmp_path)

        write_settings(path=path, watchlist=_items("AAA"), generated_at=STAMP, backup=False)

        assert isinstance(yaml.safe_load(path.read_text())["watchlist"][0]["token"], str)

    def test_the_blocks_on_either_side_are_untouched(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        write_settings(path=path, watchlist=_items("AAA"), generated_at=STAMP, backup=False)
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)

        assert loaded["capital"]["session_budget_inr"] == 50000.0
        assert loaded["positions"]["max_concurrent"] == 2
        assert loaded["logging"]["directory"] == "logs"
        assert "# INTRADAY AI TACHYON - tunables" in text

    def test_the_generated_block_says_it_is_generated(self, tmp_path: Path) -> None:
        """A hand-written watchlist and a machine-written one are indistinguishable in a diff.
        The operator has to be able to tell at a glance, and to know how to get the old one."""
        path = _settings_file(tmp_path)

        write_settings(
            path=path,
            watchlist=_items("AAA"),
            generated_at=STAMP,
            provenance="top 1 by |gap%|",
            backup=False,
        )
        text = path.read_text(encoding="utf-8")

        assert "# GENERATED 2026-08-12T09:09:14+05:30" in text
        assert "# Source: top 1 by |gap%|" in text
        assert "settings.yaml.bak" in text

    def test_an_empty_watchlist_is_refused(self, tmp_path: Path) -> None:
        """The scanner's contract on a failed scan is to pass `None` and keep the old list. An
        empty tuple would mean "trade nothing", which section 8.1 makes indistinguishable from
        a broken session."""
        path = _settings_file(tmp_path)
        before = path.read_text(encoding="utf-8")

        with pytest.raises(ConfigWriteError, match="refusing to write an empty watchlist"):
            write_settings(path=path, watchlist=(), generated_at=STAMP)

        assert path.read_text(encoding="utf-8") == before

    def test_rewriting_twice_is_stable(self, tmp_path: Path) -> None:
        """The block must not accumulate headers or eat the blank line before `positions:`."""
        path = _settings_file(tmp_path)

        write_settings(path=path, watchlist=_items("AAA"), generated_at=STAMP, backup=False)
        once = path.read_text(encoding="utf-8")
        write_settings(path=path, watchlist=_items("AAA"), generated_at=STAMP, backup=False)
        twice = path.read_text(encoding="utf-8")

        assert once == twice
        assert once.count("# GENERATED") == 1

    def test_a_duplicate_symbol_is_refused(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)
        duplicated = (*_items("AAA"), *_items("AAA"))

        with pytest.raises(ConfigWriteError, match="duplicate symbols"):
            write_settings(path=path, watchlist=duplicated, generated_at=STAMP)


class TestBothAtOnce:
    def test_one_call_writes_the_budget_and_the_watchlist(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        report = write_settings(
            path=path,
            session_budget_inr=Decimal("75000"),
            watchlist=_items("AAA", "BBB", "CCC", "DDD"),
            generated_at=STAMP,
            backup=False,
        )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert report.touched_budget and report.touched_watchlist
        assert loaded["capital"]["session_budget_inr"] == 75000.0
        assert len(loaded["watchlist"]) == 4

    def test_the_result_is_a_config_the_models_accept(self, tmp_path: Path) -> None:
        """The check that makes the rest safe: the rendered file is validated against the real
        pydantic models before it lands, so a config that would fail the next boot never
        becomes the config on disk."""
        path = _settings_file(tmp_path)

        write_settings(
            path=path,
            session_budget_inr=Decimal("50000"),
            watchlist=_items("AAA"),
            generated_at=STAMP,
            backup=False,
        )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert WatchlistItem(**loaded["watchlist"][0]).symbol == "AAA"

    def test_nothing_to_write_is_refused(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        with pytest.raises(ConfigWriteError, match="nothing to write"):
            write_settings(path=path, generated_at=STAMP)


class TestFailSafety:
    def test_a_protected_constant_in_the_file_refuses_the_write(self, tmp_path: Path) -> None:
        """The same guard config.py applies at boot, applied before the file exists. A file
        naming DAILY_LOSS_LIMIT_INR would refuse the next boot; catching it here means the
        operator learns at 09:09 instead of at 09:14."""
        path = _settings_file(tmp_path, FIXTURE + "\nDAILY_LOSS_LIMIT_INR: 5000\n")
        before = path.read_text(encoding="utf-8")

        with pytest.raises(ConfigWriteError, match="breaks CLAUDE.md"):
            write_settings(path=path, session_budget_inr=Decimal("1000"), generated_at=STAMP)

        assert path.read_text(encoding="utf-8") == before

    def test_an_invalid_capital_section_refuses_the_write(self, tmp_path: Path) -> None:
        """per_trade_risk_pct above max_daily_drawdown_pct is arithmetic that cannot be
        honoured (section 1.3), so CapitalSettings rejects it — and so does this."""
        broken = FIXTURE.replace("per_trade_risk_pct: 0.4", "per_trade_risk_pct: 9.0")
        path = _settings_file(tmp_path, broken)
        before = path.read_text(encoding="utf-8")

        with pytest.raises(ConfigWriteError, match="'capital' section is invalid"):
            write_settings(path=path, session_budget_inr=Decimal("1000"), generated_at=STAMP)

        assert path.read_text(encoding="utf-8") == before

    def test_a_pre_existing_invalid_watchlist_refuses_the_budget_write(
        self, tmp_path: Path
    ) -> None:
        """We validate the whole rendered file, not only the region edited. Writing a budget
        into a file whose watchlist will not load would produce a config that boots to nothing."""
        broken = FIXTURE.replace("tick_size: 0.10, lot_size: 1}", "tick_size: 0, lot_size: 1}")
        path = _settings_file(tmp_path, broken)

        with pytest.raises(ConfigWriteError, match="'watchlist' is invalid"):
            write_settings(path=path, session_budget_inr=Decimal("1000"), generated_at=STAMP)

    def test_a_duplicated_top_level_key_is_ambiguous_and_refused(self, tmp_path: Path) -> None:
        """Two `watchlist:` keys mean the last one wins in YAML but the first one is what line
        surgery would find. Editing the wrong one produces a valid file with the old symbols."""
        path = _settings_file(tmp_path, FIXTURE + '\nwatchlist:\n  - {symbol: X, token: "9"}\n')

        with pytest.raises(ConfigWriteError, match="appears 2 times"):
            write_settings(path=path, watchlist=_items("AAA"), generated_at=STAMP)

    def test_an_unreadable_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigWriteError, match="cannot read"):
            write_settings(
                path=tmp_path / "absent.yaml",
                session_budget_inr=Decimal("1"),
                generated_at=STAMP,
            )

    def test_no_temp_file_is_left_behind(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)

        write_settings(path=path, session_budget_inr=Decimal("1"), generated_at=STAMP)

        assert not (tmp_path / "settings.yaml.tmp").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "settings.yaml",
            "settings.yaml.bak",
        ]


class TestBackupAndDryRun:
    def test_the_previous_version_is_kept_verbatim(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)
        original = path.read_text(encoding="utf-8")

        report = write_settings(path=path, session_budget_inr=Decimal("999"), generated_at=STAMP)

        assert report.backup_path is not None
        assert report.backup_path.read_text(encoding="utf-8") == original

    def test_dry_run_validates_but_writes_nothing(self, tmp_path: Path) -> None:
        path = _settings_file(tmp_path)
        original = path.read_text(encoding="utf-8")

        report = write_settings(
            path=path,
            session_budget_inr=Decimal("999"),
            watchlist=_items("AAA"),
            generated_at=STAMP,
            dry_run=True,
        )

        assert path.read_text(encoding="utf-8") == original
        assert report.backup_path is None
        assert report.symbols == ("AAA",)
        assert not (tmp_path / "settings.yaml.bak").exists()

    def test_dry_run_still_raises_on_a_config_it_could_not_write(self, tmp_path: Path) -> None:
        """That is the point of --dry-run: it must fail for the same reasons the real write
        would, or it proves nothing."""
        path = _settings_file(tmp_path, "session:\n  timezone: Asia/Kolkata\n")

        with pytest.raises(ConfigWriteError):
            write_settings(
                path=path, session_budget_inr=Decimal("1"), generated_at=STAMP, dry_run=True
            )


class TestRenderIsPure:
    def test_render_settings_touches_no_disk(self) -> None:
        rendered = render_settings(
            FIXTURE,
            session_budget_inr=Decimal("1234"),
            watchlist=_items("AAA"),
            generated_at=STAMP,
        )

        assert "session_budget_inr: 1234.00" in rendered
        assert FIXTURE.count("RELIANCE") == 1, "the input must not be mutated"
        assert "RELIANCE" not in rendered

    def test_a_trailing_newline_is_preserved(self) -> None:
        assert render_settings(
            FIXTURE, session_budget_inr=Decimal("1"), generated_at=STAMP
        ).endswith("\n")

    def test_a_file_without_a_trailing_newline_does_not_gain_one(self) -> None:
        assert not render_settings(
            FIXTURE.rstrip("\n"), session_budget_inr=Decimal("1"), generated_at=STAMP
        ).endswith("\n")


class TestAgainstTheShippedFile:
    """Read-only checks against the real ``config/settings.yaml``.

    The fixture above is a stand-in, and a stand-in only proves the writer works on itself.
    These prove the surgery can still find its landmarks in the file that actually ships —
    which is the file it will be pointed at tomorrow morning.
    """

    def test_the_shipped_file_still_has_both_regions(self, tmp_path: Path) -> None:
        copy = tmp_path / "settings.yaml"
        copy.write_text(REAL_SETTINGS_YAML.read_text(encoding="utf-8"), encoding="utf-8")

        write_settings(
            path=copy,
            session_budget_inr=Decimal("60000"),
            watchlist=_items("AAA", "BBB", "CCC", "DDD"),
            generated_at=STAMP,
            backup=False,
        )
        loaded = yaml.safe_load(copy.read_text(encoding="utf-8"))

        assert loaded["capital"]["session_budget_inr"] == 60000.0
        assert [e["symbol"] for e in loaded["watchlist"]] == ["AAA", "BBB", "CCC", "DDD"]

    def test_editing_the_shipped_file_keeps_every_other_section(self, tmp_path: Path) -> None:
        original = REAL_SETTINGS_YAML.read_text(encoding="utf-8")
        copy = tmp_path / "settings.yaml"
        copy.write_text(original, encoding="utf-8")

        write_settings(
            path=copy,
            session_budget_inr=Decimal("60000"),
            watchlist=_items("AAA"),
            generated_at=STAMP,
            backup=False,
        )
        before = yaml.safe_load(original)
        after = yaml.safe_load(copy.read_text(encoding="utf-8"))

        untouched = set(before) - {"capital", "watchlist"}
        assert untouched, "the shipped file should carry more than the two edited sections"
        for section in untouched:
            assert after[section] == before[section], f"{section} was modified"
