"""Machine edits to ``config/settings.yaml`` — CLAUDE.md §1.3, §8.1.

Why this is not ``yaml.safe_dump``
----------------------------------
``settings.yaml`` is half configuration and half documentation. Its comments carry the §1.3
arithmetic that turns a capital number into a loss limit, the reason TATAMOTORS was removed
rather than repointed, and the provenance of every tick size. A load/dump round trip through
PyYAML deletes all of it and reorders the rest, so the first machine-written config would
silently destroy the notes that explain what the numbers mean. ``ruamel.yaml`` could round-trip
them, but a new third-party dependency needs justification (§8) and this needs to touch exactly
two regions of one file.

So the edit is **line surgery**: find the region, replace the region, leave every other byte
alone. A file we cannot find the region in is not edited at all.

The safety contract
-------------------
This module writes the file that decides how much money a session may lose (``capital``) and
which instruments it may touch at all (``watchlist``, §8.1). Four rules follow from that:

1. **Validate before committing, not after.** The new text is rendered in memory, parsed, run
   through the same protected-constant guard :mod:`tachyon.core.config` applies at boot, and
   validated against the real pydantic models. Only then does it reach the disk. A config that
   would fail the next boot never becomes the config on disk.
2. **Confirm the edit landed.** Validation re-reads the values it just wrote and compares them
   to what was asked for. Line surgery that matched the wrong line would otherwise produce a
   perfectly valid file containing the old numbers.
3. **Atomic.** Rendered to a temporary file beside the target, then ``os.replace``. There is no
   window in which ``settings.yaml`` is half-written — a truncated file at 09:10 would take the
   session down.
4. **The previous version is kept**, at ``settings.yaml.bak``. The file is in git, so this is
   belt and braces; it is also the thing an operator reaches for at 09:12, which is not the
   moment to be learning ``git checkout``.

What it will not do
-------------------
It will not write a value it was not given, and it will not create a section that is missing.
An absent ``capital:`` or ``watchlist:`` key means this is not the file we think it is, and the
correct response to that is to stop, not to invent structure.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import yaml

from tachyon.core.config import CapitalSettings, WatchlistItem, _reject_protected
from tachyon.core.constants import SETTINGS_YAML
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

#: A key at column zero — the boundary that ends a top-level block.
_TOP_LEVEL_KEY: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:")

#: Suffix for the retained previous version.
BACKUP_SUFFIX: Final[str] = ".bak"

_PAISE: Final[Decimal] = Decimal("0.01")


class ConfigWriteError(RuntimeError):
    """The settings file could not be edited safely. Nothing was written."""


@dataclass(frozen=True, slots=True)
class WriteReport:
    """What one edit changed. Returned rather than logged-only, so callers can print it."""

    path: Path
    backup_path: Path | None
    session_budget_inr: Decimal | None
    symbols: tuple[str, ...] | None

    @property
    def touched_budget(self) -> bool:
        return self.session_budget_inr is not None

    @property
    def touched_watchlist(self) -> bool:
        return self.symbols is not None


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────


def format_decimal(value: Decimal) -> str:
    """Render a Decimal for YAML without exponent notation or a lost trailing zero."""
    to_paise = value.quantize(_PAISE)
    if to_paise == value:
        return f"{to_paise:f}"
    return f"{value.normalize():f}"


def render_watchlist_block(
    items: Sequence[WatchlistItem],
    *,
    generated_at: datetime,
    provenance: str,
) -> list[str]:
    """Build the replacement ``watchlist:`` block, comments included.

    The header comment is not decoration. A hand-edited watchlist and a machine-written one look
    identical in a diff, and the operator needs to know at a glance which they are looking at
    and how to get the old one back.
    """
    lines = [
        "watchlist:",
        "  # ---------------------------------------------------------------------------",
        f"  # GENERATED {generated_at.isoformat(timespec='seconds')} by tachyon.strategy.scanner.",
        f"  # Source: {provenance}" if provenance else "  # Source: pre-market scan",
        "  #",
        "  # Hand edits to this block are overwritten by the next boot_tachyon.py run. The",
        "  # previous version of this whole file is kept at settings.yaml.bak, and the file is",
        "  # tracked in git.",
        "  #",
        "  # `token` and `tick_size` come from Angel One's scrip master, not from this scanner --",
        "  # they are read out of the same file the ingestor verifies them against at boot",
        "  # (CLAUDE.md 2.3), so a symbol that reached this list has already been proved to",
        "  # exist under the name it is written with.",
        "  # ---------------------------------------------------------------------------",
    ]
    if not items:
        # Never reachable through the public API — write_settings refuses an empty watchlist —
        # but rendering [] as a bare key would produce `watchlist:` meaning null, and a config
        # that silently means "no symbols" is exactly the §8.1 failure this guards.
        raise ConfigWriteError("refusing to render an empty watchlist block")

    for item in items:
        lines.append(
            f'  - {{symbol: {item.symbol}, token: "{item.token}", '
            f"exchange: {item.exchange}, tick_size: {format_decimal(item.tick_size)}, "
            f"lot_size: {item.lot_size}}}"
        )
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Line surgery
# ──────────────────────────────────────────────────────────────────────────────


def _block_bounds(lines: Sequence[str], key: str) -> tuple[int, int]:
    """Half-open ``[start, end)`` line range of a top-level block, including its key line.

    Raises:
        ConfigWriteError: the key is absent, or present more than once.
    """
    starts = [
        index
        for index, line in enumerate(lines)
        if (match := _TOP_LEVEL_KEY.match(line)) is not None and match.group(1) == key
    ]
    if not starts:
        raise ConfigWriteError(
            f"no top-level '{key}:' key in the settings file — refusing to invent one"
        )
    if len(starts) > 1:
        raise ConfigWriteError(
            f"'{key}:' appears {len(starts)} times at the top level; the file is ambiguous"
        )

    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _TOP_LEVEL_KEY.match(lines[index]):
            end = index
            break
    return start, end


def _replace_block(lines: list[str], key: str, replacement: Sequence[str]) -> list[str]:
    """Swap a whole top-level block for new lines, preserving the surrounding file."""
    start, end = _block_bounds(lines, key)
    # Trailing blank lines belong to the *gap* between blocks, not to the block, so they are
    # carried across — otherwise every rewrite would eat one blank line and the file would
    # slowly compact itself.
    tail_blanks: list[str] = []
    index = end - 1
    while index > start and not lines[index].strip():
        tail_blanks.insert(0, lines[index])
        index -= 1
    return [*lines[:start], *replacement, *tail_blanks, *lines[end:]]


def _replace_scalar(lines: list[str], block: str, key: str, value: str) -> list[str]:
    """Replace one ``key: value`` line inside a block, keeping indentation and any comment."""
    start, end = _block_bounds(lines, block)
    pattern = re.compile(rf"^(\s+){re.escape(key)}\s*:\s*([^#]*)(#.*)?$")

    for index in range(start + 1, end):
        match = pattern.match(lines[index])
        if match is None:
            continue
        indent, _, comment = match.groups()
        suffix = f"  {comment}" if comment else ""
        updated = list(lines)
        updated[index] = f"{indent}{key}: {value}{suffix}"
        return updated

    raise ConfigWriteError(f"no '{key}:' line inside the '{block}:' block — refusing to invent one")


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


def _validate(
    text: str,
    *,
    expected_budget: Decimal | None,
    expected_symbols: tuple[str, ...] | None,
) -> None:
    """Prove the rendered text is a config this system will boot against.

    Runs the same checks ``config.py`` runs, plus a confirmation that the edit actually landed
    on the intended values.

    Raises:
        ConfigWriteError: on anything that would fail, or has silently not been applied.
    """
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigWriteError(f"the rendered settings file is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigWriteError(
            f"the rendered settings file is not a mapping (got {type(raw).__name__})"
        )

    # The §1 guard, applied before the file exists rather than after the next boot fails.
    try:
        _reject_protected(raw, origin="rendered settings.yaml")
    except Exception as exc:
        raise ConfigWriteError(f"the rendered settings file breaks CLAUDE.md §1: {exc}") from exc

    try:
        capital = CapitalSettings(**(raw.get("capital") or {}))
    except Exception as exc:
        raise ConfigWriteError(f"the rendered 'capital' section is invalid: {exc}") from exc

    entries = raw.get("watchlist") or []
    if not isinstance(entries, list):
        raise ConfigWriteError("the rendered 'watchlist' is not a list")
    try:
        items = tuple(WatchlistItem(**entry) for entry in entries)
    except Exception as exc:
        raise ConfigWriteError(f"the rendered 'watchlist' is invalid: {exc}") from exc

    symbols = [item.symbol for item in items]
    duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
    if duplicates:
        raise ConfigWriteError(f"the rendered watchlist has duplicate symbols: {duplicates}")

    if expected_budget is not None and capital.session_budget_inr != expected_budget:
        raise ConfigWriteError(
            f"post-write check failed: session_budget_inr reads "
            f"{capital.session_budget_inr}, expected {expected_budget}. The edit did not land "
            f"where it was aimed; nothing was written."
        )
    if expected_symbols is not None and tuple(symbols) != expected_symbols:
        raise ConfigWriteError(
            f"post-write check failed: watchlist reads {symbols}, expected "
            f"{list(expected_symbols)}. The edit did not land where it was aimed; nothing "
            f"was written."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def render_settings(
    text: str,
    *,
    session_budget_inr: Decimal | None = None,
    watchlist: Sequence[WatchlistItem] | None = None,
    generated_at: datetime,
    provenance: str = "",
) -> str:
    """Produce the edited settings text. Pure — no filesystem access.

    Raises:
        ConfigWriteError: the file's shape is not what an edit needs, or the result would be
            invalid.
    """
    if session_budget_inr is None and watchlist is None:
        raise ConfigWriteError("nothing to write: neither a budget nor a watchlist was given")
    if session_budget_inr is not None and session_budget_inr < 0:
        raise ConfigWriteError(f"session_budget_inr must not be negative, got {session_budget_inr}")
    if watchlist is not None and not watchlist:
        raise ConfigWriteError(
            "refusing to write an empty watchlist — a scan that found nothing must leave the "
            "existing one in place (CLAUDE.md §8.1)"
        )

    lines = text.splitlines()
    expected_budget: Decimal | None = None
    expected_symbols: tuple[str, ...] | None = None

    if session_budget_inr is not None:
        expected_budget = Decimal(format_decimal(session_budget_inr))
        lines = _replace_scalar(
            lines, "capital", "session_budget_inr", format_decimal(session_budget_inr)
        )

    if watchlist is not None:
        expected_symbols = tuple(item.symbol for item in watchlist)
        lines = _replace_block(
            lines,
            "watchlist",
            render_watchlist_block(watchlist, generated_at=generated_at, provenance=provenance),
        )

    rendered = "\n".join(lines)
    if text.endswith("\n"):
        rendered += "\n"

    _validate(rendered, expected_budget=expected_budget, expected_symbols=expected_symbols)
    return rendered


def write_settings(
    *,
    path: Path = SETTINGS_YAML,
    session_budget_inr: Decimal | None = None,
    watchlist: Sequence[WatchlistItem] | None = None,
    generated_at: datetime,
    provenance: str = "",
    backup: bool = True,
    dry_run: bool = False,
) -> WriteReport:
    """Edit ``settings.yaml`` in place, atomically, after validating the result.

    Args:
        path: the settings file. Defaults to the real one; tests pass a temp path.
        session_budget_inr: new capital figure, or ``None`` to leave it alone.
        watchlist: new watchlist, or ``None`` to leave it alone. Must be non-empty if given.
        generated_at: timestamp for the provenance comment. Passed in, not read from the
            clock, so a caller can render a reproducible file.
        provenance: one line describing where the watchlist came from.
        backup: keep the previous version at ``<path>.bak``.
        dry_run: validate and report, but write nothing.

    Returns:
        What was (or would have been) changed.

    Raises:
        ConfigWriteError: the file is unreadable, unexpectedly shaped, or the result would be
            an invalid config. In every case the file on disk is untouched.
    """
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigWriteError(f"cannot read {path}: {exc}") from exc

    rendered = render_settings(
        original,
        session_budget_inr=session_budget_inr,
        watchlist=watchlist,
        generated_at=generated_at,
        provenance=provenance,
    )

    symbols = tuple(item.symbol for item in watchlist) if watchlist is not None else None
    backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX) if backup else None

    if dry_run:
        _log.info(
            "config_writer.dry_run",
            path=str(path),
            session_budget_inr=None if session_budget_inr is None else str(session_budget_inr),
            symbols=list(symbols) if symbols else None,
            note="validated; nothing written",
        )
        return WriteReport(path, None, session_budget_inr, symbols)

    if backup_path is not None:
        try:
            backup_path.write_text(original, encoding="utf-8")
        except OSError as exc:
            # A backup we cannot write is a warning, not a refusal: the file is in git and the
            # rendered text has already been proved valid. Refusing here would turn a
            # read-only directory into a lost trading session.
            _log.warning("config_writer.backup_failed", path=str(backup_path), error=str(exc))
            backup_path = None

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise ConfigWriteError(f"cannot write {path}: {exc}") from exc

    _log.info(
        "config_writer.written",
        path=str(path),
        backup=str(backup_path) if backup_path else None,
        session_budget_inr=None if session_budget_inr is None else str(session_budget_inr),
        symbols=list(symbols) if symbols else None,
    )
    return WriteReport(path, backup_path, session_budget_inr, symbols)
