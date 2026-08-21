/* TACHYON terminal client — CLAUDE.md §7.2.
 *
 * Vanilla JS. No framework, no CDN, no build step.
 *
 * The one rule that shapes everything here: TRUTHFULNESS BEATS AESTHETICS. A value that has
 * stopped updating is rendered amber with its age in seconds, never as if it were fresh, and
 * never as a placeholder. A blank panel and a zero look identical on a P&L display, so a value
 * that has never arrived renders as an em dash, not as 0.00.
 *
 * Staleness is measured against the browser's OWN clock, from the moment a value arrived here.
 * Comparing a server timestamp against Date.now() would conflate two different faults — a
 * stalled feed and a clock skew between the two machines — and a laptop whose clock is two
 * minutes fast would paint the whole dashboard amber for no reason.
 */

'use strict';

const STALE_MS = 2000;      // CLAUDE.md §1 — the same 2 s the risk gate uses
const REPAINT_MS = 250;     // staleness re-evaluation; nothing blinks faster than 1 Hz
const RECONNECT_MS = [500, 1000, 2000, 4000, 8000];
const MAX_EVENTS = 50;

/** Arrival time (performance.now) of each tracked value, keyed by field id. */
const seen = new Map();

let socket = null;
let reconnectAttempt = 0;
let lastSnapshot = null;
let panicked = false;

// ── formatting ──────────────────────────────────────────────────────────────

const EM_DASH = '—';

/** Format a number, or an em dash when the value has never arrived. */
function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return EM_DASH;
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return EM_DASH;
  return n.toLocaleString('en-IN', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtSigned(value, digits = 2) {
  const text = fmt(value, digits);
  if (text === EM_DASH) return text;
  const n = Number(value);
  return n > 0 ? `+${text}` : text;
}

function ageSeconds(id) {
  const at = seen.get(id);
  return at === undefined ? Infinity : (performance.now() - at) / 1000;
}

function markSeen(id) {
  seen.set(id, performance.now());
}

// ── DOM helpers ─────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function setNum(id, text, cls) {
  const el = $(id);
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
  el.classList.remove('is-profit', 'is-loss', 'is-warn', 'is-neutral', 'is-long', 'is-short');
  if (cls) el.classList.add(cls);
}

function setPill(pillId, valueId, text, cls) {
  const pill = $(pillId);
  const value = $(valueId);
  const dot = $(`dot-${valueId}`);
  if (value && value.textContent !== text) value.textContent = text;
  for (const el of [pill, dot]) {
    if (!el) continue;
    el.classList.remove('is-ok', 'is-warn', 'is-breach', 'is-live', 'is-paper');
    if (cls) el.classList.add(cls);
  }
}

/**
 * Apply the stale rule to every tracked numeral.
 *
 * Runs on a timer rather than only on message arrival — precisely because the interesting case
 * is when messages have STOPPED arriving, and an update-driven repaint would never fire then.
 */
function repaintStaleness() {
  for (const el of document.querySelectorAll('[data-track]')) {
    const stale = ageSeconds(el.dataset.track) * 1000 > STALE_MS;
    el.classList.toggle('is-stale', stale);
  }

  const linkAge = ageSeconds('link');
  if (linkAge * 1000 > STALE_MS && socket && socket.readyState === WebSocket.OPEN) {
    setPill('pill-link', 'link', `STALE ${linkAge.toFixed(1)}s`, 'is-warn');
  }

  const tape = $('tape');
  if (tape) {
    for (const row of tape.querySelectorAll('tr[data-symbol]')) {
      const age = ageSeconds(`sym:${row.dataset.symbol}`);
      const stale = age * 1000 > STALE_MS;
      for (const cell of row.querySelectorAll('td')) cell.classList.toggle('is-stale', stale);
      const ageCell = row.querySelector('td.age');
      if (ageCell) ageCell.textContent = Number.isFinite(age) ? `${age.toFixed(1)}s` : EM_DASH;
    }
  }

  const obi = $('obi');
  if (obi) {
    for (const row of obi.querySelectorAll('tr[data-symbol]')) {
      const age = ageSeconds(`sym:${row.dataset.symbol}`);
      const stale = age * 1000 > STALE_MS;
      for (const cell of row.querySelectorAll('td')) cell.classList.toggle('is-stale', stale);
      const ageCell = row.querySelector('td.age');
      if (ageCell) ageCell.textContent = Number.isFinite(age) ? `${age.toFixed(1)}s` : EM_DASH;
    }
  }
}

// ── rendering ───────────────────────────────────────────────────────────────

function renderPnl(pnl) {
  if (!pnl || pnl.total === undefined) return;
  markSeen('pnl');

  const total = Number(pnl.total);
  const limit = Number(pnl.limit) || 0;
  const breached = Boolean(pnl.breached);

  setNum('pnl-total', `₹${fmtSigned(pnl.total)}`, breached || total < 0 ? 'is-loss' : 'is-profit');
  setNum('pnl-realised', `₹${fmtSigned(pnl.realised)}`, Number(pnl.realised) < 0 ? 'is-loss' : 'is-profit');
  setNum('pnl-floating', `₹${fmtSigned(pnl.floating)}`, Number(pnl.floating) < 0 ? 'is-loss' : 'is-profit');
  setNum('pnl-headroom', `₹${fmt(pnl.headroom)}`, breached ? 'is-loss' : 'is-neutral');

  // The gauge shows loss consumed against the limit, so "full" is the worst possible state.
  const consumed = limit > 0 ? Math.min(1, Math.max(0, -total / limit)) : 0;
  const fill = $('pnl-gauge');
  if (fill) {
    fill.style.width = `${(consumed * 100).toFixed(1)}%`;
    fill.classList.toggle('is-warn', consumed >= 0.5 && consumed < 1);
    fill.classList.toggle('is-breach', breached || consumed >= 1);
  }

  const foot = $('pnl-foot');
  if (foot) {
    foot.textContent = breached
      ? 'KILL SWITCH ENGAGED — no further entries this session'
      : `${(consumed * 100).toFixed(1)}% of the daily loss budget consumed`;
  }
  // The limit is no longer the ₹500 everyone has memorised — it can be derived from the
  // configured session capital. Show where it came from, so a wrong number is legible as a
  // wrong number rather than being read past.
  const limitFoot = $('limit-foot');
  if (limitFoot) {
    const capital = Number(pnl.capital ?? 0);
    limitFoot.textContent = capital > 0
      ? `limit ₹${fmt(pnl.limit)} · ${fmt(pnl.drawdown_pct)}% of ₹${fmt(pnl.capital)} · latching`
      : `limit ₹${fmt(pnl.limit)} · latching`;
    limitFoot.title = pnl.limit_source || '';
  }
  const chargesFoot = $('charges-foot');
  if (chargesFoot) chargesFoot.textContent = `charges ₹${fmt(pnl.charges)}`;
  const tradesFoot = $('trades-foot');
  if (tradesFoot) tradesFoot.textContent = `${pnl.trades_today ?? 0} trades`;
}

function renderSession(session) {
  if (!session || !session.state) return;
  markSeen('session');

  const mode = String(session.mode || '—');
  setPill('pill-mode', 'mode', mode, mode === 'LIVE' ? 'is-live' : 'is-paper');

  const state = String(session.state);
  const locked = state === 'LOCKED' || session.daily_lock_engaged;
  setPill('pill-state', 'state', locked ? `${state} 🔒` : state,
    locked ? 'is-breach' : state === 'ACTIVE' ? 'is-ok' : 'is-warn');

  const feedAge = Number(session.feed_age_seconds || 0);
  setPill('pill-feed', 'feed',
    session.feed_stale ? `STALE ${feedAge.toFixed(1)}s` : `LIVE ${feedAge.toFixed(1)}s`,
    session.feed_stale ? 'is-warn' : 'is-ok');

  const regime = String(session.macro_regime || 'NEUTRAL');
  const blocked = Boolean(session.macro_blocks_entries);
  setPill('pill-macro', 'macro',
    session.macro_degraded ? `${regime} (degraded)` : regime,
    blocked ? 'is-breach' : session.macro_degraded ? 'is-warn' : 'is-ok');

  setNum('macro-regime', regime,
    blocked ? 'is-loss' : regime === 'RISK_ON' ? 'is-profit' : 'is-neutral');
  const macroFoot = $('macro-foot');
  if (macroFoot) {
    macroFoot.textContent = session.macro_degraded
      ? 'Sentinel unavailable — trading continues under the last known good report'
      : `confidence ${session.macro_confidence ?? 0}%${blocked ? ' · NEW ENTRIES BLOCKED' : ''}`;
  }

  const open = Array.isArray(session.open_symbols) ? session.open_symbols : [];
  setNum('open-symbols', open.length ? open.join(' · ') : 'FLAT',
    open.length ? 'is-long' : 'is-neutral');

  const cooling = Array.isArray(session.cooling_symbols) ? session.cooling_symbols : [];
  const coolFoot = $('cooling-foot');
  if (coolFoot) {
    coolFoot.textContent = cooling.length
      ? `cooling: ${cooling.join(', ')}`
      : 'no symbols cooling';
  }

  if (locked) {
    panicked = true;
    const button = $('panic');
    if (button) { button.disabled = true; button.textContent = '⛔ ENTRIES BLOCKED'; }
  }
}

function renderSymbols(symbols) {
  const rows = Array.isArray(symbols) ? symbols : [];
  const tape = $('tape');
  const obi = $('obi');
  if (!tape || !obi) return;

  if (!rows.length) {
    tape.innerHTML = '<tr><td colspan="6" class="empty">no market data yet</td></tr>';
    obi.innerHTML = '<tr><td colspan="4" class="empty">no depth yet</td></tr>';
    return;
  }

  const tapeHtml = [];
  const obiHtml = [];
  for (const row of rows) {
    const symbol = String(row.symbol || '');
    markSeen(`sym:${symbol}`);
    const obiValue = Number(row.obi || 0);
    const obiWeighted = Number(row.obi_weighted || 0);
    const side = (v) => (v > 0.3 ? 'is-long' : v < -0.3 ? 'is-short' : '');

    tapeHtml.push(
      `<tr data-symbol="${escapeHtml(symbol)}">` +
      `<td class="sym">${escapeHtml(symbol)}</td>` +
      `<td>${fmt(row.ltp)}</td>` +
      `<td>${fmt(row.best_bid)}</td>` +
      `<td>${fmt(row.best_ask)}</td>` +
      `<td>${fmt(row.spread)}</td>` +
      `<td class="age">—</td></tr>`,
    );
    obiHtml.push(
      `<tr data-symbol="${escapeHtml(symbol)}">` +
      `<td class="sym">${escapeHtml(symbol)}</td>` +
      `<td class="${side(obiValue)}">${fmt(obiValue, 3)}</td>` +
      `<td class="${side(obiWeighted)}">${fmt(obiWeighted, 3)}</td>` +
      `<td class="age">—</td></tr>`,
    );
  }
  tape.innerHTML = tapeHtml.join('');
  obi.innerHTML = obiHtml.join('');
}

function renderEvents(events) {
  const list = Array.isArray(events) ? events.slice(0, MAX_EVENTS) : [];
  const container = $('events');
  if (!container) return;
  if (!list.length) {
    container.innerHTML = '<div class="empty">nothing yet</div>';
    return;
  }
  container.innerHTML = list.map((event) => {
    const severity = String(event.severity || 'INFO').toLowerCase();
    const when = event.ts_epoch
      ? new Date(Number(event.ts_epoch) * 1000).toLocaleTimeString('en-IN', { hour12: false })
      : '--:--:--';
    const text = event.reason || event.detail || event.status || event.kind || event.topic || '';
    const symbol = event.symbol ? `${event.symbol} · ` : '';
    return `<div class="event is-${escapeHtml(severity)}">` +
      `<time>${escapeHtml(when)}</time>` +
      `<span>${escapeHtml(symbol + String(text))}</span></div>`;
  }).join('');
}

/** Everything rendered from the wire goes through here. The payload is ours, but escaping it
 *  costs nothing and a broker-supplied rejection string reaches this display verbatim. */
function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function render(snapshot) {
  lastSnapshot = snapshot;
  renderPnl(snapshot.pnl);
  renderSession(snapshot.session);
  renderSymbols(snapshot.symbols);
  renderEvents(snapshot.events);
  repaintStaleness();
}

// ── transport ───────────────────────────────────────────────────────────────

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws/telemetry`);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    reconnectAttempt = 0;
    markSeen('link');
    setPill('pill-link', 'link', 'LIVE', 'is-ok');
  };

  socket.onmessage = (event) => {
    markSeen('link');
    setPill('pill-link', 'link', 'LIVE', 'is-ok');
    try {
      const text = typeof event.data === 'string'
        ? event.data
        : new TextDecoder().decode(event.data);
      render(JSON.parse(text));
    } catch (err) {
      // A frame we cannot parse is dropped. It must not stop the socket: the next frame is
      // 100 ms away and carries the complete state anyway.
      console.error('tachyon: undecodable frame', err);
    }
  };

  socket.onclose = () => {
    setPill('pill-link', 'link', 'DISCONNECTED', 'is-breach');
    const delay = RECONNECT_MS[Math.min(reconnectAttempt, RECONNECT_MS.length - 1)];
    reconnectAttempt += 1;
    setTimeout(connect, delay);
  };

  socket.onerror = () => {
    setPill('pill-link', 'link', 'ERROR', 'is-breach');
  };
}

// ── panic ───────────────────────────────────────────────────────────────────

function wirePanic() {
  const button = $('panic');
  if (!button) return;
  button.addEventListener('click', async () => {
    if (panicked) return;
    const confirmed = window.confirm(
      'PANIC\n\n' +
      'This engages the daily lock: NO NEW ENTRIES for the rest of the session, and across ' +
      'restarts.\n\n' +
      'It does NOT flatten open positions — those are closed by their broker-side stops, by ' +
      'the 15:15 square-off, or by you at the broker terminal.\n\n' +
      'It cannot be undone from this screen. Proceed?',
    );
    if (!confirmed) return;

    button.disabled = true;
    try {
      const response = await fetch('/api/panic', { method: 'POST' });
      const body = await response.json();
      if (body.engaged) {
        panicked = true;
        button.textContent = '⛔ ENTRIES BLOCKED';
      } else {
        // Never let the operator believe the market is closed to them when it is not.
        button.disabled = false;
        button.textContent = '⚠ PANIC FAILED — RETRY';
        window.alert(`PANIC FAILED — new entries are STILL PERMITTED.\n\n${body.error || ''}`);
      }
    } catch (err) {
      button.disabled = false;
      button.textContent = '⚠ PANIC FAILED — RETRY';
      window.alert(`PANIC FAILED — new entries are STILL PERMITTED.\n\n${err}`);
    }
  });
}

// ── boot ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  wirePanic();
  connect();
  setInterval(repaintStaleness, REPAINT_MS);
});

// Exported for the browser console and for tests; not used by the page itself.
window.tachyon = {
  ageSeconds,
  fmt,
  render,
  get snapshot() { return lastSnapshot; },
  STALE_MS,
};
