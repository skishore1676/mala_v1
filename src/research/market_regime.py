"""
Market regime classifier — shared vocabulary for historical per-regime
slicing (Mala's nightly M3–M5 runner) and today's activation decisions
(trade_lab's orchestrator).

The public API is two functions:

    classify(target_date)          -> MarketRegime
    classify_range(start, end)     -> dict[date, MarketRegime]

Both return a ``MarketRegime`` dataclass with three categorical fields:

    vix_band       : "low" / "mid" / "high"
    spy_trend_20d  : "up"  / "flat" / "down"
    session_type   : "normal" / "opex" / "post_fed" / "earnings_heavy"

Plus the raw numeric values used to bucket them, for debugging.

**Not to be confused with per-strategy regime columns**
(``impulse_regime_5m``, ``route_regime``) which describe an individual
strategy's internal state. Those are strategy-facing. *This* module is
the market-facing regime vocabulary that downstream consumers (the
orchestrator, the per-regime performance table) share.

Semantics of a "classification as of date D":
    * The regime is computed from data available **as of the close of
      day D**. For the orchestrator's 7 AM call on day D+1, call
      ``classify(D)`` to get yesterday's regime — which is the right
      thing for activation decisions (regime shifts are slow; today's
      close is not yet known at 7 AM).
    * Weekends and holidays fall back to the most recent prior trading
      day that has a close.

Architecture context: ``trade_lab/ARCHITECTURE.md`` section 2c.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, Optional

import requests

from src.config import settings

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

VixBand = Literal["low", "mid", "high"]
SpyTrend = Literal["up", "flat", "down"]
SessionType = Literal["normal", "opex", "post_fed", "earnings_heavy"]


@dataclass(frozen=True)
class MarketRegime:
    """A market regime snapshot for one trading day."""

    trading_date: date
    vix_band: VixBand
    spy_trend_20d: SpyTrend
    session_type: SessionType
    # Raw values for debugging and post-hoc analysis:
    vix_close: Optional[float]  # None if VIX data was not reachable
    spy_close: float
    spy_sma20: float
    spy_trend_slope_pct: float  # SMA20(D) vs SMA20(D-5), as pct/day


class VixUnavailable(RuntimeError):
    """Raised when VIX daily closes cannot be fetched from Polygon.

    This typically means the configured Polygon subscription does not
    include the indices feed. Either upgrade the plan to include
    indices, or swap in a different VIX source (yfinance ^VIX, CBOE
    daily CSV) in ``_fetch_vix_daily_closes``.
    """


# ── Thresholds (tune during W1.2 validation with real data) ─────────────────

VIX_LOW_CEILING = 15.0
VIX_HIGH_FLOOR = 22.0
# 20d SMA slope threshold: |slope| < this → flat; else up/down.
# 0.05% per day ≈ 1% over 20 days — a gentle drift, not a trend.
SPY_TREND_FLAT_THRESHOLD_PCT_PER_DAY = 0.05


# ── Polygon helpers ─────────────────────────────────────────────────────────

_POLYGON_BASE = "https://api.polygon.io"


def _polygon_daily_closes(
    ticker: str,
    start: date,
    end: date,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch daily OHLCV bars for one ticker, inclusive date range.

    Returns the list of bar dicts from Polygon (keys: t, o, h, l, c, v).
    ``t`` is unix ms of the bar's start.
    """
    key = api_key or settings.polygon_api_key
    if not key:
        raise ValueError("POLYGON_API_KEY not configured")

    url = (
        f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{start.isoformat()}/{end.isoformat()}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50_000,
        "apiKey": key,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("results", []) or []


def _fetch_spy_daily_closes(
    start: date, end: date, api_key: Optional[str] = None
) -> dict[date, float]:
    """Return {trading_date: close} for SPY across the given window."""
    bars = _polygon_daily_closes("SPY", start, end, api_key=api_key)
    out: dict[date, float] = {}
    for bar in bars:
        ts_ms = bar["t"]
        d = datetime.utcfromtimestamp(ts_ms / 1000).date()
        out[d] = float(bar["c"])
    return out


def _fetch_vix_daily_closes(
    start: date, end: date, api_key: Optional[str] = None
) -> dict[date, float]:
    """Return {trading_date: close} for VIX across the given window.

    Tries Polygon's indices feed first (``I:VIX``). If the API returns
    an error or an empty result set, raises ``VixUnavailable`` — the
    caller decides whether to degrade or fail hard.

    To swap in a fallback (yfinance / CBOE CSV), replace the body of
    this function with the alternate fetcher and return the same dict
    shape.
    """
    try:
        bars = _polygon_daily_closes("I:VIX", start, end, api_key=api_key)
    except requests.HTTPError as e:
        raise VixUnavailable(
            f"Polygon returned {e.response.status_code} for I:VIX. "
            "This usually means the current subscription tier does not "
            "include the indices feed. Upgrade the plan or replace "
            "_fetch_vix_daily_closes with a fallback source."
        ) from e

    if not bars:
        raise VixUnavailable(
            "Polygon returned no bars for I:VIX — either the date range "
            "contains no trading days or the indices feed is not "
            "enabled on this API key."
        )

    out: dict[date, float] = {}
    for bar in bars:
        ts_ms = bar["t"]
        d = datetime.utcfromtimestamp(ts_ms / 1000).date()
        out[d] = float(bar["c"])
    return out


# ── Pure classification logic (unit-testable) ───────────────────────────────


def classify_vix_band(vix_close: Optional[float]) -> VixBand:
    """Bucket a VIX close into low/mid/high.

    If ``vix_close`` is None (VIX source unavailable), default to "mid"
    — the neutral bucket. Callers that want hard failure should check
    the raw value and raise instead.
    """
    if vix_close is None:
        return "mid"
    if vix_close < VIX_LOW_CEILING:
        return "low"
    if vix_close > VIX_HIGH_FLOOR:
        return "high"
    return "mid"


def classify_spy_trend(
    sma20_today: float, sma20_five_days_ago: float
) -> tuple[SpyTrend, float]:
    """Classify SPY trend from the 5-day change in its 20-day SMA.

    Returns (label, slope_pct_per_day). The slope is expressed as
    percent per calendar day so the threshold is intuitive ("moving
    roughly 0.05% per day" → flat).
    """
    if sma20_five_days_ago == 0:
        return "flat", 0.0
    total_pct = (sma20_today / sma20_five_days_ago - 1.0) * 100.0
    slope_pct_per_day = total_pct / 5.0
    if slope_pct_per_day > SPY_TREND_FLAT_THRESHOLD_PCT_PER_DAY:
        return "up", slope_pct_per_day
    if slope_pct_per_day < -SPY_TREND_FLAT_THRESHOLD_PCT_PER_DAY:
        return "down", slope_pct_per_day
    return "flat", slope_pct_per_day


def _is_third_friday(d: date) -> bool:
    """True if *d* is the third Friday of its month (monthly opex)."""
    if d.weekday() != 4:  # Friday is 4
        return False
    # Count how many Fridays have occurred in this month including d.
    friday_number = (d.day - 1) // 7 + 1
    return friday_number == 3


def classify_session_type(target_date: date) -> SessionType:
    """Classify the session type for a given date.

    Current rules:
      * ``opex``           : third Friday of the month (monthly expiry)
      * ``post_fed``       : stubbed to False — needs FOMC calendar
      * ``earnings_heavy`` : stubbed to False — needs earnings calendar
      * ``normal``         : default

    TODO(W2.2): plug in a FOMC calendar (hardcoded list is fine) and
    an earnings calendar (Polygon reference or free source).
    """
    if _is_third_friday(target_date):
        return "opex"
    return "normal"


# ── Core classification ─────────────────────────────────────────────────────


def _compute_sma(closes: list[float]) -> float:
    return sum(closes) / len(closes)


def classify_range(
    start: date,
    end: date,
    *,
    api_key: Optional[str] = None,
    lookback_buffer_days: int = 35,
) -> dict[date, MarketRegime]:
    """Classify every trading day in [start, end].

    This is the bulk path — use it for nightly per-regime slicing over
    a backtest's full date range. One Polygon call for SPY + one for
    VIX, then in-memory bucketing.

    ``lookback_buffer_days`` pads the start date so we have enough
    history to compute the 20-day SMA on the first output day.
    """
    window_start = start - timedelta(days=lookback_buffer_days)

    spy_closes_map = _fetch_spy_daily_closes(window_start, end, api_key=api_key)
    try:
        vix_closes_map = _fetch_vix_daily_closes(window_start, end, api_key=api_key)
    except VixUnavailable as e:
        logger.warning("VIX unavailable, defaulting vix_band to 'mid': %s", e)
        vix_closes_map = {}

    # Sort trading days ascending so we can walk them and compute SMAs.
    trading_days = sorted(spy_closes_map.keys())
    if not trading_days:
        return {}

    # For every day in [start, end] we need:
    #   * SPY close on that day
    #   * SMA20 of SPY close for the trailing 20 sessions ending on D
    #   * SMA20 of SPY close for the trailing 20 sessions ending on D-5
    # Walk the sorted list and maintain a rolling window.
    out: dict[date, MarketRegime] = {}
    closes_list = [spy_closes_map[d] for d in trading_days]

    for idx, d in enumerate(trading_days):
        if d < start or d > end:
            continue
        if idx < 20:
            # Not enough history for a 20-day SMA yet.
            continue
        if idx < 25:
            # Not enough history for a 5-session-ago SMA comparison.
            continue

        sma20_today = _compute_sma(closes_list[idx - 19 : idx + 1])
        sma20_lag5 = _compute_sma(closes_list[idx - 24 : idx - 4])
        trend_label, slope = classify_spy_trend(sma20_today, sma20_lag5)

        vix_close = vix_closes_map.get(d)
        vix_label = classify_vix_band(vix_close)
        session_label = classify_session_type(d)

        out[d] = MarketRegime(
            trading_date=d,
            vix_band=vix_label,
            spy_trend_20d=trend_label,
            session_type=session_label,
            vix_close=vix_close,
            spy_close=closes_list[idx],
            spy_sma20=sma20_today,
            spy_trend_slope_pct=slope,
        )

    return out


def classify(
    target_date: date,
    *,
    api_key: Optional[str] = None,
) -> MarketRegime:
    """Classify a single trading day.

    Convenience wrapper for one-off calls (e.g., the orchestrator's
    morning run). For bulk historical slicing, call ``classify_range``
    instead — one Polygon round-trip for SPY + one for VIX covers the
    whole window.

    If ``target_date`` is not a trading day (weekend / holiday), the
    result is the regime for the most recent prior trading day that
    has data.
    """
    # 35-day buffer back from target_date gives ~25 trading days of
    # history which is what classify_range needs for the 20-day SMA
    # plus the 5-session lag comparison.
    window_start = target_date - timedelta(days=35)
    results = classify_range(window_start, target_date, api_key=api_key)
    if not results:
        raise RuntimeError(
            f"No SPY data available up to {target_date}. "
            "Polygon may be rate-limited, the date range may fall on "
            "holidays, or the API key may be misconfigured."
        )

    # Pick the most recent classified day ≤ target_date.
    latest_key = max(k for k in results.keys() if k <= target_date)
    return results[latest_key]
