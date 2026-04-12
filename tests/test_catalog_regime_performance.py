"""Unit tests for src.research.catalog_regime_performance.

All tests run against synthetic inputs — no Polygon, no Mala data
cache. The ``regime_map`` kwarg on
``compute_catalog_regime_performance`` is the test injection point:
pass a handcrafted ``{date: MarketRegime}`` dict and the classifier is
never called.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.research.catalog_regime_performance import (
    CatalogRegimeSlice,
    SCHEMA_VERSION,
    TradeRow,
    _bucket_stats,
    compute_catalog_regime_performance,
    default_catalog_regime_performance_path,
    regime_key_of,
    write_catalog_regime_performance,
)
from src.research.market_regime import MarketRegime


# ── Fixtures ─────────────────────────────────────────────────────────────


def make_regime(
    d: date,
    *,
    vix_band: str = "mid",
    spy_trend_20d: str = "up",
    session_type: str = "normal",
) -> MarketRegime:
    return MarketRegime(
        trading_date=d,
        vix_band=vix_band,
        spy_trend_20d=spy_trend_20d,
        session_type=session_type,
        vix_close=18.0,
        spy_close=500.0,
        spy_sma20=498.0,
        spy_trend_slope_pct=0.1,
    )


def make_trade(
    d: date,
    *,
    catalog_key: str = "cat-a",
    ticker: str = "SPY",
    strategy: str = "impulse",
    direction: str = "short",
    mfe: float = 2.0,
    mae: float = 1.0,
) -> TradeRow:
    return TradeRow(
        catalog_key=catalog_key,
        ticker=ticker,
        strategy=strategy,
        direction=direction,
        trade_date=d,
        mfe=mfe,
        mae=mae,
    )


# ── regime_key_of ────────────────────────────────────────────────────────


class TestRegimeKey:
    def test_format_is_stable(self) -> None:
        regime = make_regime(date(2026, 4, 10))
        assert regime_key_of(regime) == "vix=mid__spy=up__session=normal"

    def test_all_bands_distinct(self) -> None:
        # Every (vix × spy × session) combination should produce a
        # unique key — no two bucket labels collide.
        seen = set()
        for vix in ("low", "mid", "high"):
            for spy in ("up", "flat", "down"):
                for session in ("normal", "opex", "post_fed", "earnings_heavy"):
                    regime = make_regime(
                        date(2026, 4, 10),
                        vix_band=vix,
                        spy_trend_20d=spy,
                        session_type=session,
                    )
                    key = regime_key_of(regime)
                    assert key not in seen
                    seen.add(key)
        assert len(seen) == 3 * 3 * 4


# ── _bucket_stats (wraps RewardRiskWinCondition math) ───────────────────


class TestBucketStats:
    def test_empty_returns_none(self) -> None:
        assert _bucket_stats([], [], ratio=1.5, cost_r=0.05) == (None, None, None)

    def test_all_wins(self) -> None:
        # MFE=3, MAE=1 → ratio=3 wins at threshold 1.5. Confidence=1.0,
        # expectancy = 1.0 * 1.5 - 0.0 - cost = 1.5 - 0.05 = 1.45.
        confidence, exp_r, avg_ratio = _bucket_stats(
            mfes=[3.0, 3.0, 3.0],
            maes=[1.0, 1.0, 1.0],
            ratio=1.5,
            cost_r=0.05,
        )
        assert confidence == pytest.approx(1.0)
        assert exp_r == pytest.approx(1.45)
        assert avg_ratio == pytest.approx(3.0)

    def test_all_losses(self) -> None:
        # MFE=1, MAE=1 → ratio=1 never clears the 1.5 threshold.
        # Confidence=0, expectancy = -1 - cost = -1.05.
        confidence, exp_r, _ = _bucket_stats(
            mfes=[1.0, 1.0, 1.0],
            maes=[1.0, 1.0, 1.0],
            ratio=1.5,
            cost_r=0.05,
        )
        assert confidence == pytest.approx(0.0)
        assert exp_r == pytest.approx(-1.05)

    def test_mixed_outcomes(self) -> None:
        # 2 wins (mfe=3, mae=1), 2 losses (mfe=1, mae=1). Confidence=0.5.
        # expectancy = 0.5 * 1.5 - 0.5 - 0.05 = 0.75 - 0.55 = 0.2
        confidence, exp_r, _ = _bucket_stats(
            mfes=[3.0, 3.0, 1.0, 1.0],
            maes=[1.0, 1.0, 1.0, 1.0],
            ratio=1.5,
            cost_r=0.05,
        )
        assert confidence == pytest.approx(0.5)
        assert exp_r == pytest.approx(0.2)


# ── compute_catalog_regime_performance ───────────────────────────────────


class TestCompute:
    def test_empty_trade_rows_returns_empty(self) -> None:
        result = compute_catalog_regime_performance([], regime_map={})
        assert result == []

    def test_single_bucket(self) -> None:
        d = date(2026, 4, 10)
        trades = [make_trade(d, mfe=3.0, mae=1.0) for _ in range(5)]
        regime_map = {d: make_regime(d)}

        slices = compute_catalog_regime_performance(
            trades,
            regime_map=regime_map,
            ratio=1.5,
            cost_r=0.05,
        )
        assert len(slices) == 1
        s = slices[0]
        assert s.catalog_key == "cat-a"
        assert s.ticker == "SPY"
        assert s.strategy == "impulse"
        assert s.direction == "short"
        assert s.regime_key == "vix=mid__spy=up__session=normal"
        assert s.n_trades == 5
        assert s.confidence == pytest.approx(1.0)
        assert s.exp_r == pytest.approx(1.45)

    def test_two_regimes_produce_two_buckets(self) -> None:
        d1 = date(2026, 4, 10)
        d2 = date(2026, 4, 11)
        trades = [
            make_trade(d1, mfe=3.0, mae=1.0),  # will bucket to mid/up
            make_trade(d1, mfe=3.0, mae=1.0),
            make_trade(d2, mfe=1.0, mae=1.0),  # will bucket to high/down
        ]
        regime_map = {
            d1: make_regime(d1, vix_band="mid", spy_trend_20d="up"),
            d2: make_regime(d2, vix_band="high", spy_trend_20d="down"),
        }

        slices = compute_catalog_regime_performance(
            trades, regime_map=regime_map
        )
        assert len(slices) == 2
        # Sorted by catalog_key then regime_key — high/down sorts before mid/up.
        by_regime = {s.regime_key: s for s in slices}
        assert by_regime["vix=mid__spy=up__session=normal"].n_trades == 2
        assert by_regime["vix=high__spy=down__session=normal"].n_trades == 1

    def test_multiple_catalog_keys_kept_separate(self) -> None:
        d = date(2026, 4, 10)
        trades = [
            make_trade(d, catalog_key="cat-a", mfe=3.0, mae=1.0),
            make_trade(d, catalog_key="cat-b", mfe=1.0, mae=1.0),
        ]
        regime_map = {d: make_regime(d)}
        slices = compute_catalog_regime_performance(trades, regime_map=regime_map)

        assert len(slices) == 2
        by_key = {s.catalog_key: s for s in slices}
        assert by_key["cat-a"].n_trades == 1
        assert by_key["cat-a"].confidence == pytest.approx(1.0)
        assert by_key["cat-b"].n_trades == 1
        assert by_key["cat-b"].confidence == pytest.approx(0.0)

    def test_trades_without_regime_coverage_are_dropped(self) -> None:
        # A trade on d2 with no regime entry should be silently
        # skipped rather than crashing.
        d1 = date(2026, 4, 10)
        d2 = date(2026, 4, 11)
        trades = [make_trade(d1, mfe=3.0, mae=1.0), make_trade(d2, mfe=3.0, mae=1.0)]
        regime_map = {d1: make_regime(d1)}

        slices = compute_catalog_regime_performance(trades, regime_map=regime_map)
        assert len(slices) == 1
        assert slices[0].n_trades == 1

    def test_output_is_sorted_deterministically(self) -> None:
        d = date(2026, 4, 10)
        trades = [
            make_trade(d, catalog_key="cat-z", mfe=3.0, mae=1.0),
            make_trade(d, catalog_key="cat-a", mfe=3.0, mae=1.0),
            make_trade(d, catalog_key="cat-m", mfe=3.0, mae=1.0),
        ]
        regime_map = {d: make_regime(d)}
        slices = compute_catalog_regime_performance(trades, regime_map=regime_map)
        assert [s.catalog_key for s in slices] == ["cat-a", "cat-m", "cat-z"]


# ── write_catalog_regime_performance ────────────────────────────────────


class TestWrite:
    def test_writes_schema_and_rows(self, tmp_path: Path) -> None:
        d = date(2026, 4, 10)
        trades = [make_trade(d, mfe=3.0, mae=1.0)]
        regime_map = {d: make_regime(d)}
        slices = compute_catalog_regime_performance(trades, regime_map=regime_map)

        out = tmp_path / "regime_perf.json"
        result_path = write_catalog_regime_performance(slices, out_path=out)
        assert result_path == out
        assert out.is_file()

        payload = json.loads(out.read_text())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["computed_at"].endswith("Z")
        assert len(payload["rows"]) == 1
        row = payload["rows"][0]
        assert row["catalog_key"] == "cat-a"
        assert row["regime_key"] == "vix=mid__spy=up__session=normal"
        assert row["n_trades"] == 1

    def test_default_path_is_absolute_and_under_project_root(self) -> None:
        path = default_catalog_regime_performance_path()
        assert path.is_absolute()
        assert path.name == "catalog_regime_performance.json"


# ── Integration sanity: CatalogRegimeSlice dataclass shape ─────────────


def test_catalog_regime_slice_frozen() -> None:
    s = CatalogRegimeSlice(
        catalog_key="x",
        ticker="SPY",
        strategy="impulse",
        direction="short",
        vix_band="mid",
        spy_trend_20d="up",
        session_type="normal",
        regime_key="vix=mid__spy=up__session=normal",
        n_trades=1,
        confidence=0.5,
        exp_r=0.2,
        avg_mfe_mae_ratio=2.0,
        ratio=1.5,
        effective_cost_r=0.05,
    )
    with pytest.raises(Exception):
        s.n_trades = 99  # type: ignore[misc]
