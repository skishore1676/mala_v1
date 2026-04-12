"""Per-(catalog_key, market_regime) performance slicing.

This module is the other half of the shared-regime-vocabulary idea from
``src/research/market_regime.py``. Where ``market_regime`` provides the
vocabulary (a MarketRegime tuple per trading day), this module groups
realized out-of-sample trade rows by that vocabulary and produces
per-bucket expectancy / confidence / sample-count stats.

Downstream consumer is trade_lab's orchestrator: at 7 AM it classifies
today's regime, reads this file, and activates only the
(catalog_entry, regime) rows with enough sample count and positive
expectancy. See ``trade_lab/ARCHITECTURE.md`` section 2 Role 3.

## Usage pattern

This module does NOT generate signals. It consumes per-trade rows that
some upstream stage has already materialized — typically the M3 holdout
runner in ``src/research/stages/holdout.py`` with
``retain_trade_rows=True``. The pure-math path is fully testable
without Polygon: pass a synthetic ``regime_map`` to
``compute_catalog_regime_performance`` and it won't call
``market_regime.classify_range``.

## File output shape

The default output is a JSON file at
``data/playbooks/catalog_regime_performance.json`` with this shape::

    {
      "schema_version": 1,
      "computed_at": "2026-04-11T23:00:00Z",
      "rows": [
        {
          "catalog_key": "...",
          "ticker": "SPY",
          "strategy": "market_impulse_cross_reclaim",
          "direction": "short",
          "vix_band": "mid",
          "spy_trend_20d": "up",
          "session_type": "normal",
          "regime_key": "vix=mid__spy=up__session=normal",
          "n_trades": 12,
          "confidence": 0.58,
          "exp_r": 0.34,
          "avg_mfe_mae_ratio": 1.7,
          "ratio": 1.5,
          "effective_cost_r": 0.05
        },
        ...
      ]
    }

Rows with ``n_trades`` below the orchestrator's minimum are still
written — the orchestrator filters them out at read time. We keep the
raw shape here so a post-hoc analysis can see the full distribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import settings
from src.oracle.policies import RewardRiskWinCondition
from src.research.market_regime import MarketRegime, classify_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = 1


# ── Data types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeRow:
    """One realized signal row from an M3 holdout window.

    ``mfe`` and ``mae`` are in price units (same as the M3 pipeline —
    see ``src.oracle.metrics``). ``trade_date`` is the ET-adjusted date
    the signal fired, which is what we join to the regime classifier.

    ``catalog_key`` is threaded through so the downstream consumer can
    key directly into ``Strategy_Catalog`` without re-deriving it.
    Callers that don't have a catalog_key at build time should compute
    one via ``playbooks._build_catalog_key`` rather than faking it here.
    """

    catalog_key: str
    ticker: str
    strategy: str
    direction: str  # "long" / "short" / "combined"
    trade_date: date
    mfe: float
    mae: float


@dataclass(frozen=True)
class CatalogRegimeSlice:
    """One row in the catalog_regime_performance output."""

    catalog_key: str
    ticker: str
    strategy: str
    direction: str
    vix_band: str
    spy_trend_20d: str
    session_type: str
    regime_key: str
    n_trades: int
    confidence: Optional[float]
    exp_r: Optional[float]
    avg_mfe_mae_ratio: Optional[float]
    ratio: float
    effective_cost_r: float


# ── Core helpers ───────────────────────────────────────────────────────────


def regime_key_of(regime: MarketRegime) -> str:
    """Deterministic string key for a regime tuple.

    Kept stable so downstream readers (orchestrator, feedback agent)
    can build the same key when looking up today's regime. Do not
    reorder the fields without bumping ``SCHEMA_VERSION`` and updating
    every reader.
    """
    return (
        f"vix={regime.vix_band}"
        f"__spy={regime.spy_trend_20d}"
        f"__session={regime.session_type}"
    )


def _bucket_stats(
    mfes: list[float],
    maes: list[float],
    ratio: float,
    cost_r: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (confidence, exp_r, avg_mfe_mae_ratio) for one bucket.

    Mirrors ``evaluate_df`` in walk_forward.py but runs on a plain
    numpy array so we can call it from a pure pytest context without
    a Polars dependency. Uses the same RewardRiskWinCondition policy.
    """
    if not mfes or not maes:
        return None, None, None

    mfe_arr = np.asarray(mfes, dtype=float)
    mae_arr = np.asarray(maes, dtype=float)
    policy = RewardRiskWinCondition(ratio=ratio)
    confidence = float(policy.confidence(mfe_arr, mae_arr))
    exp_r = float(policy.expectancy(mfe_arr, mae_arr, cost_r))

    valid_mask = np.isfinite(mfe_arr) & np.isfinite(mae_arr) & (mae_arr > 0)
    if np.any(valid_mask):
        avg_ratio: Optional[float] = float(
            np.mean(mfe_arr[valid_mask] / mae_arr[valid_mask])
        )
    else:
        avg_ratio = None

    return (
        round(confidence, 4),
        round(exp_r, 4),
        round(avg_ratio, 4) if avg_ratio is not None else None,
    )


# ── Public API ─────────────────────────────────────────────────────────────


def compute_catalog_regime_performance(
    trade_rows: list[TradeRow],
    *,
    ratio: float = 1.5,
    cost_r: float = 0.05,
    regime_map: Optional[dict[date, MarketRegime]] = None,
    api_key: Optional[str] = None,
) -> list[CatalogRegimeSlice]:
    """Group trade rows by (catalog_key, regime) and compute stats.

    Args:
        trade_rows: one row per realized signal from an M3 holdout
            window. Usually collected via
            ``run_holdout_validation_for_candidates(..., retain_trade_rows=True)``.
        ratio: reward-risk ratio used to classify wins. The M3 pipeline
            typically chooses this per candidate via ``choose_ratio``;
            callers that want per-candidate ratios should call this
            function once per ratio value or pass in slices pre-grouped
            by ratio. For the common case where one ratio dominates,
            the default 1.5 is reasonable.
        cost_r: transaction cost in R units. 0.05 is the walk_forward
            default; override when known.
        regime_map: pre-computed ``{date: MarketRegime}``. When
            provided, the classifier is NOT called — this is the test
            path and also the "batch reuse" path when multiple callers
            share one classification.
        api_key: Polygon API key override. Only used when
            ``regime_map`` is None. Propagated to ``classify_range``.

    Returns:
        One ``CatalogRegimeSlice`` per (catalog_key, regime_key)
        combination that had at least one trade. Rows with
        ``n_trades`` below any downstream threshold are still emitted
        — the orchestrator filters them at read time.
    """
    if not trade_rows:
        return []

    # Build the regime classification once for the full date span.
    if regime_map is None:
        min_d = min(r.trade_date for r in trade_rows)
        max_d = max(r.trade_date for r in trade_rows)
        regime_map = classify_range(min_d, max_d, api_key=api_key)

    # Group into buckets: (catalog_key, ticker, strategy, direction, regime_key)
    # → (accumulated mfe/mae lists + the MarketRegime for labels).
    @dataclass
    class _Bucket:
        catalog_key: str
        ticker: str
        strategy: str
        direction: str
        regime: MarketRegime
        mfes: list[float] = field(default_factory=list)
        maes: list[float] = field(default_factory=list)

    buckets: dict[tuple[str, str, str, str, str], _Bucket] = {}

    for row in trade_rows:
        regime = regime_map.get(row.trade_date)
        if regime is None:
            # Trade fell on a date the classifier has no coverage for
            # (weekend / holiday / pre-SMA window). Skip silently —
            # logging would be too noisy for nightly runs.
            continue
        rk = regime_key_of(regime)
        bucket_key = (row.catalog_key, row.ticker, row.strategy, row.direction, rk)
        bucket = buckets.get(bucket_key)
        if bucket is None:
            bucket = _Bucket(
                catalog_key=row.catalog_key,
                ticker=row.ticker,
                strategy=row.strategy,
                direction=row.direction,
                regime=regime,
            )
            buckets[bucket_key] = bucket
        bucket.mfes.append(row.mfe)
        bucket.maes.append(row.mae)

    slices: list[CatalogRegimeSlice] = []
    for (_, _, _, _, regime_key), bucket in buckets.items():
        confidence, exp_r, avg_ratio = _bucket_stats(
            bucket.mfes, bucket.maes, ratio, cost_r
        )
        slices.append(
            CatalogRegimeSlice(
                catalog_key=bucket.catalog_key,
                ticker=bucket.ticker,
                strategy=bucket.strategy,
                direction=bucket.direction,
                vix_band=bucket.regime.vix_band,
                spy_trend_20d=bucket.regime.spy_trend_20d,
                session_type=bucket.regime.session_type,
                regime_key=regime_key,
                n_trades=len(bucket.mfes),
                confidence=confidence,
                exp_r=exp_r,
                avg_mfe_mae_ratio=avg_ratio,
                ratio=ratio,
                effective_cost_r=cost_r,
            )
        )

    # Stable sort: catalog_key, then regime_key. Makes diffs between
    # nightly runs readable.
    slices.sort(key=lambda s: (s.catalog_key, s.regime_key))
    return slices


def default_catalog_regime_performance_path() -> Path:
    """Mirror of ``default_master_playbook_catalog_path`` for the new tab.

    Uses ``settings.catalog_regime_performance_path`` if set, else the
    conventional location next to the master catalog.
    """
    configured = getattr(
        settings,
        "catalog_regime_performance_path",
        "data/playbooks/catalog_regime_performance.json",
    )
    path = Path(configured)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def write_catalog_regime_performance(
    slices: list[CatalogRegimeSlice],
    *,
    out_path: Optional[Path] = None,
) -> Path:
    """Write slices as JSON to the canonical location.

    Overwrites any prior run's output — this file is a snapshot of
    "latest per-regime performance," not an append-only log. Nightly
    runs replace it in full.
    """
    target = out_path or default_catalog_regime_performance_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": [asdict(s) for s in slices],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return target
