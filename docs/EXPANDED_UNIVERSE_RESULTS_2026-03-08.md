# Expanded Universe Results (2026-03-08)

## Objective

Test whether edge improves when:
- extending history window to `2024-01-01` → `2026-02-28`
- adding news-heavy single names beyond index ETFs

Universe tested:
- `SPY`, `QQQ`, `IWM`, `NVDA`, `TSLA`, `META`, `AMD`, `PLTR`, `AAPL`

## What Was Added

1. **Regime Router strategy integrated**
- File: `src/strategy/regime_router.py`
- Added to runners:
  - `scripts/run_novel_ideas.py`
  - `scripts/run_walk_forward_novel.py`

2. **Mixed-cache timestamp compatibility fix**
- File: `src/chronos/storage.py`
- `load_bars()` now normalizes `timestamp` to `Datetime[us, UTC]` at read time.
- This prevents concat failures when old cache files are timezone-naive and new files are timezone-aware.

## Key Runs

1. In-sample/whole-period robustness:
```bash
python scripts/run_novel_ideas.py \
  --tickers SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL \
  --start 2024-01-01 --end 2026-02-28 \
  --bootstrap-iters 1200 --ratios 1.0,1.25,1.5,2.0 --cost-r 0.05
```
Outputs:
- `data/results/novel_strategy_summary_2026-03-08_15-21-51.csv`
- `data/results/novel_strategy_robustness_2026-03-08_15-21-51.csv`

2. Walk-forward OOS:
```bash
python scripts/run_walk_forward_novel.py \
  --tickers SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL \
  --start 2024-01-01 --end 2026-02-28 \
  --train-months 6 --test-months 3 \
  --ratios 1.0,1.25,1.5,2.0 --cost-r 0.05 --min-signals 20
```
Outputs:
- `data/results/walk_forward_novel_detail_2026-03-08.csv`
- `data/results/walk_forward_novel_summary_2026-03-08.csv`

## Findings

## 1) Index-efficiency hypothesis: partially supported
- Index ETFs still show edge, but most robust new edges appear in selected single names.
- Strongest consistency now clusters in:
  - `AAPL + Elastic Band Reversion` (combined/long)
  - `PLTR + Elastic Band Reversion` (long)
  - `TSLA` short-side variants (Elastic/Compression/Router)

## 2) 1:1 vs higher R targets
- `1.0:1` can work for some combinations, but higher ratios remain generally stronger after friction for top setups.
- Best rows still concentrate around `1.5` to `2.0` for robust positive expectancy.

## 3) Regime Router status
- Router improved some short-side OOS cases (e.g., TSLA short), but not broadly dominant yet.
- It is useful as a framework for conditional deployment, not yet a universal winner.

## 4) Kinematic Ladder status
- Occasional high expectancy pockets exist, but sample size remains low and unstable.
- Treat as selective/specialist logic until more OOS evidence accumulates.

## Practical Takeaway

- The most credible near-term deployment path is:
  1. **Elastic Band as base engine**
  2. **Ticker/side specialization** (e.g., AAPL long, PLTR long, TSLA short)
  3. **Optional router overlay** where OOS supports it

- This is better than assuming one parameter set should work uniformly across all tickers.

## Next Step

Run a targeted walk-forward with stricter robustness gate:
- Keep only (ticker, strategy, direction) with:
  - `oos_windows >= 6`
  - `pct_positive_oos_windows >= 0.67`
  - `oos_signals >= 3000`
- Then perform friction stress test (`cost_r`: `0.05`, `0.08`, `0.12`) to check edge durability.
