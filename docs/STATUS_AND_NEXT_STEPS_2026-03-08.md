# Status And Next Steps (2026-03-08)

## What We Built

### New Strategy Ideas Implemented

1. **Elastic Band Reversion**
- Mean-reversion around VPOC stretch + exhaustion flip.
- File: `src/strategy/elastic_band_reversion.py`

2. **Kinematic Ladder**
- Multi-timeframe-inspired regime/setup/trigger approach using velocity, acceleration, jerk.
- File: `src/strategy/kinematic_ladder.py`

3. **Compression Expansion Breakout**
- Trade breakouts after volatility compression with trend/volume filters.
- File: `src/strategy/compression_breakout.py`

4. **Regime Router (Kinematic + Compression)**
- Routes entries by regime (trend vs compression) into specialized sub-strategies.
- File: `src/strategy/regime_router.py`

### New Evaluation Tooling

1. **Measurement sensitivity + Monte Carlo**
- File: `scripts/run_measurement_sensitivity.py`
- Purpose: compare `1.0, 1.25, 1.5, 2.0` reward:risk thresholds with expectancy robustness.

2. **Integrated novel strategy runner**
- File: `scripts/run_novel_ideas.py`
- Purpose: run all novel strategies and emit both summary + robustness outputs.

3. **Walk-forward out-of-sample validator**
- File: `scripts/run_walk_forward_novel.py`
- Purpose: choose ratio in train window, evaluate in next OOS window.

4. **Convergence pipeline (multi-cost gate runner)**
- File: `scripts/run_convergence_pipeline.py`
- Purpose: run walk-forward at multiple friction assumptions and promote only candidates that pass strict robustness gates.

### Data-Time Alignment Fix

- Added timezone utilities (`src/time_utils.py`) to convert stored UTC timestamps to ET.
- Updated strategy/session filters and walk-forward/date boundaries to use ET session time.
- This prevents accidental timezone drift in open/close filters.

## Measurement Logic (Current)

For each strategy group (ticker + direction):

- Confidence at ratio `r`: `p = P(MFE >= r * MAE)`
- Friction-adjusted expectancy in R units:
  - `E[R] = p * r - (1 - p) - cost_r`
- Break-even confidence:
  - `p_be = (1 + cost_r) / (1 + r)`

Monte Carlo robustness uses bootstrap/binomial sampling of confidence to estimate:
- `P(E[R] > 0)`

## Findings So Far

### In-Sample / Full-Period Robustness

- **Best overall candidate:** `Elastic Band Reversion`
  - strongest at `1.5:1` and `2.0:1`.
  - high sample size and high `P(E[R] > 0)` across SPY/QQQ/IWM.
- `1.0:1` is not consistently positive once friction is applied.
- `Kinematic Ladder` is inconsistent and mostly weak except small pockets.
- `Compression Breakout` is mostly weak in robustness.

### Walk-Forward OOS (6m train / 3m test)

- **Most credible OOS candidate currently:** `SPY + Elastic Band Reversion` (especially short side).
- `QQQ + Compression Breakout (short)` showed slight positive OOS average, but still needs stronger confirmation.
- Most other combinations were negative or unstable out-of-sample.

### Expanded Universe + Longer Window (2024-01-01 to 2026-02-28)

- Universe expanded to: `SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL`.
- This improved discovery of ticker-specific edge pockets vs index-only testing.
- Most credible OOS leaders now include:
  - `AAPL + Elastic Band (combined/long)`
  - `PLTR + Elastic Band (long)`
  - `TSLA` short-side variants (Elastic/Compression/Router)
- This supports a **ticker-specialized deployment** view rather than one universal parameter set.

See details: `docs/EXPANDED_UNIVERSE_RESULTS_2026-03-08.md`.
Convergence execution details: `docs/CONVERGENCE_PLAN_2026-03-08.md`.
Holdout confirmation details: `docs/HOLDOUT_VALIDATION_2026-03-08.md`.

## Practical Interpretation For Trading

- We likely have **one promising directional entry engine** (`Elastic Band`) but not a broad universal engine yet.
- `2:1` is useful and currently appears more robust than `1:1` for these strategies under friction assumptions.
- The next bottleneck is not feature generation; it is parameter stability and OOS persistence.

## Recommended Next Steps

1. **Narrow focus:**
- Optimize only `Elastic Band Reversion` (SPY first, then QQQ/IWM portability check).

2. **Structured sweep:**
- Sweep: `stretch_pct`, `volume_multiplier`, session window, and optional trend filter.
- Keep one untouched OOS segment as a final holdout.

3. **Execution bridge to options:**
- Convert directional signal to option structure rules (DTE, delta band, spread width).
- Add conservative slippage/theta assumptions to avoid overstating edge.

4. **Promotion criteria (for live trial):**
- Positive OOS expectancy in multiple windows.
- Acceptable drawdown proxy.
- Stable edge under slightly worse friction assumptions.
