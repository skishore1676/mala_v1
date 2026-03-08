# Universal Truths Strategy Plan

## Goal
Build directional strategy hypotheses for options trading decisions using only underlying stock price/volume data. The objective is to detect repeatable directional edge from market microstructure and momentum/mean-reversion behavior, then validate via backtests.

## Universal Truths (Operationalized)

1. Volatility clusters.
2. Market microstructure overshoots mean-revert after one-sided bursts.
3. Trend continuation is strongest when higher-timeframe direction aligns with lower-timeframe trigger.
4. Price dislocations from value (VPOC/VWAP-like anchors) either continue with participation or snap back.
5. Time-of-day regime matters; open/close behavior differs from midday.

## Strategy Hypotheses To Implement

### 1) Elastic Band Reversion
Premise: excessive stretch away from value with exhaustion can snap back.

Signal logic:
- Stretch long setup: `(close - vpoc_4h) / vpoc_4h <= -stretch_pct`
- Stretch short setup: `(close - vpoc_4h) / vpoc_4h >= +stretch_pct`
- Exhaustion flip long: `velocity_1m < 0` and `jerk_1m > 0`
- Exhaustion flip short: `velocity_1m > 0` and `jerk_1m < 0`
- Participation filter: `volume > volume_multiplier * volume_ma_n`

Outputs:
- `signal` (bool)
- `signal_direction` (`long` / `short`)

### 2) Kinematic Ladder (Multi-Timeframe Inspired)
Premise: use higher-timeframe direction for bias and lower-timeframe derivatives for entry timing.

Signal logic:
- Regime (HTF proxy from 1-min rolling windows):
  - Bull: `rolling_mean(velocity_1m, regime_window) > 0` and `rolling_mean(accel_1m, accel_window) >= 0`
  - Bear: opposite signs
  - EMA stack alignment confirms regime
- Setup:
  - Bull pullback into EMA zone: `ema_12 <= close <= ema_8`
  - Bear pullback into EMA zone: `ema_8 <= close <= ema_12`
- Trigger:
  - Bull: `velocity_1m > 0` and `jerk_1m > 0`
  - Bear: `velocity_1m < 0` and `jerk_1m < 0`
- Participation filter: `volume > volume_multiplier * volume_ma_n`
- Optional session filter to avoid noisy open/close tails

Outputs:
- `signal`
- `signal_direction`

## Backtest Approach

1. Use local cached 1-min bars (`data/<TICKER>/<YYYY-MM-DD>.parquet`).
2. Enrich with Newton features (`velocity_1m`, `accel_1m`, `jerk_1m`, EMA stack, volume MA, VPOC).
3. Run each directional strategy.
4. Evaluate with directional forward metrics:
   - `forward_mfe_eod`, `forward_mae_eod`, snapshots (30m/60m)
   - win flag: `forward_mfe_eod >= 2 * forward_mae_eod`
5. Summarize by combined/long/short confidence and MFE/MAE ratio.

## Success Criteria

Minimum evidence of edge:
- Sufficient signal count (avoid tiny-sample illusions).
- Combined `confidence_2to1` materially above random baseline.
- `avg_mfe_mae_ratio > 1.0` and ideally closer to/above 1.5.
- Stability across multiple tickers (SPY/QQQ/IWM) and both directions.

## Implementation Plan

1. Add two strategy classes in `src/strategy/`.
2. Add unit tests for signal shape, direction output, and required-column validation.
3. Add research runner script in `scripts/` to execute both strategies and produce result artifacts.
4. Run tests + backtests, then compare both strategies on signal quality and robustness.

## Notes For Options Execution Layer (Future)

- Keep directional model independent of option chain noise.
- Map directional signals to option structures by regime:
  - trend continuation: debit spreads or directional calls/puts
  - mean reversion: shorter target windows, tighter risk and time stops
- Add later once underlying-direction edge is proven.
