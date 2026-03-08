# Timezone Fix Impact Report (2026-03-08)

## Purpose
Quantify how ET session-time correction changed strategy metrics.

## Compared Files

- Before fix: `data/results/novel_strategy_robustness_2026-03-08_13-58-29.csv`
- After fix:  `data/results/novel_strategy_robustness_2026-03-08_14-35-34.csv`
- Full diff:  `data/results/timezone_fix_comparison_2026-03-08.csv`

## High-Level Summary

1. **Elastic Band Reversion**
- Signal counts were unchanged in this comparison (`delta_signals = 0`) because it does not use explicit time-of-day entry filters.
- Metrics shifted slightly due to ET-corrected day boundaries for EOD MFE/MAE computation.
- Net effect: modest expectancy/confidence changes, generally small and slightly negative.

2. **Kinematic Ladder**
- Meaningful changes in signals and metrics because it uses ET session windows.
- Effects are mixed by ticker/direction; some segments improved, others deteriorated.
- This confirms timezone alignment was materially affecting this strategy.

3. **Compression Expansion Breakout**
- Also materially changed (time-window gated strategy).
- Some previously weak slices improved, but overall profile remains mixed/mostly weak.

## Practical Interpretation

- The timezone correction was necessary and removed a structural bias.
- Post-fix results are more trustworthy for options-session decision-making.
- `Elastic Band Reversion` remains the most stable candidate.
- Time-window-dependent strategies need retuning under ET-corrected logic.

## Next Action

Run a focused parameter sweep on ET-corrected data for:
1. `Kinematic Ladder`
2. `Compression Expansion Breakout`

while keeping `Elastic Band` as the benchmark anchor.
