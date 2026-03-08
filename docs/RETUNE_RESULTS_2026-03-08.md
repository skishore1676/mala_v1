# Retune Results (ET-Corrected) — 2026-03-08

## Scope
Focused retune sweep for:
- `Kinematic Ladder`
- `Compression Expansion Breakout`

Data:
- Tickers: `SPY`, `QQQ`, `IWM`
- Period: `2025-01-01` to `2026-02-28`
- Ratio for sweep objective: `1.5`
- Friction: `cost_r = 0.05`

## Output Artifacts

- Sweep full: `data/results/retune_sweep_full_2026-03-08_14-40-14.csv`
- Sweep top: `data/results/retune_sweep_top_2026-03-08_14-40-14.csv`
- Walk-forward check (min 20 signals/window): `data/results/retune_walkforward_check_2026-03-08.csv`
- Walk-forward check (min 10 signals/window): `data/results/retune_walkforward_check_min10_2026-03-08.csv`

## What Improved

### Kinematic Ladder
- ET-corrected retuning discovered profitable pockets at `r=1.5`.
- Best in-sample combined configs used:
  - `vol=1.0`
  - `aw=8`
  - `rw in {20,35}`
  - session mostly `09:35–15:30` or `09:45–15:20` ET

### Compression Breakout
- In-sample still mostly weak, but retune reduced losses.
- Best variants favored:
  - `cf=0.95`
  - `vol=1.05`
  - broader day session windows

## OOS Read (Important)

Using walk-forward and selected top configs:

- With strict `min_signals=20`, only compression configs consistently produced enough test signals.
- With `min_signals=10`, several kinematic configs showed strong positive OOS expectancy, but sample sizes were small (10–32 signals across windows).

Interpretation:
- Kinematic may have a real edge in sparse, high-selectivity regimes.
- Compression has larger sample support but weaker edge quality.
- Do not trust kinematic results yet without more OOS windows / larger sample.

## Recommended Next Step

1. Keep two parallel tracks:
- **Track A (Quality):** Kinematic Ladder (high expectancy, low sample)
- **Track B (Capacity):** Compression Breakout (high sample, low expectancy)

2. Extend walk-forward horizon to add more OOS windows.

3. Add regime-conditioned deployment:
- Use kinematic only in high-trend regime.
- Use compression only in low-vol expansion regime.

4. Compare both against Elastic Band baseline before promotion.
