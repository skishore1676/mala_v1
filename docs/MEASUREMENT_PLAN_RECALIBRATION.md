# Measurement Plan Recalibration (1:1 vs 2:1 + Monte Carlo)

## Question
Is the current `2:1` win definition too strict for options-oriented signal validation?

## Short Answer
Partly yes.

- `2:1` is useful as a **high-payoff stress test**.
- It should not be the **only** success criterion for entry signals.
- A `1:1` criterion can be valid if confidence is high enough, but it requires a higher hit rate threshold after costs.

## Why This Matters More For Options

Options are not linear to underlying movement. Delta, gamma, theta, and vega all impact realized option P&L, so a stock-only directional metric should be treated as an entry-quality proxy, not a direct options P&L proxy.

References:
- OCC ODD (Characteristics and Risks of Standardized Options): https://www.theocc.com/getmedia/9d3854cd-b782-40be-bc73-68e0e53a19a1/odd.pdf
- Cboe Greeks glossary: https://www.cboe.com/optionsinstitute/tools/greeks-calculator/

## Core Math (Expectancy in R units)

Assume fixed payoff profile:
- Winner: `+r R`
- Loser: `-1 R`
- Friction cost per trade: `cost_r`

Expectancy:

`E[R] = p * r - (1 - p) - cost_r`

Break-even confidence:

`p_be = (1 + cost_r) / (1 + r)`

Implications:
- For `r = 2.0`, break-even confidence is lower.
- For `r = 1.0`, break-even confidence is higher (roughly above 50% before costs).

So your hypothesis can be right, but only if measured confidence is sufficiently above break-even after friction.

## Updated Measurement Plan

Use a multi-threshold framework instead of a single 2:1 gate.

1. Entry quality panel (stock-direction proxy):
- Evaluate confidence at `r in {1.0, 1.25, 1.5, 2.0}`.
- Report edge vs break-even confidence for each `r`.

2. Robustness panel:
- Bootstrap/Monte Carlo confidence intervals of expectancy.
- Report `P(E[R] > 0)` under each ratio and friction assumption.

3. Overfitting controls:
- Keep out-of-sample splits and walk-forward evaluation.
- Use multiple-testing aware diagnostics where possible.

References:
- White (2000), data-snooping reality check: https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf
- Hansen (2005), SPA test: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=302682
- Bailey & Lopez de Prado, Deflated Sharpe Ratio: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

## Implementation in This Repo

New script:
- `scripts/run_measurement_sensitivity.py`

What it does:
- Runs current directional strategies.
- Computes directional forward metrics.
- Evaluates multiple reward:risk thresholds.
- Runs Monte Carlo bootstrap for confidence/expectancy uncertainty.
- Saves a result CSV in `data/results/`.

## Recommended Decision Rule

Do not accept/reject a strategy on 2:1 confidence alone.

Accept candidates that satisfy all:
1. Signal count above minimum sample threshold.
2. Positive median expectancy at one or more practical ratios.
3. High `P(E[R] > 0)` under realistic `cost_r`.
4. Stability across tickers/time and out-of-sample windows.
