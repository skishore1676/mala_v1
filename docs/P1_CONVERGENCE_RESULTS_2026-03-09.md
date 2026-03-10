# P1 & Convergence Pipeline Results (2026-03-09)

## What Was Covered

The objective of the P0 and P1 iterations was to address the failures observed during the post-upgrade M4 gate evaluations and to add better pipeline tools for isolating the true robust edges. Specifically:

1. **Relative Friction:**
   - Modified the friction modeling from a fixed dollar-based `cost_r` to a relative `cost-bps` (basis points of the entry price).
   - This removes the severe bias against tickers with large price variance (like SPY vs TSLA), allowing the expectancy `E[R]` and confidence `P(MFE >= r*MAE)` to be computed with realistic slippage.
2. **Strategy Registry:**
   - Added `research_state.yaml` to govern the status (`active`, `candidate`, `under_eval`, `dead`) of all models.
   - The walk-forward and convergence pipelines now reference this logic to allow testing of multiple pre-tuned variants side-by-side (e.g., param-aware strategies), skipping those with no edge to save time.
3. **P0: Z-Score Grid Search:**
   - Evaluated `z_score_threshold` and `z_score_window` for `Elastic Band Reversion`.
   - Identified that `SPY` had zero robust edge, while `META`, `TSLA`, `IWM`, and `QQQ` had strong pockets of performance if tuned properly (e.g., lower threshold/longer window for META, tight threshold for TSLA).
4. **P1: Volume Filter Ablation & Directional Mass:**
   - Added `use_volume_filter` toggles to `Kinematic Ladder` & `Compression Expansion Breakout`.
   - Added `use_directional_mass` to `Elastic Band Reversion`.
   - Demonstrated that the volume gate actually *killed* the Kinematic Ladder edge completely, and removing it revealed a `+0.22 E[R]` edge on `TSLA short`.
   - Demonstrated that Directional Mass is highly protective—removing it floods the engine with noise and tanks E[R].

## Convergence Pipeline Final Run

With strict gates (`min OOS windows >= 4`, `min signals >= 1000`, `% positive OOS windows >= 60%`, `E[R] >= 0`), the `run_convergence_pipeline.py` script was executed over 10 active strategy variants across `AAPL, META, TSLA, QQQ, IWM`. 

**10 parameter combinations passed the strictest OOS stability gates, ranked by `E[R]`.**

| Ticker | Strategy | Direction | OOS % Positive | Min E[R] |
|---|---|---|---|---|
| **META** | Elastic Band z=1.0/w=240+dm | short | **100%** | **+0.135** |
| **TSLA** | Elastic Band z=1.75/w=120+dm | short | **100%** | **+0.133** |
| **TSLA** | Elastic Band z=1.25/w=360+dm | short | **100%** | **+0.117** |
| **IWM** | Elastic Band z=1.25/w=360+dm | short | 60% | **+0.074** |
| **TSLA** | Elastic Band z=2.0/w=240+dm | combined | 67% | **+0.061** |
| **TSLA** | Elastic Band z=1.25/w=360+dm | combined | 67% | **+0.046** |
| **META** | Elastic Band z=1.0/w=240+dm | combined | 67% | **+0.045** |
| **META** | Elastic Band z=1.25/w=360+dm | long | 67% | **+0.038** |
| **IWM** | Elastic Band z=2.0/w=240+dm | combined | 75% | **+0.026** |
| **META** | Elastic Band z=1.75/w=120+dm | combined | 67% | **+0.031** |

### Key Takeaways
- **Directional Mass works.** It forces `Elastic Band` signals into exceptionally clean exhaustion regimes.
- **TSLA & META** are our highest-expectancy underlyings, particularly on the **short side**.
- **Regime Router and Compression Breakout are dead.** They have been moved to `status: dead` in `research_state.yaml` and skipped in the pipeline.
- The pipeline correctly filters, identifies, and organizes parametrically optimal strategies for untouched holdout.

## Next Steps

With the convergence pipeline producing pristine, cross-validated candidates:
1. Promote the 10 candidates out of the convergence (walk-forward test) phase directly into the **untouched holdout validation** execution.
2. Advance toward **M6: Option Execution Synthesis** — taking the core E[R] metrics from our top `Elastic Band` candidates and mapping them into explicit defined-risk option spreads mapping delta, DTE, and theta assumptions.
