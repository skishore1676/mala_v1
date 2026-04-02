# Kinematic Engine

A state-of-the-art backtesting environment that models market price action as a physical object moving through a resistive medium.

## Quick Start

```bash
# Install dependencies (uv)
uv sync

# Or pip fallback
pip install -r requirements.txt

# Set your Polygon.io API key
echo "POLYGON_API_KEY=your_key_here" > .env

# Run the full pipeline
uv run python main.py --tickers SPY --start 2024-01-01 --end 2024-12-31

# Run with cached data only (skip download)
uv run python main.py --skip-download --tickers SPY
```

## What This Project Does

The engine answers three questions:

1. Is there a valid setup right now? (`Strategy` signals)
2. What typically happens after that setup? (`Oracle` forward metrics: MFE/MAE/confidence)
3. If traded with explicit exits, is it actually profitable? (`TradeSimulator` for Market Impulse)

## Architecture (End-to-End Flow)

| Module       | Path              | Purpose                                      |
|:-------------|:------------------|:---------------------------------------------|
| **Chronos**  | `src/chronos/`    | Polygon.io data pipeline with Parquet cache   |
| **Newton**   | `src/newton/`     | Physics engine (Velocity, Acceleration, Jerk) |
| **Strategy** | `src/strategy/`   | Configurable strategy agents                  |
| **Oracle**   | `src/oracle/`     | MFE/MAE metrics and reporting                 |

### Refactor Direction

The current architectural pressure point is not `Chronos`; it is orchestration and duplicated research logic across the growing `run_*.py` surface area.

- `Chronos` remains the stable data foundation for now.
- `Newton` is moving toward composable feature transforms plus centralized multi-timeframe handling instead of letting MTF logic spread into scripts.
- `Newton` now exposes a composable transform pipeline and reusable timeframe resampler while keeping `PhysicsEngine` as the compatibility facade.
- `Newton` now exposes additive parameterized kinematic features so agents can request bounded multi-bar derivatives without redefining existing `*_1m` semantics.
- `Oracle` is moving toward a cleaner split between evaluation policy, simulation logic, and persistence/reporting.
- `Strategy` is moving toward richer metadata for orchestration, including required features, parameter space, and evaluation mode.

The target workflow is a **hybrid research architecture** with an API-first execution surface:

- M1-M5 stage gates remain deterministic and explicitly defined.
- An AI research controller may run bounded experiments inside those rules.
- The AI may not redefine gates, silently promote a strategy, or bypass failed stages.
- The supported execution path should flow through `ResearchOrchestrator` and `src/research/tools.py`, not through ad hoc runner selection.
- Legacy `run_*.py` scripts are quarantined under `scripts/legacy/` for provenance and comparison only.
- `scripts/run_research_orchestrator.py` remains the only top-level orchestration CLI.

### Timezone And Session Handling

- Raw bars are stored with UTC timestamps.
- All market-session filters (open/close windows) are evaluated in `America/New_York` time.
- This avoids accidental use of pre/post-market bars when a strategy is intended for regular options session logic.

### Data + Backtest Flow

```text
Polygon API (1-min OHLCV)
  -> Chronos client fetches in chunks with pagination/retries
  -> Chronos storage saves daily parquet: data/<TICKER>/<YYYY-MM-DD>.parquet
  -> load_bars(ticker, start, end)
  -> Newton enrich():
       feature-targeted transform pipeline based on selected strategy requirements
       (velocity_1m, accel_1m, jerk_1m, velocity_3/5, accel_3/5, jerk_3/5,
       ema_*, volume_ma_*,
       internal_strength, directional_mass, directional_mass_ma_*, vpoc_4h, MTF features)
  -> Strategy generate_signals():
       signal (and signal_direction for directional strategies)
  -> Oracle:
       add_forward_metrics() and summarise_signals()
       or TradeSimulator.simulate() for trade-level P&L
  -> Reporter:
       data/results/<timestamp>_<ticker>_<strategy>/
         experiment.json, trade_log.csv, summary.txt
```

## How Metrics Are Calculated

### Physics Features (Newton)

Newton can now be used in two ways:

- compatibility mode: `PhysicsEngine.enrich(df)` applies the default core transform set,
- feature-targeted mode: `PhysicsEngine.enrich_for_features(df, required_features)` builds only the transforms needed for the selected strategies.

For multi-timeframe research, `MarketImpulseTransform` now uses a reusable `TimeframeResampler` so higher-timeframe joins are handled consistently instead of being hand-coded inside scripts. The feature pipeline also accepts parameterized transform specs such as `market_impulse:15m`, allowing agents to sweep regime timeframes without adding Newton special cases.

Agent-safe Newton surface:

- Existing names like `velocity_1m`, `accel_1m`, and `jerk_1m` keep their original one-bar meaning.
- Additive multi-bar variants such as `velocity_3`, `accel_3`, and `jerk_3` are resolved on demand through `required_features` / `enrich_for_features(...)`.
- Market Impulse VWMA stacks are structurally validated: exactly three periods, strictly increasing.
- Canonical bounded sweeps should prefer discrete choices such as `kinematic_periods_back in {1, 3, 5}` and VWMA stacks like `(5, 13, 21)`, `(8, 21, 34)`, `(10, 20, 40)`.

- `velocity_1m = close[t] - close[t-1]`
- `accel_1m = velocity_1m[t] - velocity_1m[t-1]`
- `jerk_1m = accel_1m[t] - accel_1m[t-1]`
- `velocity_n = close[t] - close[t-n]` for bounded multi-bar differencing
- `accel_n = velocity_n[t] - velocity_n[t-n]`
- `jerk_n = accel_n[t] - accel_n[t-n]`
- `ema_p`: exponential moving average of `close` for period `p`
- `volume_ma_n`: rolling mean of `volume` over `n` bars
- `internal_strength = ((close - low) - (high - close)) / (high - low)` (0 when `high == low`)
- `directional_mass = volume * internal_strength`
- `directional_mass_ma_n`: rolling mean of `directional_mass` over `n` bars
- `vpoc_4h`: most volume-concentrated rounded price in rolling `vpoc_lookback_bars` window (default 240 bars)

### Forward Metrics (Oracle)

For each bar `i` (using future bars only):

- `entry_price = close[i]`
- `future_high = max(high[i+1 : i+1+N])`
- `future_low = min(low[i+1 : i+1+N])`
- `MFE = future_high - entry_price`
- `MAE = entry_price - future_low`
- `win = MFE > 2 * MAE` (default 2:1 reward:risk threshold)

Where `N = forward_window_bars` (default 15 bars = 15 minutes).

Summary metrics are computed on rows where `signal == True` and forward metrics are non-null:

- `total_signals`, `wins`, `losses`
- `confidence_score = wins / total_signals`
- distribution stats: `avg/median/max MFE` and `avg/median/max MAE`

### Trade-Level Metrics (Market Impulse Simulator)

The `TradeSimulator` converts directional signals into non-overlapping trades:

- Entry: at signal bar close
- Exit policy:
  - `vma_trailing` (default): long stop on first full 1-min bar with `high < vma_10_5m`, short stop on first full 1-min bar with `low > vma_10_5m`
  - `fixed_rr`: bounded fixed stop / take-profit exits such as `stop_loss=1.0`, `reward_multiple=2.0`
  - Otherwise exit at end of session (`15:59` bar)

Per-trade P&L:

- Long: `pnl = exit_price - entry_price`
- Short: `pnl = entry_price - exit_price`

Aggregate metrics:

- `win_rate = winners / total_trades`
- `avg_winner`, `avg_loser`
- `profit_factor = gross_wins / abs(gross_losses)`
- `expectancy = total_pnl / total_trades`
- `total_pnl`

## Strategy Logic (Current)

### EMA Momentum (`src/strategy/ema_momentum.py`)

Signal when all gates are true:

- Trend gate: `EMA(4) > EMA(8) > EMA(12)`
- Location gate: `close > vpoc_4h`
- Force gate: `volume > volume_ma_20`

### Market Impulse (`src/strategy/market_impulse.py`)

- Regime from 5-min VWMA stack: bullish or bearish
- Trigger on 1-min cross-and-reclaim of VMA
- Time filter: default entry window `09:33` to `10:30` ET
- Agent-safe parameter surface includes bounded VWMA stack choices with structural validation
- Emits both `signal` and `signal_direction` (`long`/`short`)

### Elastic Band Reversion (`src/strategy/elastic_band_reversion.py`)

- Uses dynamic stretch via rolling z-score of VPOC distance (instead of fixed % stretch).
- Velocity exhaustion remains mandatory.
- Jerk confirmation is now an explicit optional ablation.
- Directional-mass participation is also an explicit optional ablation:
  - Long requires `directional_mass > 0`
  - Short requires `directional_mass < 0`

### Regime Router (`src/strategy/regime_router.py`)

- Uses rolling volatility ratio and trend-velocity to classify regime.
- Routes to `Kinematic Ladder` (trend) or `Compression Breakout` (compression).
- Emits `route_regime` with directional signals.

### Opening Drive Classifier (`src/strategy/opening_drive_classifier.py`)

- Classifies opening impulse using first 25 minutes after `09:30` ET.
- Entry phase begins strictly after the opening window in the canonical surface.
- If opening drive is strong up/down:
  - `continue` mode: trade range expansion in drive direction.
  - `fail` mode: trade reclaim through opening midpoint against initial drive.
- Acceleration confirmation remains mandatory.
- Jerk, volume-vs-opening-baseline, and directional-mass confirmations are explicit optional ablations.

Variant now supported:
- **Opening Drive v2 (Short Continue)**: short-only, continuation-only, stricter drive/volume filters.

## How We Backtest

### Standard Pipeline Backtest (EMA-style)

Run:

```bash
python main.py --tickers SPY --start 2024-01-01 --end 2024-12-31
```

This does:

1. Load/download bars.
2. Enrich bars with Newton features.
3. Generate signals.
4. Compute forward MFE/MAE over configured lookahead.
5. Summarize signal quality and save immutable experiment artifacts.

### Legacy Provenance

Legacy runners are preserved under `scripts/legacy/` for provenance and historical comparison only.

- They are not the recommended path for new research work.
- The active execution path is `ResearchOrchestrator` plus `src/research/tools.py`.
- If you need an old runner for comparison, see [scripts/legacy/README.md](/Users/suman/kg_env/projects/mala_v1/scripts/legacy/README.md).

## Refactor Validation Set

The refactor is not considered successful unless the strategies below can still be instantiated, run, and produce comparable outputs through the new orchestration path.

- `Elastic Band Reversion`: primary regression baseline and strongest current research candidate.
- `Opening Drive Classifier`: validates session-aware logic, opening-window rules, and directional output.
- `Kinematic Ladder`: validates the momentum/regime path and ensures weaker candidates still move correctly through the workflow.
- `Jerk-Pivot Momentum (tight)`: validates an alternate directional feature dependency built around VPOC proximity and jerk inflection.
- `Market Impulse (Cross & Reclaim)`: validates the reusable multi-timeframe regime and entry path.

## Default Agent Catalog

The default agent-facing research loop is intentionally narrower than the full strategy namespace to reduce multiple-testing pressure.

- Core default catalog:
  - `Elastic Band Reversion`
  - `Jerk-Pivot Momentum (tight)`
  - `Opening Drive Classifier`
  - `Market Impulse (Cross & Reclaim)`
- Lineage strategies such as `Kinematic Ladder`, `Compression Expansion Breakout`, and `Regime Router (Kinematic + Compression)` remain buildable by explicit name for provenance, replay, and tests, but are excluded from default discovery/orchestrator exploration.

Minimum acceptance for the validation set:

- Each strategy can be constructed from the new orchestration layer without hand-editing scripts.
- Each strategy can generate signals on enriched data with its expected feature dependencies resolved automatically.
- Each strategy can complete at least the walk-forward evaluation stage through the new orchestration path.
- Result artifacts remain comparable enough to prior runs that a future agent can detect regressions instead of rediscovering baseline behavior.

## Refactor Roadmap

This repo is moving toward a hybrid agentic research workflow with the following boundaries:

- Deterministic stage governance: M1-M5 gates stay rule-based and reproducible.
- Bounded AI authority: the agent can propose and run experiments within declared budgets and parameter spaces.
- Dual-layer entrypoint model: keep current scripts as internal building blocks while adding one canonical orchestrator above them.
- Reusable library-first internals: shared research logic should live in modules, not be duplicated across runner scripts.

## Research Agent Contract

The repo now carries a dedicated research-agent skill contract at `skills/research-experiment-agent/SKILL.md`.

That contract defines:

- the agent persona: skeptical, evidence-first, and conservative about promotion,
- the objective: shepherd raw ideas to `promote`, `retune`, `gather_more_evidence`, or `kill`,
- the allowed tasks: bounded experiments, result analysis, and stage-appropriate recommendations,
- the forbidden actions: changing gates, bypassing holdout discipline, or silently rewriting the research rules.

Future agentic workflow work should treat that file as the canonical operating brief for the experiment-running agent.

The first reusable orchestration pieces now live in:

- `src/research/registry.py`: repo-memory-backed strategy discovery and instantiation,
- `src/research/tools.py`: callable experiment tools (`parameter_sweep`, `evaluate_config`, `query_incumbent`, `query_pareto_front`, `query_neighborhood`, `query_dead_zones`, `baseline_comparison`, `ablation_check`, `walk_forward`, `holdout_validation`, `execution_mapping`),
- `src/research/stages/walk_forward.py`: reusable walk-forward math and aggregation,
- `src/research/stages/convergence.py`: reusable convergence gate logic,
- `src/research/stages/holdout.py`: reusable holdout candidate selection, ratio fitting, and pass/fail summarization,
- `src/research/stages/execution.py`: reusable execution mapping and Monte Carlo stage logic,
- `scripts/run_research_orchestrator.py`: the only supported top-level orchestration CLI for inspecting tracked strategies, validation fixtures, and next allowed actions.

Legacy stage runners now live under `scripts/legacy/`, and the active execution model is the reusable API surface in `src/research/` plus the top-level orchestrator preview CLI.

Because the long-term system is intentionally agent-driven over a wide parameter surface, legacy runners should be treated as archived reference material rather than a parallel workflow. Future research changes should prefer reusable logic in `src/research/`, `src/oracle/`, and `src/newton/`, even when preserving backward compatibility for old scripts.

The research flow also uses strategy-declared `required_features` to ask Newton only for the needed transforms, including parameterized `MarketImpulseTransform` variants through the same `enrich_for_features(...)` pipeline used by other strategies.

Agent-native search conventions now in force:

- When a strategy exposes `search_spec`, that is the canonical optimization surface for sweeps and point evaluation.
- `parameter_sweep()` applies gating and constraint normalization before spending search budget, so inactive parameters are pruned rather than rediscovered through downstream deduping.
- `evaluate_config()` records deterministic config signatures plus compact memory rows for iterative optimization.
- Memory queries can be scoped to the active research slice, not just strategy name, so agents can stay inside the same symbol/date/evaluation context while optimizing.
- `query_incumbent()` and `query_pareto_front()` rank only statistically competitive configs by default; low-signal or otherwise insufficient evaluations are excluded unless explicitly requested.
- Slice-local memory queries are part of the canonical stage loop in M1/M2, alongside sweeps and point evaluation, rather than an out-of-band helper surface.
- `src.research` is safe to import from strategy-adjacent code because the package surface now lazily loads heavier orchestration modules instead of importing them eagerly.

## Nightly Review Loop

The canonical operator-facing nightly workflow is now `scripts/run_nightly_regime_matrix.py`.

That nightly loop:

- scouts the default Tier 1 watchlist across the core research families at `M1/M2` only,
- merges nightly M2 survivors into a durable human review queue,
- refreshes local Plotly chart artifacts for each queued survivor,
- executes approved follow-up actions under explicit nightly budget caps, and
- writes a review workbook/CSV bundle for the next morning's inspection pass.

The intended architecture is explicit:

- nightly scout finds and organizes candidates
- human review queue decides what deserves expensive validation
- only validated `M3/M4/M5` follow-ups produce deployable playbooks

That means an empty top-level `deployment_candidates.json` on a scout-only night is expected behavior, not a broken pipeline.

The nightly CLI now prints an explicit summary block such as:

```text
scout_only_run = true
deployment_candidates_generated = 0
reason = no M3-M5 follow-up executed
```

Read that as:

- the scout ran correctly
- candidates may still have been queued for review
- no validated follow-up produced deployable playbooks yet
- the queue/workbook/charts are the primary operating outputs for that run

Even when a night produces zero M2 survivors, the operator contract is still to publish an initialized queue/history/workbook surface rather than failing or leaving no review artifacts behind.

Important defaults live in [config/nightly_regime_matrix.yaml](/Users/suman/kg_env/projects/mala_v1/config/nightly_regime_matrix.yaml).
The canonical operator checklist now lives in [docs/bionic_loop_checklist.md](/Users/suman/kg_env/projects/mala_v1/docs/bionic_loop_checklist.md).
Nightly research reference details for the queue, review artifacts, and terminal-state rules live in [docs/nightly_regime_matrix.md](/Users/suman/kg_env/projects/mala_v1/docs/nightly_regime_matrix.md).

A canonical execution pattern now exists for agentic research work:

```python
from src.research import ResearchOrchestrator, ResearchStage

orchestrator = ResearchOrchestrator()
result = orchestrator.run_action(
    ResearchStage.M1_DISCOVERY,
    "parameter_sweep",
    strategy_name="Elastic Band Reversion",
)
```

Research tool calls return a structured `ResearchToolResult` with `summary` and `artifacts`. When injecting in-memory `ticker_frames`, keep `start_date` and `end_date` aligned with the actual frame coverage so walk-forward windows are built against the same span you loaded.

Execution hygiene for agents and teammates:

- Prefer `./.venv/bin/python` over assuming the shell `python` is already the project interpreter.
- Prefer the orchestrator/toolbox API or one-shot commands over creating ad hoc scratch scripts in the repo root.
- Start with tracked tickers from `research_state.yaml`; if you broaden the symbol set for M1 discovery, state that explicitly.

## Script Governance

Not every script in `scripts/` should be treated as equally supported.

- Top-level orchestration CLI: `run_research_orchestrator.py`
- Top-level utility: `query_results_db.py`
- All runner-style CLIs now live under `scripts/legacy/`
- New research execution should prefer `ResearchOrchestrator` and `src/research/tools.py` over direct legacy script use
- Legacy scripts are archived for provenance and compatibility, not ongoing feature development

The current keep/migrate/archive decision table lives in [scripts/STATUS.md](/Users/suman/kg_env/projects/mala_v1/scripts/STATUS.md).

## How Success Is Measured

Use both layers below; they answer different questions:

1. Signal quality (Oracle forward metrics):
   - High `confidence_score`
   - Higher `avg_mfe` than `avg_mae`
   - Stable median values (not only max outliers)
2. Executability/profitability (Trade simulator):
   - `profit_factor > 1.0` (preferably > 1.3-1.5)
   - Positive `expectancy`
   - Acceptable drawdown proxy via loser size and MAE behavior

Practical interpretation:

- A setup can have good forward excursion stats but still fail as a tradable system if exits/holding rules produce poor expectancy.
- A robust strategy should pass both filters: probabilistic edge and executable trade economics.

## Local Results Database

In addition to CSV/JSON files, core research scripts persist rows into:

- `data/results/results.db`

Tables:

- `runs`: metadata for each script execution
- `artifact_rows`: normalized rows with query keys (`ticker`, `strategy`, `direction`, `signals`, `confidence`, `exp_r`, `decision`) + full JSON payload

Quick query:

```bash
uv run python scripts/query_results_db.py --artifact-type walk_forward_novel_summary --limit 20
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Configuration

All defaults are in `src/config.py` and are overridable via `.env`:

| Variable           | Default           | Description                     |
|:-------------------|:------------------|:--------------------------------|
| `POLYGON_API_KEY`  | *(required)*      | Polygon.io API key              |
| `DEFAULT_TICKERS`  | SPY, QQQ, IWM     | Tickers to download             |
| `LOOKBACK_YEARS`   | 2                  | Years of historical data        |
| `VPOC_LOOKBACK_BARS` | 240              | Rolling window for VPOC (bars)  |
| `EMA_PERIODS`      | 4, 8, 12          | EMA stack periods               |
| `FORWARD_WINDOW_BARS` | 15             | Forward-look window for MFE/MAE |
