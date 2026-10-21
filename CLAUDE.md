# Mala v1 — Claude Code Project Context

This file is auto-loaded by Claude Code at the start of every session in
this repo. Keep it tight and factual. The **Hypothesis Workbench** section
at the bottom is a living log — update it after interactive workbench
sessions with anything future-you would benefit from remembering.

Trading-system role: **Mala is the research authority.** It discovers,
validates, and catalogs strategies. Approved strategies flow into a Google
Sheet control tower (`Strategy_Catalog` tab), which a downstream
`trade_lab` orchestrator reads to pick what to run each day. Bhiksha is
the dumb executor on the other side of the sheet. Full architecture:
`~/kg_env/projects/trade_lab/ARCHITECTURE.md`.

---

## How to Run

```bash
# Install / sync
uv sync

# Canonical research CLIs
uv run python main.py --tickers SPY --start 2024-01-01 --end 2024-12-31
uv run python scripts/run_research_orchestrator.py
uv run python scripts/run_nightly_regime_matrix.py

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_market_regime.py -v
```

Polygon API key goes in `.env` as `POLYGON_API_KEY=...`. Pydantic settings
(`src/config.py`) picks it up automatically.

---

## Key Directories

| Module | Path | What it does |
|---|---|---|
| Chronos | `src/chronos/` | Polygon.io data fetch + Parquet cache |
| Newton | `src/newton/` | Physics features (velocity, acceleration, jerk, VPOC, EMAs, regime columns) |
| Strategy | `src/strategy/` | Registered strategy classes (see `factory.py:_NAMED_BUILDERS`) |
| Oracle | `src/oracle/` | MFE/MAE metrics, forward returns, trade logs |
| Research | `src/research/` | Orchestrators, stage gates M1–M5, playbook catalog, nightly scout, Google Sheets bridge |
| Config | `config/` | YAML for nightly matrix, strategy presets, regime router |

---

## Stage Gates (M1–M5)

Research stages are deterministic gates a candidate must pass to earn a
`Strategy_Catalog` entry:

- **M1**: signal-level sanity (confidence, signal count, monotonic structure)
- **M2**: stability across parameter perturbations and symbol variants
- **M3**: walk-forward holdout validation (`src/research/stages/holdout.py`)
- **M4**: cross-regime robustness
- **M5**: ensemble + deployment candidate selection

Only M3–M5 passers end up in the master playbook catalog via
`sync_master_playbook_catalog_sheet` (`src/research/playbooks.py:483`).

---

## Strategy Registry

Authoritative list of callable strategy builders:
`src/strategy/factory.py:_NAMED_BUILDERS` (line 113).

Current entries (as of 2026-04):

- `Elastic Band Reversion`
- `Kinematic Ladder`
- `Compression Expansion Breakout`
- `Regime Router (Kinematic + Compression)`
- `Opening Drive Classifier`
- `Opening Drive v2 (Short Continue)`
- `Jerk-Pivot Momentum (tight)`
- `Market Impulse (Cross & Reclaim)`

Helper: `available_strategy_names()` returns the tuple.
Research tracking: `src/research/registry.py:DEFAULT_AGENT_CATALOG`.

---

## Hypothesis Workbench

**Purpose:** when the human has a trading hypothesis (fresh idea or a row
from the `Idea_Intake` gSheet tab), work through it interactively with
Claude to decide: reusable in Mala as-is, needs a small extension, needs
a larger Newton feature, or kill.

### How to start a workbench session

1. Open Claude Code in the `mala_v1/` repo root.
2. Paste the prompt from `prompts/workbench-start.md` and fill in the idea.
3. Claude produces a feasibility report and waits for go-ahead before
   running `evaluate_config` or drafting code.
4. At session end, add one or two lines to the "Cross-session lessons"
   subsection below if you learned something future-you would benefit from.

### Feasibility tags (the output Claude should produce)

- **`config-only`** — an existing strategy class can represent this with
  parameter changes alone. Lowest lift. Draft a YAML config or direct
  `evaluate_config` call and we're done.
- **`new-class`** — needs a new strategy class in `src/strategy/` but
  reuses existing Newton features. Scope the class (methods, ~line
  count, test file) before writing.
- **`new-feature`** — needs a new physical feature in Newton
  (`src/newton/`) OR a new Oracle metric. Biggest lift. Scope the
  feature addition before writing.

### What Claude should read first (in order)

1. This "Hypothesis Workbench" section (accumulated lessons)
2. `src/strategy/factory.py:113` (current strategy registry)
3. `src/research/stages/holdout.py` (how M3 walk-forward is invoked)
4. `src/config.py` (settings class, Polygon key access pattern)
5. `src/chronos/client.py` (data access)
6. The idea source (Idea_Intake row or free-text hypothesis the human provides)

### Output on promote

- Set `Idea_Intake.triage_status = promote_to_m3` in the gSheet
- Drop a lore file at `mala_v1/strategies/{catalog_key}.md` containing:
  hypothesis in the human's words, source URL if any, rules, feasibility
  notes, M1/M2 results, the config or code that made it work
- The nightly M3–M5 runner picks it up automatically

### Output on kill

- Set `Idea_Intake.triage_status = killed` with a one-line reason
- That's it. The `Idea_Intake_Seen` tab will prevent the scout from
  re-surfacing the same URL.

### Cross-session lessons

_This subsection is the compounding journal. Append one or two lines
per session only when there's something worth remembering. Keep entries
short — if it gets long, prune the stale ones._

- _(no entries yet — first workbench session will seed this)_

---

## Operational Notes

- **Data is cached**: raw 1-minute bars are stored as Parquet under
  `data/` — don't re-download unnecessarily. `LocalStorage` in
  `src/chronos/storage.py` handles this.
- **Per-strategy regime columns** (`impulse_regime_5m`, `route_regime`,
  etc.) are strategy-internal. They describe the strategy's own state,
  not market context. For cross-strategy market context, use the new
  `src/research/market_regime.py` module (shared vocabulary for the
  downstream orchestrator).
- **Nightly scout runs via launchd** — plist at
  `scripts/launchd/com.mala.nightly_regime_matrix.plist`. Install with
  `launchctl load ...` once and forget.
