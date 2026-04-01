---
name: nightly-research-operator
description: Use when running or supervising Mala's nightly research loop, especially when you need to launch the nightly regime matrix, inspect the stateful M2 human review queue, interpret local chart/workbook artifacts, handle zero-survivor family outcomes without treating them as failures, or prepare the next human review pass.
---

# Nightly Research Operator

## Overview

Run the nightly scout, keep the human review queue healthy, and hand the next morning's work to a reviewer in a clean, inspectable form.

This skill is for operations, not strategy invention. Use it when the job is "make tonight's loop run correctly and leave behind the right review surface."

## When To Use It

- Launching `scripts/run_nightly_regime_matrix.py`
- Inspecting or editing `data/results/nightly_regime_matrix/research_control/m2_human_review_queue.csv`
- Reviewing `review_bundle/*.csv`, `human_review_workbook.xlsx`, or `charts/*.html`
- Debugging nightly failures in family runners or queue/export integration
- Preparing a Codex automation that should run the nightly scout and surface review-ready outputs
- Converting operational lessons from a live nightly run into stable repo procedure

## Core Loop

1. Run the nightly matrix with the project interpreter:

```bash
./.venv/bin/python scripts/run_nightly_regime_matrix.py
```

2. Read the run outputs from:

- dated bundle: `data/results/nightly_regime_matrix/<YYYY-MM-DD>/nightly_regime_matrix/<HH-MM-SS>/`
- control area: `data/results/nightly_regime_matrix/research_control/`

3. Confirm these artifacts exist before asking for human review:

- `m2_human_review_queue.csv`
- `m2_human_review_history.csv`
- `review_bundle/human_review_workbook.xlsx`
- `review_bundle/m2_review.csv`
- `review_bundle/execution_queue.csv`
- `review_bundle/full_survivors.csv`
- `review_bundle/charts_index.csv`
- `charts/*.html`

4. Summarize the run for the human reviewer:

- which families completed
- how many new M2 survivors were merged into the queue
- whether any rows are already `PENDING`, `EXECUTED`, `KILLED`, `STALE`, or `ERROR`
- where the workbook, queue CSV, and charts live
- whether a human decision is needed now

5. If the human has already edited the queue, rerun the nightly matrix or the targeted follow-up path and report:

- which rows were consumed
- which stayed `PENDING` because of budgets
- which rows became `EXECUTED`, `KILLED`, or `ERROR`

## Critical Rules

- Nightly family sweeps can run for a long time with little or no terminal output.
- Do not kill or restart the nightly matrix just because stdout is quiet for several minutes.
- Treat the process as alive if the parent nightly runner or a family subprocess is still present and consuming time or CPU.
- The broad nightly scout is intentionally `M1/M2` only.
- Do not treat the absence of fresh `M3/M4/M5` artifacts in the broad family runs as a failure.
- Deep validation belongs to queue-approved follow-up runs.
- Zero survivors is a valid research outcome.
- Do not treat empty M1 shortlists, empty M2 convergence results, or empty later-stage CSVs as fatal by default.
- Even on a zero-survivor night, the operator must still leave behind an initialized review surface:
  - `m2_human_review_queue.csv` with headers
  - `m2_human_review_history.csv`
  - `review_bundle/*.csv`
  - `human_review_workbook.xlsx`
- Terminal queue states matter:
  - `EXECUTED` and `KILLED` must not be silently reopened by a later scout.
  - Only refresh observational fields unless the human edited the row.
- The queue CSV is the source of truth for human decisions. The workbook/CSV bundle is only a projection.
- Prefer continuing the nightly loop with explicit zero-result artifacts over crashing the whole run.
- If you are validating queue execution bugs after a full scout has already finished, prefer replaying `HumanReviewQueueManager.refresh_queue(...)` against the completed bundle instead of rerunning every family sweep.

## Human Review Contract

The reviewer should work in `m2_human_review_queue.csv` and may edit:

- `human_decision`
- `human_notes`
- `priority`
- `queue_status`

Supported `human_decision` values:

- `promote_to_m3`
- `retune`
- `expand_symbols`
- `kill`

When handing off to a human, always include the queue path, workbook path, and charts directory.

## Failure Handling

- If a family runner exits non-zero, capture the failing family, the exact exception, and whether the failure is:
  - operational data issue
  - empty-artifact handling bug
  - candidate-selection logic bug
  - queue/export integration bug
- Patch the smallest reusable layer that fixes the class of problem.
- Add a regression test before retrying the full loop.
- Re-run the relevant pytest subset before retrying the live nightly job.

## Automation Guidance

When creating a Codex automation for this workflow:

- use the repo root as the working directory
- call this skill explicitly in the automation prompt
- have the automation report:
  - final bundle path
  - queue path
  - workbook path
  - charts path
  - queue counts by status
  - any blocker that needs a human queue edit
- schedule it early enough that a human can review the queue afterward the same evening or next morning

## Good Output Shape

A good operator summary is short and concrete:

- nightly run status
- bundle path
- queue/workbook/charts paths
- survivor counts
- failures fixed or still blocking
- exact human action needed next, if any
