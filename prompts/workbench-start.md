# Hypothesis Workbench — Session Start

Paste this into a fresh Claude Code session at the root of `mala_v1/`,
fill in the "Idea source" block at the bottom, and send.

---

I'm starting a Hypothesis Workbench session. Please work through the
following steps **in order**:

## Step 1 — Load context

Read these files before doing anything else:

1. `CLAUDE.md` — specifically the `## Hypothesis Workbench` section, for
   accumulated meta-lessons from prior sessions and the feasibility tag
   definitions.
2. `src/strategy/factory.py` lines 113–125 — the current strategy
   registry (`_NAMED_BUILDERS` dict + `available_strategy_names`).
3. `src/research/stages/holdout.py` — how M3 walk-forward is invoked, so
   you know what a testable config needs to look like.
4. `src/config.py` — the Pydantic settings class (Polygon key access).

## Step 2 — Read the idea

The idea source is at the bottom of this prompt. It will be either:
- A pasted `Idea_Intake` row (with columns like `source_url`,
  `one_line_thesis`, `extracted_rules`, `feasibility_tag`), or
- A free-text hypothesis I typed myself.

If the fields are ambiguous or the rules are implicit, ask me one or two
clarifying questions before proceeding. Don't guess silently.

## Step 3 — Produce a feasibility report

Format exactly like this:

```
## Feasibility Report

**Summary (1 sentence):** {restate the hypothesis in your own words}

**Closest existing strategy:** {one of the registered class names, or "none"}

**Feasibility tag:** {config-only | new-class | new-feature}

**If config-only:**
  - Strategy: {name}
  - Draft YAML or evaluate_config call:
    ```yaml
    ...
    ```
  - Justification for the parameter choices
  - What M1/M2 thresholds this is likely to hit, and where it might fail

**If new-class:**
  - Proposed class name and file path
  - What methods it needs (generate_signals signature, any helpers)
  - Estimated line count
  - Test file plan (what cases to cover)
  - Dependencies on existing Newton features

**If new-feature:**
  - Which Newton feature (or Oracle metric) to add
  - Why existing features can't represent this
  - Rough scope (lines of code + tests)
  - What breaks downstream if the feature is added

**Concerns (always include this section):**
  - Things that could make this hypothesis misleading or hard to measure
  - Data availability or quality issues
  - Signal-count risk (too rare to validate)
  - Correlation with existing catalog entries
```

## Step 4 — Wait

**Do NOT run `evaluate_config`, draft code, or touch files until I say
"proceed".** I will read your report and either say "proceed", "revise
with X", or "kill — reason".

## Step 5 — On proceed

- If `config-only`: run the evaluate_config call, show me M1/M2 output,
  we decide kill / iterate / promote-to-M3.
- If `new-class` or `new-feature`: draft the code and a test file, wait
  for my review, then run the test, then the M1/M2 eval.

## Step 6 — On promote

- Create `mala_v1/strategies/{catalog_key}.md` with the hypothesis in my
  own words, the source URL, the rules, feasibility notes, and M1/M2
  results. This is the strategy's lore file — the compounding journal.
- If there's an `Idea_Intake` row, remind me to set `triage_status =
  promote_to_m3` in the gSheet.

## Step 7 — On kill

- If there's an `Idea_Intake` row, remind me to set `triage_status =
  killed` with a one-line reason.
- No lore file needed.

## Step 8 — Session end

Before we wrap, ask me: "Anything worth adding to CLAUDE.md's
Cross-session lessons?" If yes, append one or two lines of meta-takeaway
that future-you (in a fresh session) would benefit from. Keep it short.

---

## Idea source

<!-- Paste an Idea_Intake row here, OR write a free-text hypothesis below. -->

{IDEA GOES HERE}
