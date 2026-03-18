#!/usr/bin/env python3
"""Registry-backed orchestration preview for the phased refactor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research import ResearchOrchestrator, ResearchStage


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the new research orchestration layer.")
    parser.add_argument("--strategy", help="Tracked strategy name to inspect.")
    parser.add_argument(
        "--stage",
        default=ResearchStage.M1_DISCOVERY.value,
        choices=[stage.value for stage in ResearchStage],
        help="Research stage to plan.",
    )
    parser.add_argument(
        "--validation-set",
        action="store_true",
        help="Print the current refactor validation set instead of a single strategy plan.",
    )
    return parser.parse_args()


def print_validation_set(orchestrator: ResearchOrchestrator) -> None:
    table = Table(title="Refactor Validation Set", show_lines=True)
    table.add_column("Strategy")
    table.add_column("Status")
    table.add_column("Tickers")
    table.add_column("Directions")
    table.add_column("Eval")

    for entry in orchestrator.validation_entries():
        table.add_row(
            entry.name,
            entry.status.value,
            ", ".join(entry.tickers),
            ", ".join(entry.directions),
            entry.evaluation_mode,
        )
    console.print(table)


def print_strategy_plan(orchestrator: ResearchOrchestrator, strategy_name: str, stage: ResearchStage) -> None:
    entry = orchestrator.strategy_entry(strategy_name)
    actions = orchestrator.next_actions(stage)

    console.rule(f"[bold cyan]{entry.name}[/]")
    console.print(f"Status: [green]{entry.status.value}[/] | Eval mode: [bold]{entry.evaluation_mode}[/]")
    console.print(f"Tickers: {', '.join(entry.tickers) or '-'}")
    console.print(f"Directions: {', '.join(entry.directions) or '-'}")
    console.print(f"Required features: {', '.join(entry.required_features)}")

    if entry.parameter_space:
        params_table = Table(title="Parameter Space", show_lines=True)
        params_table.add_column("Parameter")
        params_table.add_column("Candidates")
        for key, values in entry.parameter_space.items():
            params_table.add_row(key, ", ".join(str(v) for v in values))
        console.print(params_table)

    actions_table = Table(title=f"Next Allowed Actions ({stage.value})", show_lines=True)
    actions_table.add_column("Action")
    actions_table.add_column("Agent")
    actions_table.add_column("Summary")
    for action in actions:
        actions_table.add_row(
            action.action,
            "run" if action.agent_can_run else "recommend",
            action.summary,
        )
    console.print(actions_table)


def main() -> None:
    args = parse_args()
    orchestrator = ResearchOrchestrator()

    if args.validation_set:
        print_validation_set(orchestrator)
        return

    if not args.strategy:
        raise SystemExit("Pass --validation-set or --strategy.")

    stage = ResearchStage(args.stage)
    print_strategy_plan(orchestrator, args.strategy, stage)


if __name__ == "__main__":
    main()
