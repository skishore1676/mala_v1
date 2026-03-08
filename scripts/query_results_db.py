#!/usr/bin/env python3
"""
Quick query helper for local results DB.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.table import Table


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query strategy results from SQLite store.")
    parser.add_argument("--db-path", default="data/results/results.db")
    parser.add_argument("--artifact-type", default="", help="Optional artifact filter.")
    parser.add_argument("--ticker", default="", help="Optional ticker filter.")
    parser.add_argument("--strategy", default="", help="Optional strategy filter.")
    parser.add_argument("--direction", default="", help="Optional direction filter.")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/]")
        return

    filters = []
    params: list[object] = []
    if args.artifact_type:
        filters.append("artifact_type = ?")
        params.append(args.artifact_type)
    if args.ticker:
        filters.append("ticker = ?")
        params.append(args.ticker.upper())
    if args.strategy:
        filters.append("strategy = ?")
        params.append(args.strategy)
    if args.direction:
        filters.append("direction = ?")
        params.append(args.direction)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"""
        SELECT
            run_id, artifact_type, ticker, strategy, direction,
            signals, confidence, exp_r, decision, source_path, created_at
        FROM artifact_rows
        {where_sql}
        ORDER BY exp_r DESC, confidence DESC, signals DESC
        LIMIT ?
    """
    params.append(args.limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print("[yellow]No rows matched filters.[/]")
        return

    table = Table(title=f"Top Results ({db_path})", show_lines=True)
    table.add_column("Run")
    table.add_column("Artifact")
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("Signals", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Exp(R)", justify="right")
    table.add_column("Decision")
    for row in rows:
        conf = "-" if row["confidence"] is None else f"{float(row['confidence']):.4f}"
        exp_r = "-" if row["exp_r"] is None else f"{float(row['exp_r']):+.4f}"
        sig = "-" if row["signals"] is None else str(int(row["signals"]))
        table.add_row(
            str(row["run_id"]),
            str(row["artifact_type"]),
            str(row["ticker"] or ""),
            str(row["strategy"] or ""),
            str(row["direction"] or ""),
            sig,
            conf,
            exp_r,
            str(row["decision"] or ""),
        )
    console.print(table)


if __name__ == "__main__":
    main()
