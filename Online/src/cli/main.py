"""Main entry point for the WorldReasoner unified CLI.

This module provides the `wr` command with subcommands for:
- Database management (wr db)
- Evidence pipeline (wr evidence) - to be added in Phase 3
- Forecasting (wr forecast) - to be added in Phase 3
- Benchmarking (wr benchmark) - to be added in Phase 3
- And more...

Usage:
    wr --help
    wr db stats
    wr db list questions --domain politics
"""

import sqlite3
from dataclasses import dataclass

import typer
import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console

from src.config import get_config
from src.cli.commands import (
    benchmark,
    db,
    evidence,
    finance,
    finance_experiment,
    forecast,
    graph,
    question,
)
from src.core.database import GenericDatabase


@dataclass(frozen=True, slots=True)
class CliContextState:
    """Typed state shared by the root callback and nested commands."""

    verbose: bool


# Create the main Typer app
app = typer.Typer(
    name="wr",
    help="WorldReasoner CLI - LLM Forecasting Research & Pipeline Management",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# Register command groups
app.add_typer(db.app, name="db", help="Database management commands")
app.add_typer(
    question.app, name="question", help="Question management and collection commands"
)
app.add_typer(evidence.app, name="evidence", help="Evidence pipeline commands")
app.add_typer(forecast.app, name="forecast", help="Forecasting commands")
app.add_typer(graph.app, name="graph", help="Graph building and audit commands")
app.add_typer(benchmark.app, name="benchmark", help="LLM benchmark research commands")

finance_experiment.register_finance_experiment_commands(finance.app)
app.add_typer(
    finance.app, name="finance", help="Offline finance seed and pipeline commands"
)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """WorldReasoner CLI for forecasting research and pipeline management.

    The unified CLI consolidates all example scripts and database management
    into a single command-line interface with consistent UX.

    Examples:
        wr db stats
        wr db list questions --domain politics
        wr db show question q_abc123
        wr db clear-evidence q_abc123 --dry-run

    For help on any command:
        wr <command> --help
    """
    ctx.obj = CliContextState(verbose=verbose)

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")

    # Finance commands own an explicit immutable DB path and must not touch the
    # legacy default database initialized by the callback below.
    if ctx.invoked_subcommand == "finance":
        return

    # Ensure all database tables exist before any subcommand runs
    try:
        cfg = get_config()
        db: GenericDatabase[BaseModel] = GenericDatabase(cfg.database.db_path)
        tables = db.initialize_all_tables()
        if verbose:
            console.print(f"[dim]Initialized database; ensured {tables} tables[/dim]")
    except (
        OSError,
        sqlite3.Error,
        ValueError,
        ValidationError,
        yaml.YAMLError,
    ) as error:
        console.print(
            f"[yellow]Warning:[/yellow] Failed to initialize database tables: {error}"
        )


def cli():
    """Entry point for the CLI when installed via pip."""
    app()


if __name__ == "__main__":
    app()
