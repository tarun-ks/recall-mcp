"""Recall CLI (typer-based).

Public entry point: ``recall = "recall.cli:main"`` in ``pyproject.toml``.

Phase 2 ships only ``recall eval``. Subcommands for ``recall index`` (the
indexer) and ``recall serve`` (the MCP server entry point) land in later
commits — they're declared here as placeholders so ``recall --help`` lists
the eventual surface.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import (
    SOFT_RUNTIME_WARN_S,
    EvalRuntimeError,
    run_eval,
)

app = typer.Typer(no_args_is_help=True, help="Recall — semantic shell history.")


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _format_breakdown(breakdown: dict[str, float], total: float) -> str:
    lines = []
    for name, secs in breakdown.items():
        lines.append(f"  {name:>14}: {secs:>6.2f}s")
    lines.append(f"  {'total':>14}: {total:>6.2f}s")
    return "\n".join(lines)


@app.command(name="eval")
def eval_cmd(
    dataset: Annotated[
        str,
        typer.Option(help="Dataset name (only 'nl2bash' supported in 2.5)."),
    ] = "nl2bash",
    output: Annotated[
        Path,
        typer.Option(help="Append the result record to this JSON file (history, not snapshot)."),
    ] = Path("eval/results.json"),
    no_append: Annotated[
        bool,
        typer.Option("--no-append", help="Print the result but do NOT append to history."),
    ] = False,
) -> None:
    """Run the eval harness against a dataset and report retrieval metrics.

    Output: human-readable summary on stdout + structured record appended
    to ``eval/results.json`` (an append-only history of all eval runs).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if dataset != "nl2bash":
        typer.secho(
            f"unknown dataset: {dataset!r}; only 'nl2bash' is supported in this build",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    ds = Nl2BashDataset()
    typer.echo(f"Running eval on {ds.name}: {len(ds.corpus())} corpus / {len(ds.cases())} queries")

    try:
        result = run_eval(ds)
    except EvalRuntimeError as e:
        typer.secho(f"eval HARD-FAILED on runtime budget: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo("")
    typer.echo(f"Recall@1:                 {result.recall_at_1:.4f}")
    typer.echo(f"Recall@5:                 {result.recall_at_5:.4f}")
    typer.echo(f"MRR:                      {result.mrr:.4f}")
    typer.echo(f"Random baseline recall@5: {result.random_baseline_recall_at_5:.6f}")
    typer.echo(
        f"  (semantic delta over random: "
        f"{result.recall_at_5 - result.random_baseline_recall_at_5:+.4f})"
    )
    typer.echo("")
    typer.echo("Runtime breakdown:")
    typer.echo(_format_breakdown(result.runtime_breakdown, result.runtime_seconds))
    if result.runtime_seconds > SOFT_RUNTIME_WARN_S:
        typer.secho(
            f"  ⚠  exceeded soft runtime warning threshold ({SOFT_RUNTIME_WARN_S}s)",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if no_append:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if output.exists():
        history = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            typer.secho(
                f"{output} exists but is not a JSON array — refusing to overwrite",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2)
    record: dict[str, Any] = {
        "commit_sha": _git_head_sha(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": result.dataset,
        "recall_at_1": result.recall_at_1,
        "recall_at_5": result.recall_at_5,
        "mrr": result.mrr,
        "runtime_seconds": round(result.runtime_seconds, 2),
        "runtime_breakdown": {k: round(v, 2) for k, v in result.runtime_breakdown.items()},
        "model_name": result.model_name,
        "model_revision": result.model_revision,
        "random_baseline_recall_at_5": result.random_baseline_recall_at_5,
        "n_queries": result.n_queries,
        "n_corpus": result.n_corpus,
    }
    history.append(record)
    output.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    typer.echo("")
    typer.echo(f"appended record to {output}")


@app.command(name="index")
def index_cmd() -> None:
    """Index shell history into the local SQLite store. Lands in Commit 2.8."""
    typer.secho(
        "recall index is not yet implemented (lands in Commit 2.8 — the indexer).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(2)


@app.command(name="serve")
def serve_cmd() -> None:
    """Run the MCP stdio server. Lands in Phase 3 (Commit 3.9)."""
    typer.secho(
        "recall serve is not yet implemented (lands in Phase 3 — the MCP server).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(2)


def main() -> None:
    """Entry point for the ``recall`` script."""
    app()
