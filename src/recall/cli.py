"""Recall CLI (typer-based).

Public entry point: ``recall = "recall.cli:main"`` in ``pyproject.toml``.

Phase 2 ships ``recall eval``. Subcommands for ``recall index`` (the
indexer, Commit 2.8) and ``recall serve`` (the MCP stdio server, Phase 3)
are declared as placeholders so ``recall --help`` lists the eventual
surface — and so typer doesn't auto-collapse to single-command mode
(see CLAUDE.md "Composition is where bugs live", 2.5 named instance).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import (
    EvalResult,
    EvalRuntimeError,
    run_eval,
)
from recall.retrieve.base import Ranker
from recall.retrieve.bm25 import Bm25Ranker
from recall.retrieve.fuzzy import FuzzyRanker
from recall.retrieve.semantic import SemanticRanker
from recall.retrieve.substring import NaiveSubstringRanker
from recall.retrieve.token_overlap import TokenOverlapRanker

# Cumulative all-rankers budget (separate from the per-ranker hard fail in
# runner.HARD_RUNTIME_FAIL_S — that's an anomaly detector for any single
# ranker blowing past 120s; this is the aggregate across the all-rankers
# loop). Soft warning printed; hard fail raises.
ALL_RANKERS_SOFT_WARN_S = 120.0
ALL_RANKERS_HARD_FAIL_S = float(os.environ.get("RECALL_EVAL_ALL_HARD_FAIL_S", "180"))

app = typer.Typer(no_args_is_help=True, help="Recall — semantic shell history.")


# Ranker registry: (name, factory). Order here is the run order, not the
# display order — the all-rankers mode runs them in this order (semantic
# first so its model load + corpus embed warms HF/torch caches before
# the lexical rankers run; lexical rankers cheap so order doesn't matter
# among them) and prints the summary sorted by recall@5 descending.
_RANKER_FACTORIES: dict[str, Callable[[], Ranker]] = {
    "semantic": lambda: SemanticRanker(),
    "naive": lambda: NaiveSubstringRanker(),
    "token-overlap": lambda: TokenOverlapRanker(),
    "bm25": lambda: Bm25Ranker(),
    "fuzzy": lambda: FuzzyRanker(),
}
_DEFAULT_RANKER_ORDER: tuple[str, ...] = (
    "semantic",
    "naive",
    "token-overlap",
    "bm25",
    "fuzzy",
)


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


def _print_per_ranker(result: EvalResult, idx: int, total: int) -> None:
    typer.echo("")
    typer.echo(f"[{idx}/{total}] {result.ranker} ranker")
    typer.echo(_format_breakdown(result.runtime_breakdown, result.runtime_seconds))
    typer.echo(f"   Recall@1: {result.recall_at_1:.4f}")
    typer.echo(f"   Recall@5: {result.recall_at_5:.4f}")
    typer.echo(f"        MRR: {result.mrr:.4f}")


def _print_summary(results: Sequence[EvalResult], total_wall_clock: float) -> None:
    """Sorted-by-recall@5-desc table with deltas to semantic.

    Format matches what the README / commit-message will lift verbatim:

        Semantic    : recall@5 = 0.4486    (baseline)
        BM25        : recall@5 = 0.XXXX    (semantic is N.NX× better)
        ...

    The multiplier in parens is the value-prop. Computing it here means
    it lands in commit history at the point of measurement.
    """
    by_recall = sorted(results, key=lambda r: r.recall_at_5, reverse=True)
    semantic = next((r for r in results if r.ranker == "semantic"), None)
    sem_r5 = semantic.recall_at_5 if semantic is not None else None

    # Pretty-name lookup for the headline column.
    pretty = {
        "semantic": "Semantic",
        "naive": "Naive",
        "token-overlap": "Token",
        "bm25": "BM25",
        "fuzzy": "Fuzzy",
    }
    typer.echo("")
    typer.echo("=== Summary (sorted by recall@5 desc) ===")
    for r in by_recall:
        name = pretty.get(r.ranker, r.ranker)
        if r.ranker == "semantic" or sem_r5 is None:
            tag = "(baseline)" if r.ranker == "semantic" else ""
        elif r.recall_at_5 == 0:
            tag = "(semantic is ∞× better)"
        else:
            mult = sem_r5 / r.recall_at_5
            if mult >= 1.0:
                tag = f"(semantic is {mult:.2f}× better)"
            else:
                tag = f"(BEATS semantic by {1.0 / mult:.2f}×)"
        typer.echo(f"  {name:<10}: recall@5 = {r.recall_at_5:.4f}    {tag}")
    typer.echo("")
    typer.echo(f"Wall-clock total: {total_wall_clock:.2f}s")


def _resolve_rankers(name: str) -> tuple[str, ...]:
    if name == "all":
        return _DEFAULT_RANKER_ORDER
    if name not in _RANKER_FACTORIES:
        valid = ", ".join(["all", *sorted(_RANKER_FACTORIES.keys())])
        typer.secho(
            f"unknown ranker: {name!r}; expected one of {valid}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    return (name,)


def _result_record(result: EvalResult, commit_sha: str, timestamp: str) -> dict[str, Any]:
    return {
        "commit_sha": commit_sha,
        "timestamp": timestamp,
        "dataset": result.dataset,
        "ranker": result.ranker,
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


@app.command(name="eval")
def eval_cmd(
    dataset: Annotated[
        str,
        typer.Option(help="Dataset name (only 'nl2bash' supported in this build)."),
    ] = "nl2bash",
    ranker: Annotated[
        str,
        typer.Option(
            help="Ranker to run: all | semantic | naive | token-overlap | bm25 | fuzzy. "
            "Default 'all' runs every ranker; one record per (commit, dataset, "
            "ranker) appended to the history file."
        ),
    ] = "all",
    output: Annotated[
        Path,
        typer.Option(
            help="Append result record(s) to this JSON file (history, not snapshot).",
        ),
    ] = Path("eval/results.json"),
    no_append: Annotated[
        bool,
        typer.Option("--no-append", help="Print results but do NOT append to history."),
    ] = False,
) -> None:
    """Run the eval harness and report retrieval metrics.

    Phase 2 cadence: every retrieval-touching commit must report all
    five rankers' recall@5. ``--ranker all`` (default) does that in one
    invocation; ``--ranker <name>`` is for iteration.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if dataset != "nl2bash":
        typer.secho(
            f"unknown dataset: {dataset!r}; only 'nl2bash' is supported in this build",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    ranker_names = _resolve_rankers(ranker)

    ds = Nl2BashDataset()
    typer.echo(f"Running eval on {ds.name}: {len(ds.corpus())} corpus / {len(ds.cases())} queries")
    typer.echo(f"Rankers: {', '.join(ranker_names)}")

    results: list[EvalResult] = []
    overall_start = time.perf_counter()
    for i, name in enumerate(ranker_names, start=1):
        try:
            result = run_eval(ds, _RANKER_FACTORIES[name])
        except EvalRuntimeError as e:
            typer.secho(
                f"eval HARD-FAILED on runtime budget for {name!r}: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from e
        results.append(result)
        _print_per_ranker(result, i, len(ranker_names))
        # Cumulative-budget check: per-ranker hard fail is in runner.py
        # (anomaly detector for any single ranker); this is the aggregate
        # gate across the all-rankers loop, separately tunable.
        cumulative = time.perf_counter() - overall_start
        if cumulative > ALL_RANKERS_HARD_FAIL_S:
            typer.secho(
                f"cumulative all-ranker wall-clock {cumulative:.1f}s > "
                f"{ALL_RANKERS_HARD_FAIL_S}s — aggregate hard runtime budget "
                "exceeded. Override via RECALL_EVAL_ALL_HARD_FAIL_S env var "
                "(CI sets 600s; local default 180s).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

    overall_total = time.perf_counter() - overall_start

    _print_summary(results, overall_total)

    if overall_total > ALL_RANKERS_SOFT_WARN_S:
        typer.secho(
            f"  ⚠  exceeded all-rankers soft runtime warning threshold "
            f"({ALL_RANKERS_SOFT_WARN_S}s); revisit budget before next commit",
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
    sha = _git_head_sha()
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    new_records = [_result_record(r, sha, ts) for r in results]
    history.extend(new_records)
    output.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    typer.echo("")
    typer.echo(f"appended {len(new_records)} record(s) to {output}")


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
