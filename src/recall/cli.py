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

from recall.eval.dogfood import DogfoodDataset, DogfoodQueryMeta
from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import (
    Dataset,
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
# `naive` was dropped from the default at 2.7.5-hotfix per the locked
# throughput-gate fallback chain (CLAUDE.md "Eval all-rankers wall-clock
# tension"). The deterministic-tie-breaking fix added ~57s to semantic's
# CI Linux runtime; naive's recall@5 of 0.0857 (the trivial-floor
# baseline) was locked into the calibrated table and contributes minimal
# evaluative signal beyond what's already documented. `recall eval
# --ranker naive` still works explicitly — it's just not on the default
# critical-path eval that runs every CI cycle.
_DEFAULT_RANKER_ORDER: tuple[str, ...] = (
    "semantic",
    "token-overlap",
    "bm25",
    "fuzzy",
)

# Dataset registry. The ``all`` option resolves to (nl2bash, dogfood) — both
# numbers must move together for the project's value-prop to be defensible
# (CLAUDE.md "Phase 2 gating rules"). Dogfood lazily-loads from disk
# (eval/dogfood.toml) so a missing/malformed file is a clear single error.
_DATASET_FACTORIES: dict[str, Callable[[], Dataset]] = {
    "nl2bash": Nl2BashDataset,
    "dogfood": DogfoodDataset,
}
_DEFAULT_DATASET_ORDER: tuple[str, ...] = ("nl2bash", "dogfood")


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


def _resolve_datasets(name: str) -> tuple[str, ...]:
    if name == "all":
        return _DEFAULT_DATASET_ORDER
    if name not in _DATASET_FACTORIES:
        valid = ", ".join(["all", *sorted(_DATASET_FACTORIES.keys())])
        typer.secho(
            f"unknown dataset: {name!r}; expected one of {valid}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    return (name,)


def _print_dogfood_per_query(
    queries: Sequence[DogfoodQueryMeta],
    results: Sequence[EvalResult],
) -> None:
    """Per-query inline format for dogfood: semantic vs best-lexical comparison.

    The format is the headline data point — with a 5-query existence test,
    aggregate recall@5 is signal but per-query pass/fail is the actual
    evidence. Locked at 2.6.5 in CLAUDE.md "Phase 2 gating rules" via
    the dogfood-specific comparison-table requirement.

    Output shape (semantic + bm25 both available):

        === Dogfood (5 queries, semantic vs best lexical) ===
        [L] #42 grep useAgent           : semantic PASS@1   bm25 PASS@1   tie
        [M] #19 webhook-platform deploy : semantic PASS@2   bm25 PASS@4   semantic-wins
        [H] #52 mwbe                    : semantic PASS@4   bm25 FAIL     semantic-only ★
        [H] #71 processed contracts     : semantic FAIL     bm25 FAIL     both-fail

        semantic recall@5: 4/5  bm25 recall@5: 3/5
        semantic-only wins: 1 (#52, the domain-knowledge test)

    If only semantic ran (no bm25 in results), prints semantic-only rows
    without the comparison verdict — the inline format still adds value
    over aggregate metrics for a 5-query set, just less than with both.
    """
    sem = next((r for r in results if r.ranker == "semantic"), None)
    bm25 = next((r for r in results if r.ranker == "bm25"), None)
    if sem is None and bm25 is None:
        # Neither comparison ranker ran; skip the inline format.
        return

    typer.echo("")
    if sem is not None and bm25 is not None:
        typer.echo("=== Dogfood (5 queries, semantic vs best lexical) ===")
    else:
        present = "semantic" if sem is not None else "bm25"
        typer.echo(f"=== Dogfood (5 queries, {present} only) ===")

    def fmt_rank(r: EvalResult | None, idx: int) -> str:
        if r is None:
            return "-"
        rank = r.per_query_ranks[idx]
        return f"PASS@{rank}" if rank is not None and rank <= 5 else "FAIL"

    def verdict(s: str | None, b: str | None) -> str:
        # Pre-condition: both s and b are non-None (caller guards on sem+bm25).
        if s == "FAIL" and b == "FAIL":
            return "both-fail"
        if s == "FAIL":
            return "bm25-only"
        if b == "FAIL":
            return "semantic-only"
        # Both PASS@N — compare ranks (s/b are PASS@N strings; parse N).
        sn = int(s.split("@")[1]) if s else 0
        bn = int(b.split("@")[1]) if b else 0
        if sn == bn:
            return "tie"
        return "semantic-wins" if sn < bn else "bm25-wins"

    # ★ marker shown next to short label, always (regardless of verdict) — so a
    # reader can always find the headline query in the table even when its
    # row isn't a semantic-only win. (The 2.6.5 actual data has #52 fail on
    # both rankers — the value-prop holds via #19, #48, #71 — and the star
    # is still useful for spotting which row the design centered on.)
    def short_label(q: DogfoodQueryMeta) -> str:
        return f"{q.short} ★" if q.star else q.short

    semantic_only_ids: list[int] = []
    starred_ids: list[int] = []
    for idx, q in enumerate(queries):
        s = fmt_rank(sem, idx)
        b = fmt_rank(bm25, idx)
        if q.star:
            starred_ids.append(q.id)
        v_text = ""
        if sem is not None and bm25 is not None:
            v = verdict(s, b)
            if v == "semantic-only":
                semantic_only_ids.append(q.id)
            v_text = f"  {v}"
        label = short_label(q)
        if sem is not None and bm25 is not None:
            typer.echo(f"  [{q.tier}] #{q.id:<2} {label:<28}: semantic {s:<8}  bm25 {b:<8}{v_text}")
        elif sem is not None:
            typer.echo(f"  [{q.tier}] #{q.id:<2} {label:<28}: semantic {s}")
        else:
            typer.echo(f"  [{q.tier}] #{q.id:<2} {label:<28}: bm25 {b}")

    typer.echo("")
    summary_parts = []
    if sem is not None:
        summary_parts.append(
            f"semantic recall@5: {int(sem.recall_at_5 * sem.n_queries)}/{sem.n_queries}"
        )
    if bm25 is not None:
        summary_parts.append(
            f"bm25 recall@5: {int(bm25.recall_at_5 * bm25.n_queries)}/{bm25.n_queries}"
        )
    typer.echo("  " + "  ".join(summary_parts))

    if sem is not None and bm25 is not None:
        if semantic_only_ids:
            id_list = ", ".join(f"#{i}" for i in semantic_only_ids)
            note = f"semantic-only wins: {len(semantic_only_ids)} ({id_list})"
            # If the starred query is among the semantic-only wins, call it
            # out — that's the strongest possible dogfood headline.
            if starred_ids and any(s in semantic_only_ids for s in starred_ids):
                hits = [s for s in starred_ids if s in semantic_only_ids]
                hit_str = ", ".join(f"#{i}" for i in hits)
                note += f" — includes the domain-knowledge test ({hit_str} ★)"
            typer.echo(f"  {note}")
        else:
            typer.echo("  semantic-only wins: 0")


def _result_record(result: EvalResult, commit_sha: str, timestamp: str) -> dict[str, Any]:
    record: dict[str, Any] = {
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
    # Persist per-query ranks for dogfood only — the per-query PASS→FAIL
    # regression detector (.github/scripts/check_eval_regression.py) needs
    # this signal. Skipping nl2bash keeps the file ~10× smaller given its
    # ~10k-query footprint vs dogfood's 5.
    if result.dataset == "dogfood":
        record["per_query_ranks"] = list(result.per_query_ranks)
    return record


@app.command(name="eval")
def eval_cmd(
    dataset: Annotated[
        str,
        typer.Option(
            help="Dataset to run: all | nl2bash | dogfood. Default 'all' runs both "
            "(public benchmark + real-history dogfood per CLAUDE.md Phase 2 gating)."
        ),
    ] = "all",
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

    Phase 2 cadence: every retrieval-touching commit reports all five
    rankers' recall@5 on both nl2bash (public benchmark) and dogfood
    (real-history). ``--dataset all --ranker all`` (the defaults) does
    that in one invocation; one record per (commit, dataset, ranker)
    is appended to the history file.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    dataset_names = _resolve_datasets(dataset)
    ranker_names = _resolve_rankers(ranker)

    all_results: list[EvalResult] = []
    overall_start = time.perf_counter()

    for dsname in dataset_names:
        ds = _DATASET_FACTORIES[dsname]()
        typer.echo("")
        typer.echo(
            f"=== Running eval on {ds.name}: "
            f"{len(ds.corpus())} corpus / {len(ds.cases())} queries ==="
        )
        typer.echo(f"Rankers: {', '.join(ranker_names)}")

        ds_results: list[EvalResult] = []
        ds_start = time.perf_counter()
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
            ds_results.append(result)
            all_results.append(result)
            _print_per_ranker(result, i, len(ranker_names))
            # Cumulative-budget check across ALL datasets and rankers — the
            # aggregate gate is wall-clock-since-start, not per-dataset, so
            # a slow dogfood lane after a borderline-OK nl2bash lane still
            # trips the budget.
            cumulative = time.perf_counter() - overall_start
            if cumulative > ALL_RANKERS_HARD_FAIL_S:
                typer.secho(
                    f"cumulative wall-clock {cumulative:.1f}s > "
                    f"{ALL_RANKERS_HARD_FAIL_S}s — aggregate hard runtime budget "
                    "exceeded. Override via RECALL_EVAL_ALL_HARD_FAIL_S env var "
                    "(CI sets 600s; local default 180s).",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(1)

        ds_total = time.perf_counter() - ds_start
        _print_summary(ds_results, ds_total)
        # Dogfood-specific per-query inline format — the headline data
        # point on a 5-query existence test (CLAUDE.md "Phase 2 gating").
        if dsname == "dogfood" and isinstance(ds, DogfoodDataset):
            _print_dogfood_per_query(ds.queries, ds_results)

    overall_total = time.perf_counter() - overall_start

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
    new_records = [_result_record(r, sha, ts) for r in all_results]
    history.extend(new_records)
    output.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    typer.echo("")
    typer.echo(f"appended {len(new_records)} record(s) to {output}")


@app.command(name="index")
def index_cmd(
    source: Annotated[
        str,
        typer.Option(
            help="Source(s) to index: all | zsh | bash | atuin. "
            "Default 'all' indexes every available source sequentially. "
            "Sources without a usable history (e.g. atuin not installed) "
            "are skipped with a warning."
        ),
    ] = "all",
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Drop and re-create commands/commands_vec/commands_fts before "
            "indexing (CLAUDE.md §4b). Salt preserved unless --new-salt also "
            "passed. Cursor metas reset; all sources re-index from scratch.",
        ),
    ] = False,
    new_salt: Annotated[
        bool,
        typer.Option(
            "--new-salt",
            help="Rotate the dedup salt (CLAUDE.md §4b). Requires --rebuild — "
            "rotating the salt without rebuilding leaves old-salt and new-salt "
            "hashes coexisting in one table, silently breaking dedup.",
        ),
    ] = False,
) -> None:
    """Index shell history into the local SQLite store at ``~/.recall/db.sqlite``.

    Pulls entries from each source's ``iter_entries`` (incremental from
    the per-source cursor in meta), runs each through the scrubber,
    embeds the scrubbed text via ``Embedder``, and writes
    (text_scrubbed, text_hash, vector, metadata) rows in 1024-row
    transactions (CLAUDE.md §2.8).
    """
    # Lazy-import indexer + sources here (not at module top) to keep
    # `recall --help`, `recall eval`, and pytest collection from paying
    # the model-load cost when the user just wanted the eval lane.
    # Same lesson as Embedder's lazy sentence-transformers import in 2.5.
    import logging

    from recall.db import connect, migrate, rotate_dedup_salt
    from recall.embed import Embedder
    from recall.indexer import index_sources
    from recall.indexer import rebuild as do_rebuild
    from recall.sources.atuin import AtuinSchemaError, AtuinSource
    from recall.sources.base import HistorySource
    from recall.sources.bash import BashSource
    from recall.sources.zsh import ZshSource

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if new_salt and not rebuild:
        typer.secho(
            "--new-salt without --rebuild is rejected: rotating the salt without "
            "rebuilding leaves old-salt and new-salt hashes coexisting in one "
            "table, silently breaking dedup. Pass --rebuild together.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    valid_sources = {"all", "zsh", "bash", "atuin"}
    if source not in valid_sources:
        typer.secho(
            f"unknown source: {source!r}; expected one of {sorted(valid_sources)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    conn = connect()
    migrate(conn)

    if rebuild:
        if new_salt:
            rotate_dedup_salt(conn)
        do_rebuild(conn)

    # Resolve sources. Each source factory either returns a ready-to-use
    # HistorySource or raises (missing histfile, atuin not installed, etc.)
    # in which case we skip it with a warning. Sequential per Q3.
    requested = [source] if source != "all" else ["zsh", "bash", "atuin"]
    active: list[HistorySource] = []
    for name in requested:
        try:
            if name == "zsh":
                active.append(ZshSource())
            elif name == "bash":
                active.append(BashSource())
            elif name == "atuin":
                active.append(AtuinSource())
        except (FileNotFoundError, AtuinSchemaError) as e:
            typer.secho(
                f"skipping {name}: {e}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    if not active:
        typer.secho(
            "no usable sources found. Did the histfiles get nuked?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    typer.echo(f"recall index: sources={[s.name for s in active]} rebuild={rebuild}")

    embedder = Embedder()
    result = index_sources(conn, active, embedder=embedder)

    typer.echo("")
    typer.echo("=== Indexing summary ===")
    typer.echo(f"  inserted:           {result.inserted}")
    typer.echo(f"  skipped (dedup):    {result.skipped_dedup}")
    typer.echo(f"  total processed:    {result.total_processed()}")
    typer.echo("  by source:")
    for src, n in result.by_source.items():
        typer.echo(f"    {src:<8} {n}")
    typer.echo("")
    typer.echo(f"  runtime:            {result.runtime_seconds:.2f}s")
    typer.echo(f"    embedder:         {result.embedder_seconds:.2f}s")
    typer.echo(f"    db writes:        {result.db_write_seconds:.2f}s")


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
