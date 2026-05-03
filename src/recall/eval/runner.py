"""Generic eval harness.

Decoupled from any specific dataset — datasets implement the ``Dataset``
Protocol and the runner indexes their corpus and queries them.

Runtime discipline (per CLAUDE.md "Phase 2 gating rules"):
  - target: <= 60s on M-series Mac for full nl2bash
  - soft warning at 90s (printed by CLI)
  - hard failure at 120s (raised here as ``EvalRuntimeError``)

The hard fail is what makes the runtime gate live in the harness itself,
not just CI — so an accidental 10x slowdown trips the discipline before
CI even sees it.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import sqlite_vec

from recall.embed import Embedder

# Runtime gates (seconds). See module docstring.
SOFT_RUNTIME_WARN_S = 90.0
# Hard-fail default is 120s for local dev. CI runners (especially GitHub-hosted
# Linux without GPU) can be ~7x slower than M-series Mac because
# sentence-transformers falls back to CPU there while it uses MPS on the Mac.
# `RECALL_EVAL_HARD_FAIL_S` env override lets CI raise the ceiling without
# softening the local discipline gate.
HARD_RUNTIME_FAIL_S = float(os.environ.get("RECALL_EVAL_HARD_FAIL_S", "120"))


@dataclass(frozen=True)
class EvalCase:
    """One eval query + its multi-reference gold commands."""

    nl_query: str
    gold_commands: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    """End-to-end metrics + runtime breakdown for one eval run."""

    dataset: str
    n_queries: int
    n_corpus: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    runtime_seconds: float
    runtime_breakdown: dict[str, float]
    model_name: str
    model_revision: str | None
    random_baseline_recall_at_5: float


class Dataset(Protocol):
    """Eval dataset surface: a name, a list of cases, and a corpus to search."""

    name: str

    def cases(self) -> Sequence[EvalCase]: ...
    def corpus(self) -> Sequence[str]: ...


class EvalRuntimeError(Exception):
    """Raised when an eval run exceeds the hard runtime budget."""


def run_eval(
    dataset: Dataset,
    embedder: Embedder | None = None,
    k_max: int = 5,
) -> EvalResult:
    """Embed corpus, build in-memory sqlite-vec index, run queries, report metrics.

    ``embedder`` is optional; if omitted, a default ``Embedder()`` is constructed
    inside the timed region (so the model-load duration shows up in the runtime
    breakdown — important for first-run vs cached-run distinction).
    """
    breakdown: dict[str, float] = {}
    start_total = time.perf_counter()

    # --- Stage 1: model load (separate timing for first-run vs cached visibility) ---
    t0 = time.perf_counter()
    if embedder is None:
        embedder = Embedder()
    breakdown["model_load"] = time.perf_counter() - t0

    # --- Stage 2: embed corpus ---
    corpus = list(dataset.corpus())
    n_corpus = len(corpus)
    t0 = time.perf_counter()
    corpus_emb = embedder.encode(corpus)
    breakdown["corpus_embed"] = time.perf_counter() - t0

    # --- Stage 3: build :memory: sqlite-vec index ---
    t0 = time.perf_counter()
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        f"CREATE VIRTUAL TABLE vec USING vec0("
        f"id INTEGER PRIMARY KEY, embedding FLOAT[{embedder.dim}])"
    )
    conn.executemany(
        "INSERT INTO vec(id, embedding) VALUES (?, ?)",
        [(i, vec.tobytes()) for i, vec in enumerate(corpus_emb)],
    )
    breakdown["index_build"] = time.perf_counter() - t0

    # --- Stage 4: embed all queries ---
    cases = list(dataset.cases())
    n_queries = len(cases)
    if n_queries == 0:
        raise ValueError(f"dataset {dataset.name!r} has no eval cases")
    queries = [c.nl_query for c in cases]
    t0 = time.perf_counter()
    query_emb = embedder.encode(queries)
    breakdown["query_embed"] = time.perf_counter() - t0

    # --- Stage 5: search + metrics ---
    t0 = time.perf_counter()
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    total_gold = 0

    for case, qvec in zip(cases, query_emb, strict=True):
        rows = conn.execute(
            "SELECT id FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (qvec.tobytes(), k_max),
        ).fetchall()
        top_commands = [corpus[r[0]] for r in rows]
        gold_set = set(case.gold_commands)
        total_gold += len(gold_set)

        rank: int | None = None
        for r, cmd in enumerate(top_commands, start=1):
            if cmd in gold_set:
                rank = r
                break

        if rank is not None:
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    breakdown["search"] = time.perf_counter() - t0
    conn.close()

    recall_at_1 = hits_at_1 / n_queries
    recall_at_5 = hits_at_5 / n_queries
    mrr = sum(reciprocal_ranks) / n_queries

    # Random baseline analytical formula for "any gold in top k":
    # exact: 1 - C(n-g, k) / C(n, k); approximation g*k/n is tight for g << n.
    avg_gold = total_gold / n_queries
    random_baseline = min(1.0, avg_gold * k_max / n_corpus)

    runtime_total = time.perf_counter() - start_total

    if runtime_total > HARD_RUNTIME_FAIL_S:
        raise EvalRuntimeError(
            f"eval exceeded hard runtime budget: {runtime_total:.1f}s > "
            f"{HARD_RUNTIME_FAIL_S}s. breakdown: {breakdown}"
        )

    return EvalResult(
        dataset=dataset.name,
        n_queries=n_queries,
        n_corpus=n_corpus,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        mrr=mrr,
        runtime_seconds=runtime_total,
        runtime_breakdown=breakdown,
        model_name=embedder.model_name,
        model_revision=embedder.model_revision,
        random_baseline_recall_at_5=random_baseline,
    )


__all__ = (
    "HARD_RUNTIME_FAIL_S",
    "SOFT_RUNTIME_WARN_S",
    "Dataset",
    "EvalCase",
    "EvalResult",
    "EvalRuntimeError",
    "run_eval",
)
