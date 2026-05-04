"""Generic eval harness.

Loops generic over a ``Ranker`` (Phase 2 of Recall ships five rankers —
``recall.retrieve.{semantic,substring,token_overlap,bm25,fuzzy}``).

Runtime discipline (per CLAUDE.md "Phase 2 gating rules"):
  - target ≤ 60 s for one ranker; ≤ 90 s for ``--ranker all`` on M-series Mac
  - soft warning at 90 s (printed by CLI)
  - hard failure at 120 s (raised here as ``EvalRuntimeError``)

The hard fail in the harness — not just CI — is what makes an
accidental 10× slowdown trip the discipline before CI even sees it.
``RECALL_EVAL_HARD_FAIL_S`` env override raises the ceiling for CI
runners specifically without softening the local 120 s gate.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from recall.retrieve.base import Ranker

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
    """End-to-end metrics + runtime breakdown for one eval run.

    One ``EvalResult`` per ``(dataset, ranker)`` pair. The CLI's
    ``--ranker all`` mode produces five ``EvalResult`` instances per
    invocation, one per ranker.
    """

    dataset: str
    ranker: str
    n_queries: int
    n_corpus: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    runtime_seconds: float
    runtime_breakdown: dict[str, float]
    model_name: str | None
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
    ranker_factory: Callable[[], Ranker],
    k_max: int = 5,
) -> EvalResult:
    """Construct ranker, build index, run search loop, compute metrics.

    ``ranker_factory`` is called once inside the timed region so the
    init duration (model load, etc.) shows up in the runtime breakdown
    — important for first-run vs cached-run distinction on the
    semantic ranker, and the right place for the lexical rankers'
    near-zero init time to be visible too.
    """
    breakdown: dict[str, float] = {}
    start_total = time.perf_counter()

    # --- Stage 1: instantiate ranker (model load for semantic; ~0 for lexical) ---
    t0 = time.perf_counter()
    ranker = ranker_factory()
    breakdown["init"] = time.perf_counter() - t0

    # --- Stage 2: build the in-memory index ---
    corpus = list(dataset.corpus())
    n_corpus = len(corpus)
    t0 = time.perf_counter()
    ranker.index(corpus)
    breakdown["index"] = time.perf_counter() - t0

    # --- Stage 3: search loop ---
    cases = list(dataset.cases())
    n_queries = len(cases)
    if n_queries == 0:
        raise ValueError(f"dataset {dataset.name!r} has no eval cases")
    queries = [c.nl_query for c in cases]
    t0 = time.perf_counter()
    all_top_ids = ranker.search(queries, k=k_max)
    breakdown["search"] = time.perf_counter() - t0

    if len(all_top_ids) != n_queries:
        raise ValueError(
            f"ranker {ranker.name!r} returned {len(all_top_ids)} result-lists for "
            f"{n_queries} queries"
        )

    # --- Stage 4: metrics ---
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    total_gold = 0

    for case, top_ids in zip(cases, all_top_ids, strict=True):
        top_commands = [corpus[i] for i in top_ids]
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

    recall_at_1 = hits_at_1 / n_queries
    recall_at_5 = hits_at_5 / n_queries
    mrr = sum(reciprocal_ranks) / n_queries

    avg_gold = total_gold / n_queries
    random_baseline = min(1.0, avg_gold * k_max / n_corpus)

    runtime_total = time.perf_counter() - start_total
    if runtime_total > HARD_RUNTIME_FAIL_S:
        raise EvalRuntimeError(
            f"eval ({ranker.name!r}) exceeded hard runtime budget: "
            f"{runtime_total:.1f}s > {HARD_RUNTIME_FAIL_S}s. breakdown: {breakdown}"
        )

    return EvalResult(
        dataset=dataset.name,
        ranker=ranker.name,
        n_queries=n_queries,
        n_corpus=n_corpus,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        mrr=mrr,
        runtime_seconds=runtime_total,
        runtime_breakdown=breakdown,
        model_name=getattr(ranker, "model_name", None),
        model_revision=getattr(ranker, "model_revision", None),
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
