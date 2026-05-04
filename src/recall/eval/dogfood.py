"""Dogfood eval dataset — real (NL, command) pairs from the project author's
zsh + bash history.

Companion to ``Nl2BashDataset``: nl2bash provides public-benchmark
credibility (numbers comparable to published results); dogfood provides
real-use-quality evidence on small-N, high-realism queries. Per
CLAUDE.md "Phase 2 gating rules": both numbers must move together for
the project to be worth shipping; one without the other is a partial
picture.

The data lives in ``eval/dogfood.toml`` next to ``eval/results.json``.
This module loads it via stdlib ``tomllib`` (no new dep), normalizes
single-vs-multi gold (``gold`` string ↔ ``golds`` array), and exposes
``DogfoodDataset`` that satisfies ``recall.eval.runner.Dataset``.

Per-query metadata (``id``, ``tier``, ``short``, ``star``) is preserved
on ``DogfoodDataset.queries`` for the CLI's per-query inline output.
The runner itself only sees ``EvalCase(nl_query, gold_commands)`` —
metadata is presentational, not eval logic.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from recall.eval.runner import EvalCase

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "eval" / "dogfood.toml"


@dataclass(frozen=True)
class DogfoodQueryMeta:
    """Presentational metadata for one dogfood query.

    The runner (``run_eval``) doesn't see this; only the CLI does, for
    the per-query inline output format. Kept on the dataset rather than
    on ``EvalCase`` to keep the eval-runner abstraction clean.
    """

    id: int
    tier: str  # "L" | "M" | "H"
    short: str
    star: bool


class DogfoodDataset:
    """Real-history eval dataset, loaded from ``eval/dogfood.toml``."""

    name = "dogfood"

    def __init__(self, toml_path: Path | None = None) -> None:
        path = toml_path if toml_path is not None else _DEFAULT_PATH
        with path.open("rb") as f:
            parsed = tomllib.load(f)

        raw_corpus = parsed.get("corpus")
        if not isinstance(raw_corpus, list) or not all(isinstance(c, str) for c in raw_corpus):
            raise ValueError(f"{path}: 'corpus' must be a non-empty list of strings")

        raw_queries = parsed.get("query")
        if not isinstance(raw_queries, list) or not raw_queries:
            raise ValueError(f"{path}: 'query' must be a non-empty array of tables")

        # Build cases + parallel metadata in TOML-declaration order.
        cases: list[EvalCase] = []
        meta: list[DogfoodQueryMeta] = []
        corpus_set = set(raw_corpus)

        for i, q in enumerate(raw_queries):
            if not isinstance(q, dict):
                raise ValueError(f"{path}: query[{i}] must be a table")
            nl = q.get("nl")
            if not isinstance(nl, str) or not nl.strip():
                raise ValueError(f"{path}: query[{i}] missing non-empty 'nl'")
            # Normalize gold | golds into a tuple. Field-name-as-arity-signal
            # is per the locked 2.6.5 schema.
            if "gold" in q and "golds" in q:
                raise ValueError(
                    f"{path}: query[{i}] specifies both 'gold' and 'golds'; "
                    "use 'gold' for single-reference, 'golds' for multi-reference"
                )
            if "gold" in q:
                if not isinstance(q["gold"], str):
                    raise ValueError(f"{path}: query[{i}].gold must be a string")
                gold_tuple: tuple[str, ...] = (q["gold"],)
            elif "golds" in q:
                gs = q["golds"]
                if not isinstance(gs, list) or not gs or not all(isinstance(g, str) for g in gs):
                    raise ValueError(
                        f"{path}: query[{i}].golds must be a non-empty list of strings"
                    )
                gold_tuple = tuple(gs)
            else:
                raise ValueError(f"{path}: query[{i}] missing 'gold' or 'golds'")

            # Each gold must appear in the corpus — otherwise recall is
            # unachievable by construction (silent eval bug if not enforced).
            for g in gold_tuple:
                if g not in corpus_set:
                    raise ValueError(
                        f"{path}: query[{i}] gold not present in corpus "
                        f"(first 60 chars: {g[:60]!r})"
                    )

            cases.append(EvalCase(nl_query=nl, gold_commands=gold_tuple))

            # Metadata fields (all required for the CLI's inline format).
            for field in ("id", "tier", "short"):
                if field not in q:
                    raise ValueError(f"{path}: query[{i}] missing '{field}'")
            qid = q["id"]
            tier = q["tier"]
            short = q["short"]
            if not isinstance(qid, int):
                raise ValueError(f"{path}: query[{i}].id must be an int")
            if tier not in {"L", "M", "H"}:
                raise ValueError(f"{path}: query[{i}].tier must be one of L|M|H")
            if not isinstance(short, str) or not short.strip():
                raise ValueError(f"{path}: query[{i}].short must be a non-empty string")
            star = bool(q.get("star", False))
            meta.append(DogfoodQueryMeta(id=qid, tier=tier, short=short, star=star))

        self._cases: tuple[EvalCase, ...] = tuple(cases)
        self._corpus: tuple[str, ...] = tuple(raw_corpus)
        self.queries: tuple[DogfoodQueryMeta, ...] = tuple(meta)

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    def corpus(self) -> Sequence[str]:
        return self._corpus


__all__ = ("DogfoodDataset", "DogfoodQueryMeta")
