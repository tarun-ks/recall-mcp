"""Tests for the eval harness.

The big test is marked ``@pytest.mark.embed`` — it loads the real
sentence-transformers model and runs an end-to-end synthetic eval. This
is the FIRST embed-marked test in the project; its existence closes the
deferred-issue about the embed-lane exit-5 tolerance in ci.yml (which is
removed in this same commit).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from recall.eval.runner import EvalCase, run_eval


@dataclass
class _SyntheticDataset:
    """Tiny in-memory dataset for testing eval pipeline shape + metrics."""

    name: str
    _cases: tuple[EvalCase, ...]
    _corpus: tuple[str, ...]

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    def corpus(self) -> Sequence[str]:
        return self._corpus


_SYNTHETIC_CASES = (
    EvalCase(nl_query="list files in the current directory", gold_commands=("ls",)),
    EvalCase(nl_query="show the system hostname", gold_commands=("hostname",)),
    EvalCase(
        nl_query="copy a file to another location",
        gold_commands=("cp source dest",),
    ),
    EvalCase(
        nl_query="search for text in files recursively",
        gold_commands=("grep -r 'pattern' .",),
    ),
    EvalCase(nl_query="show running processes", gold_commands=("ps aux",)),
)
_SYNTHETIC_CORPUS = (
    "ls",
    "ls -la",
    "ls /etc",
    "hostname",
    "uname -a",
    "uptime",
    "cp source dest",
    "mv source dest",
    "rm file",
    "grep -r 'pattern' .",
    "find . -name '*.py'",
    "awk '{print $1}'",
    "ps aux",
    "top",
    "kill -9 1234",
    "echo hello",
    "cat /etc/hostname",
    "df -h",
    "du -sh *",
    "history | tail",
)


@pytest.mark.embed
def test_synthetic_eval_runs_end_to_end() -> None:
    """End-to-end on a 5-query / 20-command synthetic dataset.

    Verifies the eval harness pipeline (model load → corpus embed → index
    build → query embed → search → metrics) completes cleanly with the
    real bge-small-en-v1.5 model.
    """
    ds = _SyntheticDataset(
        name="synthetic",
        _cases=_SYNTHETIC_CASES,
        _corpus=_SYNTHETIC_CORPUS,
    )
    result = run_eval(ds)

    assert result.n_queries == 5
    assert result.n_corpus == 20

    # Conservative thresholds — bge-small handles these clean cases easily but
    # we don't want flakiness from minor floating-point variation across runners.
    assert result.recall_at_5 >= 0.6, f"synthetic recall@5 too low: {result.recall_at_5}"
    assert result.mrr > 0.0
    assert result.recall_at_5 > result.random_baseline_recall_at_5

    # Schema sanity.
    assert result.model_name.endswith("bge-small-en-v1.5")
    for stage in ("model_load", "corpus_embed", "index_build", "query_embed", "search"):
        assert stage in result.runtime_breakdown, f"missing stage: {stage}"

    # Synthetic should finish well under 60s even with cold model load.
    assert result.runtime_seconds < 60.0, f"synthetic eval too slow: {result.runtime_seconds}s"


def test_eval_case_is_frozen() -> None:
    """``EvalCase`` is frozen — accidental mutation would break test
    determinism (cases are reused across multiple eval runs in CI)."""
    case = EvalCase(nl_query="x", gold_commands=("ls",))
    with pytest.raises((AttributeError, TypeError)):
        case.nl_query = "y"  # type: ignore[misc]
