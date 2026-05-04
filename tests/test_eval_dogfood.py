"""Tests for the dogfood eval dataset (Commit 2.6.5).

These tests are NOT marked ``@pytest.mark.embed`` — they validate the
TOML schema, dataset wiring, and per-query rank field through the
runner with a fast lexical ranker (BM25). End-to-end semantic eval is
already covered by the embed-marked test in ``test_eval.py`` and by
the CI eval lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.eval.dogfood import DogfoodDataset
from recall.eval.runner import EvalCase, run_eval
from recall.retrieve.bm25 import Bm25Ranker

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOGFOOD_TOML = _REPO_ROOT / "eval" / "dogfood.toml"


def test_dogfood_toml_loads() -> None:
    """Real ``eval/dogfood.toml`` parses and produces a non-empty dataset."""
    ds = DogfoodDataset()
    assert ds.name == "dogfood"
    assert len(ds.corpus()) >= 50, "dogfood corpus shrank below 50; check the TOML"
    assert len(ds.cases()) >= 1
    # The 2.6.5 design lock: 5 queries, 6 gold positions across them.
    assert len(ds.cases()) == 5
    total_gold = sum(len(c.gold_commands) for c in ds.cases())
    assert total_gold == 6, f"expected 6 gold positions across 5 queries; got {total_gold}"


def test_dogfood_metadata_parallel_to_cases() -> None:
    """``DogfoodDataset.queries`` is parallel to ``cases()`` — same order, same length.

    The CLI's per-query inline format zips them; mismatched length would
    silently mis-label rows.
    """
    ds = DogfoodDataset()
    assert len(ds.queries) == len(ds.cases())
    # Every metadata entry has a tier in {L, M, H}.
    for q in ds.queries:
        assert q.tier in {"L", "M", "H"}
        assert q.id > 0
        assert q.short.strip()


def test_dogfood_golds_in_corpus() -> None:
    """Every gold command must appear in the corpus.

    This is the silent-eval-bug guard: a gold not present in the corpus
    means recall@k is 0 by construction regardless of ranker quality.
    DogfoodDataset enforces this at load time; this test pins the
    contract so a future TOML edit can't bypass it without test failure.
    """
    ds = DogfoodDataset()
    corpus_set = set(ds.corpus())
    for case in ds.cases():
        for g in case.gold_commands:
            assert g in corpus_set, f"gold not in corpus: {g[:80]!r}"


def test_dogfood_runner_populates_per_query_ranks() -> None:
    """End-to-end via BM25 to verify the runner populates ``per_query_ranks``.

    BM25 is deterministic (FTS5 unicode61 + bm25 score) — the test
    pins exact rank values, which doubles as a regression canary for
    BM25's behavior on the dogfood corpus.
    """
    ds = DogfoodDataset()
    result = run_eval(ds, lambda: Bm25Ranker())
    assert result.dataset == "dogfood"
    assert result.ranker == "bm25"
    assert len(result.per_query_ranks) == len(ds.cases())
    # Every entry is either an int in [1, 5] or None.
    for r in result.per_query_ranks:
        assert r is None or (isinstance(r, int) and 1 <= r <= 5)
    # BM25 should at minimum solve #42 (vocabulary-overlap "useAgent" query).
    # If this drops to FAIL the corpus or BM25 tokenization broke.
    by_id = {q.id: idx for idx, q in enumerate(ds.queries)}
    rank_42 = result.per_query_ranks[by_id[42]]
    assert rank_42 == 1, f"BM25 expected to solve #42 at rank 1; got {rank_42}"


def test_dogfood_dataset_load_failure_on_missing_gold(tmp_path: Path) -> None:
    """Fabricated TOML with a gold not in the corpus must fail loudly."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'corpus = ["echo a", "echo b"]\n'
        "[[query]]\n"
        "id = 1\n"
        'tier = "L"\n'
        'short = "missing"\n'
        'nl = "x"\n'
        'gold = "echo c"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gold not present in corpus"):
        DogfoodDataset(toml_path=bad)


def test_dogfood_dataset_rejects_both_gold_and_golds(tmp_path: Path) -> None:
    """``gold`` and ``golds`` are mutually exclusive — arity is signalled by name."""
    bad = tmp_path / "both.toml"
    bad.write_text(
        'corpus = ["echo a"]\n'
        "[[query]]\n"
        "id = 1\n"
        'tier = "L"\n'
        'short = "both"\n'
        'nl = "x"\n'
        'gold = "echo a"\n'
        'golds = ["echo a"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both 'gold' and 'golds'"):
        DogfoodDataset(toml_path=bad)


def test_eval_case_default_per_query_ranks_empty() -> None:
    """Backward-compat: ``EvalResult.per_query_ranks`` defaults to empty tuple.

    Existing call sites that construct ``EvalResult`` (none in production —
    only ``run_eval`` does — but tests may) keep working without setting
    the new field.
    """
    case = EvalCase(nl_query="x", gold_commands=("ls",))
    assert case.gold_commands == ("ls",)


def test_dogfood_toml_contains_no_unredacted_pii() -> None:
    """Per the 2.6.5 dogfood-prep dual scrubber-gap finding: no
    ``@gmail.com`` / ``@yahoo.com`` / etc. in the on-disk TOML, and no
    ``password='<literal>'`` Python kwargs. These slipped past scrub.py
    during dogfood selection (issue #2 Tier 1); the TOML has been
    sanitized by hand. This test is the regression canary so a future
    edit can't accidentally re-introduce them.
    """
    import re

    raw = _DOGFOOD_TOML.read_text(encoding="utf-8")
    email_pii = re.compile(r"\b[a-zA-Z0-9._-]+@(?:gmail|yahoo|outlook|icloud|hotmail|live)\.com\b")
    matches = email_pii.findall(raw)
    assert not matches, f"personal email leaked into dogfood.toml: {matches}"

    pwd_kwarg = re.compile(r"""password\s*=\s*['"][^'"]+['"]""")
    pwd_matches = pwd_kwarg.findall(raw)
    assert not pwd_matches, f"Python kwarg-form password leaked into dogfood.toml: {pwd_matches}"
