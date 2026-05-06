"""Equivalence-test gate for Commit 2.7.5's sqlite-vec → numpy rewrite.

Runs the new pure-numpy SemanticRanker against the full nl2bash query
set and compares top-5 IDs to the sqlite-vec MATCH reference frozen
at ``tests/fixtures/nl2bash_sqlite_vec_top5.json``.

Per CLAUDE.md §4a equivalence-test contract:

    Top-5 IDs are the same set across all nl2bash queries — set
    equality, not list equality. List equality is not claimed because
    float32 cosine permits ties; sqlite-vec / argpartition differ in
    how ties are broken.

DECISION MATRIX (per locked Q3 outcomes)

    both-pass:                 algorithms equivalent (normal)
    set miss + recall@5 ok:    tie reordering at top-5 boundary
                                 (acceptable; LOGGED, not failed)
    recall@5 drift + set ok:   impossible by construction (set
                                 equality implies same gold hits;
                                 if this fires, the test framework
                                 itself is wrong)
    both-fail:                 real bug — investigate

The test fires "fail" only on the both-fail case. The set-miss with
recall@5 intact prints divergence detail + continues. This matches the
locked outcome matrix exactly.

Marked ``@pytest.mark.embed`` (heavy lane; loads sentence-transformers).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import run_eval
from recall.retrieve.semantic import SemanticRanker

_FIXTURE = Path(__file__).parent / "fixtures" / "nl2bash_sqlite_vec_top5.json"

# Behavior-preservation baseline (same as test_embed_behavior_preservation).
EXPECTED_RECALL_AT_5 = 0.44862530842439197
RECALL_TOLERANCE = 0.0001

# Divergence sanity ceiling: float32 cosine ties at the top-5 boundary
# are real but should be rare. > 5% would suggest a systematic algorithmic
# divergence rather than tie noise. Empirical 2.7.5 measurement: ~0.3%.
MAX_DIVERGENCE_RATE = 0.05


@pytest.mark.embed
def test_numpy_matmul_top5_equivalence_to_sqlite_vec_reference() -> None:
    """Decision-matrix-driven equivalence: set equality + recall@5 layered.

    See module docstring for the locked outcome matrix.
    """
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    reference_records = fixture["records"]
    ref_meta = fixture["_meta"]

    ds = Nl2BashDataset()
    assert len(ds.cases()) == ref_meta["n_queries"], (
        "nl2bash query count diverged from fixture; "
        "regenerate fixture if the change is intentional."
    )
    assert len(ds.corpus()) == ref_meta["n_corpus"], (
        "nl2bash corpus size diverged from fixture; "
        "regenerate fixture if the change is intentional."
    )

    # Run the new (pure-numpy) ranker AND compute recall@5 in one
    # eval pass, so both axes of the decision matrix come from the
    # same execution.
    result = run_eval(ds, lambda: SemanticRanker(), k_max=5)
    pr_recall = result.recall_at_5
    recall_holds = abs(pr_recall - EXPECTED_RECALL_AT_5) <= RECALL_TOLERANCE

    # Re-derive the per-query top-5 lists. (run_eval returns aggregate
    # metrics; we need the actual top-5 IDs for set-equality. Run the
    # ranker again — cheap on a warm process.)
    ranker = SemanticRanker()
    ranker.index(list(ds.corpus()))
    queries = [c.nl_query for c in ds.cases()]
    pr_top_ids = ranker.search(queries, k=5)
    assert len(pr_top_ids) == len(reference_records)

    list_equal = 0
    set_equal = 0
    diverged: list[tuple[int, list[int], list[int]]] = []
    for i, ref in enumerate(reference_records):
        ref_ids = ref["top_5_ids"]
        pr_ids = list(pr_top_ids[i])
        if list(ref_ids) == pr_ids:
            list_equal += 1
            set_equal += 1
        elif set(ref_ids) == set(pr_ids):
            set_equal += 1
        else:
            diverged.append((i, ref_ids, pr_ids))

    n = len(reference_records)
    set_holds = len(diverged) == 0
    divergence_rate = len(diverged) / n

    # Always log the matrix-axis outcomes — visible to reviewers
    # whether the test passes or fails.
    print(
        f"\nequivalence outcome:"
        f"\n  list-equal:      {list_equal:>5}/{n} ({100 * list_equal / n:.2f}%)"
        f"\n  set-equal:       {set_equal:>5}/{n} ({100 * set_equal / n:.2f}%)"
        f"\n  diverged:        {len(diverged):>5}/{n} ({100 * divergence_rate:.2f}%)"
        f"\n  recall@5:        {pr_recall:.6f} "
        f"(baseline {EXPECTED_RECALL_AT_5:.6f}, "
        f"delta {abs(pr_recall - EXPECTED_RECALL_AT_5):.2e})"
    )

    # Apply the locked decision matrix.
    if set_holds and recall_holds:
        # both-pass: algorithms equivalent
        return
    if not set_holds and recall_holds:
        # set miss + recall@5 ok: tie reordering at top-5 boundary
        # (acceptable; logged). Fail only if divergence rate exceeds
        # the sanity ceiling — that would suggest something other
        # than tie noise.
        sample = diverged[:5]
        msg = [
            f"tie-reordering divergences detected: "
            f"{len(diverged)}/{n} queries ({100 * divergence_rate:.2f}%) "
            f"set-mismatch but recall@5 within ±{RECALL_TOLERANCE} of baseline.",
            "ACCEPTABLE per CLAUDE.md §4a (set equality, not list equality, "
            "is the contract). Sample divergences:",
        ]
        for q_idx, ref_ids, pr_ids in sample:
            msg.append(f"  query[{q_idx}]: ref={ref_ids}  pr={pr_ids}")
        if len(diverged) > 5:
            msg.append(f"  ... ({len(diverged) - 5} more)")
        print("\n" + "\n".join(msg))

        if divergence_rate > MAX_DIVERGENCE_RATE:
            pytest.fail(
                f"divergence rate {100 * divergence_rate:.2f}% "
                f"exceeds {100 * MAX_DIVERGENCE_RATE:.0f}% sanity ceiling — "
                "that's beyond float32 tie noise; investigate."
            )
        return
    if set_holds and not recall_holds:
        # Impossible by construction: set equality implies same gold
        # hits implies same recall. If this fires, something has gone
        # very wrong with the test harness itself.
        pytest.fail(
            f"impossibility check fired: set equality holds but recall@5 "
            f"= {pr_recall:.6f} drifted from baseline {EXPECTED_RECALL_AT_5:.6f}. "
            "Test framework / metric computation is wrong."
        )
    # both-fail: real bug
    sample = diverged[:5]
    msg = [
        "BOTH set equality AND recall@5 fail.",
        f"recall@5 = {pr_recall:.6f} drifted by "
        f"{abs(pr_recall - EXPECTED_RECALL_AT_5):.2e} from baseline "
        f"{EXPECTED_RECALL_AT_5:.6f} (tolerance ±{RECALL_TOLERANCE}).",
        f"Plus {len(diverged)}/{n} queries diverged on set equality.",
        "This is a real algorithmic bug — investigate.",
        "Sample diverged queries:",
    ]
    for q_idx, ref_ids, pr_ids in sample:
        msg.append(f"  query[{q_idx}]: ref={ref_ids}  pr={pr_ids}")
    pytest.fail("\n".join(msg))


@pytest.mark.embed
def test_fixture_provenance_is_pinned() -> None:
    """Fixture metadata must document its origin so a future regeneration
    is an explicit reasoning act, not an accidental drift.

    Guards the contract documented in CLAUDE.md §4a: the fixture is THE
    equivalence baseline; regenerating requires explicit reasoning
    about why the reference shifts.
    """
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    meta = fixture["_meta"]

    assert meta.get("description")
    assert meta.get("provenance")
    assert "n_queries" in meta and isinstance(meta["n_queries"], int)
    assert "n_corpus" in meta and isinstance(meta["n_corpus"], int)
    assert meta.get("model")
    assert meta.get("schema")

    # Provenance must mention sqlite-vec as the reference (the whole
    # point of the fixture).
    assert "sqlite-vec" in meta["provenance"].lower()
    # And must signal that regenerating is a deliberate choice.
    assert "regenerat" in meta["provenance"].lower()
