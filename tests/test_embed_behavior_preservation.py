"""Behavior-preservation gate for the SemanticRanker eval pipeline.

Pins the nl2bash semantic recall@5 to the deterministic algorithm's
canonical output within ±0.0001. Tighter than the ±0.01 cross-platform
noise band because the algorithm is now deterministic across runners
(2.7.5-hotfix); the only remaining variance source is intentional code
change, which is exactly what this gate catches.

BASELINE HISTORY (anchor shifts ARE landmark events; record them here):

    2.5 / 2.6 / 2.6.5 / 2.7 / 2.7.5 (sqlite-vec reference / argpartition):
        0.44862530842439197 — value sqlite-vec produced; matched by
        argpartition + M-series accidentally. Cross-runner CI surfaced
        that this value depended on argpartition's implementation-
        specific tie-breaking happening to coincide with sqlite-vec's
        internal ordering on ~39 tie-affected queries.

    2.7.5-hotfix (deterministic numpy algorithm, low-index tie-break):
        0.44836094465985193 — the canonical output of the composite-
        key argsort algorithm. Identical across M-series, Linux CI,
        and any future runner. The ~3e-04 shift from the sqlite-vec
        anchor is de-aliasing, not regression: the old value was an
        artifact of platform-specific tie-breaking that happened to
        match sqlite-vec; the new value is what the deterministic
        algorithm produces canonically.

Run via ``pytest -m embed`` (the heavy lane; loads the actual model).
"""

from __future__ import annotations

import pytest

from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import run_eval
from recall.retrieve.semantic import SemanticRanker

# Deterministic-algorithm baseline (re-anchored 2.7.5-hotfix). Pinned at
# full float64 precision; tolerance applied separately so a stable
# implementation logs delta = 0.0 and a regressed one logs the magnitude
# for reviewers to inspect.
EXPECTED_RECALL_AT_5 = 0.44836094465985193
TOLERANCE = 0.0001


@pytest.mark.embed
def test_nl2bash_semantic_recall_at_5_pinned() -> None:
    """nl2bash semantic recall@5 must match 2.5/2.6 baseline within tolerance."""
    ds = Nl2BashDataset()
    result = run_eval(ds, lambda: SemanticRanker(), k_max=5)

    assert result.dataset == "nl2bash"
    assert result.ranker == "semantic"
    assert result.recall_at_5 == pytest.approx(EXPECTED_RECALL_AT_5, abs=TOLERANCE)

    # Log the actual delta so reviewers see the magnitude even when the
    # gate passes. Bit-identical → 0.0; float-noise from batch-grouping
    # changes → some small magnitude that we can decide is acceptable.
    delta = abs(result.recall_at_5 - EXPECTED_RECALL_AT_5)
    print(f"behavior-preservation delta: {delta:.2e}")
