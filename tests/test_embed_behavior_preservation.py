"""Behavior-preservation gate for Commit 2.7's embed.py rewrite.

The 2.7 rewrite changed embed.py internals (MPS warmup at construction,
explicit batch-size control, DEBUG-level encode logging) under a frozen
public API. This test pins the eval recall@5 to the calibrated 2.5/2.6
baseline within ±0.0001 — tighter than the ±0.01 cross-platform noise
band, because within-process determinism on the same model + corpus
should hold to far better than that.

If this fails, the rewrite has changed observable behavior and needs
investigation before merge. Run via ``pytest -m embed`` (the heavy lane;
loads the actual model).
"""

from __future__ import annotations

import pytest

from recall.eval.nl2bash import Nl2BashDataset
from recall.eval.runner import run_eval
from recall.retrieve.semantic import SemanticRanker

# Bit-identical baseline from 2.5 / 2.6 / 2.6.5 (eval/results.json).
# Pinned at full float64 precision; tolerance applied separately so a
# bit-identical 2.7 rewrite logs delta = 0.0 and a noisy one logs the
# magnitude for reviewers to inspect.
EXPECTED_RECALL_AT_5 = 0.44862530842439197
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
