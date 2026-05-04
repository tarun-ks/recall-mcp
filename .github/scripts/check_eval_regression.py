#!/usr/bin/env python3
"""Regression gate for the eval lane (CI-only).

Reads CI's just-computed eval result from a JSON history file (one or
more records produced by ``recall eval --output /tmp/ci_eval.json``)
and compares against the most recent semantic-ranker record for the
same dataset on ``origin/main:eval/results.json``.

Two gates fire on the **semantic** ranker only:

1. **Aggregate recall@5 gate (nl2bash only).** Fails if recall@5
   dropped by more than the noise band (±0.01) on the nl2bash dataset.
   Excluded for dogfood: with N=5 queries one query flipping = 0.2
   recall delta, swamping the noise band; the per-query detector below
   is the more meaningful gate for dogfood (CLAUDE.md "Phase 2 gating").

2. **Per-query PASS→FAIL detector (dogfood only).** Fails if any
   specific dogfood query transitioned from PASS@5 on ``origin/main``
   to FAIL on the PR — that's an existence-test regression, not a
   noise-band gate. Reads ``per_query_ranks`` (added 2.6.5) parallel to
   the dataset's case order; ``None`` is FAIL, integer ≤ 5 is PASS.

Lexical rankers are deterministic given the corpus + query, so any
change is a bug not a regression and is caught by per-ranker exact-
recall unit tests in ``tests/test_retrieve.py`` — those gates do not
fire here.

If no baseline exists (first eval commit, or first commit to introduce
a new dataset), exits 0 with an informational message.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

NOISE_BAND = 0.01


def _load_baseline_history() -> list[dict[str, object]] | None:
    """Return origin/main's eval/results.json as a list, or None if absent."""
    try:
        result = subprocess.run(
            ["git", "show", "origin/main:eval/results.json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    try:
        history = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(history, list):
        return None
    return [r for r in history if isinstance(r, dict)]


def _semantic_records(records: list[dict[str, object]], dataset: str) -> list[dict[str, object]]:
    return [r for r in records if r.get("dataset") == dataset and r.get("ranker") == "semantic"]


def _check_aggregate_nl2bash(
    pr_record: dict[str, object],
    baseline_history: list[dict[str, object]],
) -> int:
    """Aggregate recall@5 ±0.01 noise band on nl2bash semantic ranker."""
    pr_recall = pr_record["recall_at_5"]
    if not isinstance(pr_recall, (int, float)):
        raise SystemExit(f"PR record has non-numeric recall_at_5: {pr_recall!r}")
    print(f"PR semantic recall@5 (nl2bash): {pr_recall:.4f}")

    baseline_semantic = _semantic_records(baseline_history, "nl2bash")
    if not baseline_semantic:
        print("no nl2bash semantic record on origin/main — skipping aggregate gate.")
        return 0

    base = baseline_semantic[-1]
    base_recall = base["recall_at_5"]
    if not isinstance(base_recall, (int, float)):
        raise SystemExit(f"baseline record has non-numeric recall_at_5: {base_recall!r}")
    base_sha = str(base.get("commit_sha", "unknown"))[:7]
    print(f"baseline semantic recall@5 (nl2bash, commit {base_sha}): {base_recall:.4f}")

    delta = pr_recall - base_recall
    print(f"delta: {delta:+.4f} (noise band: ±{NOISE_BAND})")

    if delta < -NOISE_BAND:
        print(
            f"::error::recall@5 regression on nl2bash: {base_recall:.4f} → {pr_recall:.4f} "
            f"(delta {delta:+.4f}, beyond noise band ±{NOISE_BAND}). "
            "PRs touching retrieval logic must not regress nl2bash recall@5."
        )
        return 1

    print("OK: nl2bash recall@5 within noise or improved.")
    return 0


def _check_dogfood_per_query(
    pr_record: dict[str, object],
    baseline_history: list[dict[str, object]],
) -> int:
    """Per-query PASS→FAIL detector on dogfood semantic ranker.

    A query is PASS if its rank is an int in [1, 5]; FAIL if rank is
    None (no gold in top-5) or absent. PASS→FAIL on the same query
    index is an existence-test regression and fails the PR.
    """
    pr_ranks = pr_record.get("per_query_ranks")
    if not isinstance(pr_ranks, list):
        raise SystemExit(
            "PR dogfood semantic record missing 'per_query_ranks' — "
            "the eval lane must populate this field (added 2.6.5)."
        )
    print(f"PR dogfood per_query_ranks (semantic): {pr_ranks}")

    baseline_semantic = _semantic_records(baseline_history, "dogfood")
    if not baseline_semantic:
        print("no dogfood semantic record on origin/main — skipping per-query gate.")
        return 0

    base = baseline_semantic[-1]
    base_ranks = base.get("per_query_ranks")
    if not isinstance(base_ranks, list):
        # Old baseline predates per_query_ranks; can't compare. Track-only.
        print(
            "baseline dogfood semantic record predates per_query_ranks "
            "(commit before 2.6.5) — skipping per-query gate."
        )
        return 0
    base_sha = str(base.get("commit_sha", "unknown"))[:7]
    print(f"baseline per_query_ranks (commit {base_sha}, semantic): {base_ranks}")

    if len(pr_ranks) != len(base_ranks):
        # Adding/removing dogfood queries is intentional (the eval is
        # versioned with the corpus); skip rather than fail.
        print(
            f"per-query length mismatch (pr={len(pr_ranks)}, base={len(base_ranks)}) — "
            "dogfood query set changed; skipping per-query gate."
        )
        return 0

    def is_pass(rank: object) -> bool:
        return isinstance(rank, int) and 1 <= rank <= 5

    regressions: list[tuple[int, object, object]] = []
    for idx, (pr_r, base_r) in enumerate(zip(pr_ranks, base_ranks, strict=True)):
        if is_pass(base_r) and not is_pass(pr_r):
            regressions.append((idx, base_r, pr_r))

    if regressions:
        print("::error::dogfood per-query regression on semantic ranker:")
        for idx, base_r, pr_r in regressions:
            print(f"  query[{idx}]: PASS@{base_r} (origin/main) → {pr_r} (PR)")
        print(
            "PRs touching retrieval logic must not flip a dogfood query from "
            "PASS to FAIL on the semantic ranker. The 5-query dogfood set is "
            "an existence test — every flipped query is a real-use regression."
        )
        return 1

    print("OK: no dogfood per-query PASS→FAIL on semantic ranker.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: check_eval_regression.py <ci-eval-json-path>",
            file=sys.stderr,
        )
        return 2

    pr_history_path = Path(sys.argv[1])
    pr_history = json.loads(pr_history_path.read_text(encoding="utf-8"))
    if not isinstance(pr_history, list):
        raise SystemExit(f"{pr_history_path}: expected JSON array")

    pr_history_dicts = [r for r in pr_history if isinstance(r, dict)]
    pr_nl2bash = _semantic_records(pr_history_dicts, "nl2bash")
    pr_dogfood = _semantic_records(pr_history_dicts, "dogfood")
    if not pr_nl2bash and not pr_dogfood:
        raise SystemExit(
            f"{pr_history_path}: no semantic-ranker records on nl2bash or dogfood; "
            "the eval lane must run the semantic ranker on at least one dataset."
        )

    baseline_history = _load_baseline_history()
    if baseline_history is None:
        print("no eval/results.json on origin/main — skipping regression gate (first eval commit).")
        return 0

    rc = 0
    if pr_nl2bash:
        rc |= _check_aggregate_nl2bash(pr_nl2bash[-1], baseline_history)
        print()
    if pr_dogfood:
        rc |= _check_dogfood_per_query(pr_dogfood[-1], baseline_history)

    return rc


if __name__ == "__main__":
    sys.exit(main())
