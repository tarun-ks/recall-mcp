#!/usr/bin/env python3
"""Regression gate for the eval lane (CI-only).

Reads CI's just-computed eval result from a JSON history file (single
record produced by ``recall eval --output /tmp/ci_eval.json``), compares
against the most recent record for the same dataset on
``origin/main:eval/results.json``. Fails if recall@5 dropped by more
than the noise band (±0.01).

If no baseline exists (first eval commit, or first commit to introduce
a new dataset), exits 0 with an informational message.

Per CLAUDE.md "Phase 2 gating rules": this gate is what makes "regression
blocks merge" enforceable in CI rather than honor-system. The noise band
(±0.01) is what makes the embed.py 2.5→2.7 "behavior-preserving rewrite"
contract testable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

NOISE_BAND = 0.01


def _load_pr_record(path: Path) -> dict[str, object]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise SystemExit(f"{path}: expected non-empty JSON array; got {type(history).__name__}")
    last = history[-1]
    if not isinstance(last, dict):
        raise SystemExit(f"{path}: last record is not a dict")
    return last


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


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: check_eval_regression.py <ci-eval-json-path>",
            file=sys.stderr,
        )
        return 2

    pr_record = _load_pr_record(Path(sys.argv[1]))
    dataset = pr_record["dataset"]
    pr_recall = pr_record["recall_at_5"]
    if not isinstance(pr_recall, (int, float)):
        raise SystemExit(f"PR record has non-numeric recall_at_5: {pr_recall!r}")

    print(f"PR recall@5 ({dataset}): {pr_recall:.4f}")

    baseline_history = _load_baseline_history()
    if baseline_history is None:
        print("no eval/results.json on origin/main — skipping regression gate (first eval commit).")
        return 0

    baseline_records = [r for r in baseline_history if r.get("dataset") == dataset]
    if not baseline_records:
        print(f"no record for dataset {dataset!r} on origin/main — skipping regression gate.")
        return 0

    base = baseline_records[-1]
    base_recall = base["recall_at_5"]
    if not isinstance(base_recall, (int, float)):
        raise SystemExit(f"baseline record has non-numeric recall_at_5: {base_recall!r}")
    base_sha = str(base.get("commit_sha", "unknown"))[:7]
    print(f"baseline recall@5 ({dataset}, commit {base_sha}): {base_recall:.4f}")

    delta = pr_recall - base_recall
    print(f"delta: {delta:+.4f} (noise band: ±{NOISE_BAND})")

    if delta < -NOISE_BAND:
        print(
            f"::error::recall@5 regression: {base_recall:.4f} → {pr_recall:.4f} "
            f"(delta {delta:+.4f}, beyond noise band ±{NOISE_BAND}). "
            "PRs touching retrieval logic must not regress recall@5."
        )
        return 1

    print("OK: recall@5 within noise or improved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
