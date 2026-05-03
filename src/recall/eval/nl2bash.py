"""nl2bash eval dataset (Lin et al., 2018).

Source repo: https://github.com/TellinaTool/nl2bash
Files used: ``data/bash/all.nl`` and ``data/bash/all.cm`` — paired by line index.

Caches at ``~/.recall/datasets/nl2bash/``. First run downloads ~10 MB total;
subsequent runs read from cache.

Multi-reference semantics (per CLAUDE.md "Phase 2 gating rules"):
  We group by NL string. If the same NL appears multiple times in the dataset
  paired with different commands, all those commands count as gold for that
  query. Recall@k = "any gold reference in top k."
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from recall.eval.runner import EvalCase

_LOG = logging.getLogger("recall.eval.nl2bash")
_CACHE_DIR = Path.home() / ".recall" / "datasets" / "nl2bash"
_BASE_URL = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash"
_FILES = ("all.nl", "all.cm")


class Nl2BashDataset:
    """nl2bash dataset, downloaded on first use, cached locally."""

    name = "nl2bash"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache = cache_dir if cache_dir is not None else _CACHE_DIR
        self._download_if_missing()
        self._cases, self._corpus = self._load()

    def _download_if_missing(self) -> None:
        self._cache.mkdir(parents=True, exist_ok=True)
        for fname in _FILES:
            target = self._cache / fname
            if target.exists():
                continue
            url = f"{_BASE_URL}/{fname}"
            _LOG.info("recall.eval.nl2bash: downloading %s", url)
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                with target.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)

    def _load(self) -> tuple[tuple[EvalCase, ...], tuple[str, ...]]:
        nl_lines = (self._cache / "all.nl").read_text(encoding="utf-8").splitlines()
        cm_lines = (self._cache / "all.cm").read_text(encoding="utf-8").splitlines()
        if len(nl_lines) != len(cm_lines):
            raise RuntimeError(
                f"nl2bash data files misaligned: {len(nl_lines)} NL lines vs "
                f"{len(cm_lines)} command lines (URL changed upstream?)"
            )
        # Multi-reference: group by NL; collect all commands paired with each NL.
        nl_to_golds: dict[str, set[str]] = {}
        for nl_raw, cm_raw in zip(nl_lines, cm_lines, strict=True):
            nl = nl_raw.strip()
            cm = cm_raw.strip()
            if not nl or not cm:
                continue
            nl_to_golds.setdefault(nl, set()).add(cm)
        cases = tuple(
            EvalCase(nl_query=nl, gold_commands=tuple(sorted(golds)))
            for nl, golds in sorted(nl_to_golds.items())
        )
        corpus = tuple(sorted({cm.strip() for cm in cm_lines if cm.strip()}))
        return cases, corpus

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    def corpus(self) -> Sequence[str]:
        return self._corpus


__all__ = ("Nl2BashDataset",)
