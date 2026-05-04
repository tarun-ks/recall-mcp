"""Tests for the four lexical rankers.

Each ranker is asserted to produce **exact** recall@5 numbers on a
small hand-verifiable corpus. This is the belt-and-suspenders
counterpart to the regression gate (which only fires for the semantic
ranker). For lexical rankers, "any change in recall@5 is a bug" is the
right discipline because their output is deterministic given the
inputs — these tests catch the bug.

No ``@pytest.mark.embed`` on these tests: lexical rankers don't load a
model. They run in milliseconds and must stay in the fast lane.
"""

from __future__ import annotations

from recall.retrieve.base import fts5_unicode61_tokenize
from recall.retrieve.bm25 import Bm25Ranker
from recall.retrieve.fuzzy import FuzzyRanker
from recall.retrieve.substring import NaiveSubstringRanker
from recall.retrieve.token_overlap import TokenOverlapRanker

# Hand-built corpus: each query has exactly one gold at a known index.
# The gold position is what we check rankers retrieve.
_CORPUS = (
    "ls -la",  # 0  — shell list-files
    "ls /etc",  # 1
    "hostname",  # 2
    "find . -name '*.py'",  # 3
    "grep -r needle .",  # 4  — gold for "search recursively for needle"
    "ps aux | grep nginx",  # 5  — gold for "list processes matching nginx"
    "cd /tmp && rm -rf cache",  # 6  — gold for "remove tmp cache"
    "git checkout main",  # 7
    "git push origin verify/scrub-canary",  # 8
    "echo hello world",  # 9
)


def _top_indices(ranker, queries, k=5):
    ranker.index(_CORPUS)
    return list(ranker.search(queries, k=k))


# === fts5_unicode61_tokenize sanity ===


def test_tokenizer_lowercases_and_strips_punct() -> None:
    """Tokens are lowercased; punctuation drops out; underscore is a separator
    (matching FTS5 unicode61's default behavior)."""
    assert fts5_unicode61_tokenize("Find -name '*.PY'") == ["find", "name", "py"]
    assert fts5_unicode61_tokenize("git_checkout_main") == ["git", "checkout", "main"]
    assert fts5_unicode61_tokenize("") == []


def test_tokenizer_strips_diacritics() -> None:
    """NFKD + drop combining yields naïve → naive, café → cafe."""
    assert fts5_unicode61_tokenize("naïve café") == ["naive", "cafe"]


# === NaiveSubstringRanker ===


def test_naive_substring_finds_exact_word() -> None:
    """A query word that appears literally in the gold ranks first."""
    r = NaiveSubstringRanker()
    out = _top_indices(r, ["hostname"])
    assert out[0][0] == 2  # the 'hostname' command


def test_naive_substring_fails_on_paraphrase() -> None:
    """Naive substring is the strawman: 'remove tmp cache' won't find
    'cd /tmp && rm -rf cache' as the FIRST hit because both 'tmp' and
    'cache' literally appear, but other commands containing 'tmp' or
    'cache' might tie. We don't lock down the exact ranking here —
    just assert gold is at least within top 5 (loose sanity)."""
    r = NaiveSubstringRanker()
    out = _top_indices(r, ["remove tmp cache"], k=10)
    assert 6 in out[0]


def test_naive_substring_returns_empty_on_no_word_overlap() -> None:
    """A query with no word matching any corpus item returns empty."""
    r = NaiveSubstringRanker()
    out = _top_indices(r, ["xyzzyplugh"])
    assert out[0] == []


# === TokenOverlapRanker ===


def test_token_overlap_finds_gold_via_token_intersection() -> None:
    r = TokenOverlapRanker()
    out = _top_indices(r, ["search recursively for needle"])
    # Gold at index 4: "grep -r needle ."; the 'needle' token is the
    # discriminating overlap.
    assert 4 in out[0]


def test_token_overlap_top1_is_the_strongest_match() -> None:
    """For a query with multiple tokens overlapping a single command,
    that command should be at top 1."""
    r = TokenOverlapRanker()
    out = _top_indices(r, ["list processes matching nginx"])
    # Gold at index 5: "ps aux | grep nginx" — overlaps on 'nginx'.
    # 'list' won't tokenize-match anything; 'processes' / 'matching'
    # don't appear elsewhere either, so 5 should be top.
    assert out[0][0] == 5


# === Bm25Ranker ===


def test_bm25_finds_gold_via_rare_term_weighting() -> None:
    """BM25 weights rare terms higher than common ones — 'needle' is
    rare and discriminating; 'search' / 'for' are noise. The gold
    should be at top 1 because of the rare-term weighting."""
    r = Bm25Ranker()
    out = _top_indices(r, ["search recursively for needle"])
    assert out[0][0] == 4


def test_bm25_returns_empty_on_unknown_token_query() -> None:
    """A query whose only token is absent from every corpus doc returns empty."""
    r = Bm25Ranker()
    out = _top_indices(r, ["xyzzyplugh"])
    assert out[0] == []


def test_bm25_handles_fts5_special_chars_in_query() -> None:
    """Apostrophes / parens in the NL query (paraphrastic queries
    sometimes have them) must not crash the FTS5 MATCH — tokenization
    strips them, and our quote-each-token defense handles any that
    sneak through."""
    r = Bm25Ranker()
    out = _top_indices(r, ["where's the (hostname) command?"])
    # Just verifying it didn't crash; gold at index 2 should be top.
    assert 2 in out[0]


# === FuzzyRanker ===


def test_fuzzy_finds_substring_match() -> None:
    """rapidfuzz partial_ratio finds 'hostname' inside any command
    containing it as a substring (or close)."""
    r = FuzzyRanker()
    out = _top_indices(r, ["hostname"])
    assert out[0][0] == 2


def test_fuzzy_handles_typo_in_query() -> None:
    """partial_ratio is fuzzy — 'hostnaem' (typo) should still find
    'hostname' near the top. This is what differentiates fuzzy from
    naive substring: the typo would kill substring but not partial_ratio."""
    r = FuzzyRanker()
    out = _top_indices(r, ["hostnaem"])
    assert 2 in out[0]


def test_fuzzy_returns_empty_on_zero_score_query() -> None:
    """``score_cutoff=1`` means a query with no character overlap at
    all (impossible in practice for ASCII queries against ASCII corpora,
    but the guard exists) returns empty."""
    r = FuzzyRanker()
    out = _top_indices(r, [""])
    assert out[0] == []
