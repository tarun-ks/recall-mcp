"""Tests for ``recall.scrub``.

Coverage strategy:
  - Per-pattern positive cases (each input MUST redact, secret MUST be absent
    from output, the expected ``<REDACTED:KIND>`` marker MUST appear).
  - Per-pattern negative cases (input MUST be returned unchanged).
  - Idempotency property over every other test's input plus already-redacted
    samples.
  - Edge cases: unicode, multi-line, multi-secret, scrubber version is set.
  - Corpus integrity: ``tests/fixtures/secrets_corpus.txt`` is scrubbed and
    asserted to contain none of the known secret patterns afterward. This is
    the systemic canary — if a new pattern leaks past per-test cases, it
    still has to also pass the corpus.

All secret values used here are synthetic. None are real credentials.

The marker ``pytest.mark.scrub`` makes ``pytest -k scrub`` (or ``-m scrub``)
run only this file — that's the pre-commit canary referenced in CLAUDE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from recall.scrub import SCRUBBER_VERSION, scrub

pytestmark = pytest.mark.scrub


# === Synthetic secret values shared across cases ===
#
# All values embed the canonical sentinel string `FAKEFAKE` so they cannot
# match real-world secret patterns and cannot trip GitHub's secret scanner
# (which blocked the very first push attempt — see the 1.2 commit message).
# Each value still satisfies the corresponding scrubber regex's length and
# character-class constraints; the tests verify both that the scrubber
# matches them AND that the synthetic value never leaks through.

_AKIA = "AKIAFAKEFAKEFAKEFAKE"  # AKIA + 16 alphanumeric uppercase
_AWS_SECRET_VAL = "FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"  # 40 chars
_GHP = "ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"  # ghp_ + 36 chars
_GHO = "gho_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
_GHS = "ghs_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
_GHU = "ghu_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
_GH_PAT = (
    "github_pat_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
    "FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFA"  # github_pat_ + exactly 82 chars
)
_JWT = "eyJFAKEFAKEFAKEFAKE.eyJFAKEFAKEFAKEFAKE.FAKEFAKEFAKEFAKE"
_OPENAI = "sk-proj-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
_ANTHROPIC = "sk-ant-api03-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
_SLACK = "xoxb-FAKEFAKE-FAKEFAKE-FAKEFAKEFAKE"
_GOOGLE = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAK"  # AIza + exactly 35 chars
_HIGH_ENTROPY = "FAKEFAKEABCdef123XYZqrs789TUVwxz456GHI"  # 38 chars, mixed → high entropy


# === POSITIVE: each input must redact; secret must be absent from output ===

_REDACT_CASES: list[tuple[str, str, str, str]] = [
    # (test_id, input, secret_substring, expected_kind)
    ("aws-access-key", f"aws s3 ls --access-key {_AKIA}", _AKIA, "AWS_ACCESS_KEY"),
    (
        "aws-secret-env",
        f"aws_secret_access_key={_AWS_SECRET_VAL} aws sts get-caller-identity",
        _AWS_SECRET_VAL,
        "AWS_SECRET",
    ),
    (
        "github-ghp-in-url",
        f"git clone https://{_GHP}@github.com/foo/bar",
        _GHP,
        "GITHUB_TOKEN",
    ),
    ("github-gho-export", f"export GITHUB_TOKEN={_GHO}", _GHO, "GITHUB_TOKEN"),
    ("github-ghs-bare", f"echo {_GHS}", _GHS, "GITHUB_TOKEN"),
    ("github-ghu-bare", f"echo {_GHU}", _GHU, "GITHUB_TOKEN"),
    (
        "github-pat-fine-grained",
        f"gh auth login --with-token {_GH_PAT}",
        _GH_PAT,
        "GITHUB_TOKEN",
    ),
    ("jwt-raw", f"the token is {_JWT}", _JWT, "JWT"),
    ("bearer-with-jwt", f'curl -H "Authorization: Bearer {_JWT}"', _JWT, "BEARER"),
    (
        "bearer-non-jwt",
        'curl -H "Authorization: Bearer FAKEFAKE_bearer_token_value"',
        "FAKEFAKE_bearer_token_value",
        "BEARER",
    ),
    ("pgpassword", "PGPASSWORD=FAKEFAKE_pgpass psql -h db -U admin", "FAKEFAKE_pgpass", "PASSWORD"),
    ("mysql-pwd-env", "MYSQL_PWD=FAKEFAKE_mysqlpwd mysql -u root", "FAKEFAKE_mysqlpwd", "PASSWORD"),
    (
        "mysql-dash-p-concat",
        "mysql -uroot -pFAKEFAKE_mypass -h localhost",
        "FAKEFAKE_mypass",
        "PASSWORD",
    ),
    (
        "mysqldump-dash-p-concat",
        "mysqldump -uadmin -pFAKEFAKE_dumppass prod_db",
        "FAKEFAKE_dumppass",
        "PASSWORD",
    ),
    (
        "docker-login-dash-p-space",
        "docker login -u alice -p FAKEFAKE_dockerpass ghcr.io",
        "FAKEFAKE_dockerpass",
        "PASSWORD",
    ),
    ("password-eq", "myapp --password=FAKEFAKE_pass --user=alice", "FAKEFAKE_pass", "PASSWORD"),
    (
        "password-space",
        "myapp --password FAKEFAKE_pass --user alice",
        "FAKEFAKE_pass",
        "PASSWORD",
    ),
    (
        "password-quoted",
        "myapp --password 'FAKEFAKE_super_pass' --user alice",
        "FAKEFAKE_super_pass",
        "PASSWORD",
    ),
    (
        "passwd-shorthand",
        "myapp --passwd=FAKEFAKE_changeme --port=8080",
        "FAKEFAKE_changeme",
        "PASSWORD",
    ),
    ("openai-key", f"export OPENAI_API_KEY={_OPENAI}", _OPENAI, "OPENAI_KEY"),
    (
        "anthropic-key",
        f"ANTHROPIC_API_KEY={_ANTHROPIC} python script.py",
        _ANTHROPIC,
        "ANTHROPIC_KEY",
    ),
    ("slack-bot-token", f"slack-cli post --token {_SLACK}", _SLACK, "SLACK_TOKEN"),
    (
        "google-api-key-url",
        f"curl 'https://www.googleapis.com/v1/foo?key={_GOOGLE}'",
        _GOOGLE,
        "GOOGLE_API_KEY",
    ),
    (
        "url-userinfo-https",
        "git clone https://alice:FAKEFAKE_pass@github.com/foo/bar",
        "alice:FAKEFAKE_pass",
        "URL_USERINFO",
    ),
    (
        "url-userinfo-postgres",
        "psql 'postgresql://admin:FAKEFAKE_pgpass@db.example.com:5432/prod'",
        "admin:FAKEFAKE_pgpass",
        "URL_USERINFO",
    ),
    (
        "ssh-private-key-inline",
        'echo "-----BEGIN OPENSSH PRIVATE KEY-----FAKEFAKE_AAAA-----END OPENSSH PRIVATE KEY-----"',
        "FAKEFAKE_AAAA",
        "SSH_PRIVATE_KEY",
    ),
    (
        "kubectl-token-jwt",
        f"kubectl --token={_JWT} --server=https://k8s.example.com",
        _JWT,
        "JWT",
    ),
    (
        "api-key-flag-high-entropy",
        f"myapp --api-key={_HIGH_ENTROPY} --region=us-east-1",
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "secret-flag",
        f"myapp --secret={_HIGH_ENTROPY}",
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "auth-flag",
        f"myapp --auth={_HIGH_ENTROPY}",
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "url-param-token",
        f"curl 'https://api.example.com/v1/data?token={_HIGH_ENTROPY}&format=json'",
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "url-param-api_key",
        f"curl 'https://api.example.com/v1/data?api_key={_HIGH_ENTROPY}'",
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "header-x-api-key",
        f'curl -H "X-API-Key: {_HIGH_ENTROPY}" https://api.example.com',
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "header-x-auth-token",
        f'curl -H "X-Auth-Token: {_HIGH_ENTROPY}" https://api.example.com',
        _HIGH_ENTROPY,
        "CREDENTIAL",
    ),
    (
        "multi-secret-line",
        f"git push https://{_GHP}@github.com && export OPENAI_API_KEY={_OPENAI}",
        _GHP,
        "GITHUB_TOKEN",
    ),
    (
        "multi-secret-line-second-still-redacted",
        f"git push https://{_GHP}@github.com && export OPENAI_API_KEY={_OPENAI}",
        _OPENAI,
        "OPENAI_KEY",
    ),
    (
        "comment-aws-key",
        f"# AWS_ACCESS_KEY_ID={_AKIA} rotate later",
        _AKIA,
        "AWS_ACCESS_KEY",
    ),
    (
        "multiline-secret-on-second-line",
        "line1\nPGPASSWORD=FAKEFAKE_multiline mysql\nline3",
        "FAKEFAKE_multiline",
        "PASSWORD",
    ),
]


@pytest.mark.parametrize(
    "text,secret,kind",
    [(c[1], c[2], c[3]) for c in _REDACT_CASES],
    ids=[c[0] for c in _REDACT_CASES],
)
def test_redacts(text: str, secret: str, kind: str) -> None:
    out = scrub(text)
    assert secret not in out, f"secret leaked through scrubber: {out!r}"
    assert _kind_marker(kind) in out, f"expected marker missing: {out!r}"


# === NEGATIVE: input must be returned unchanged ===

_NO_REDACT_CASES: list[tuple[str, str]] = [
    ("git-sha-checkout", "git checkout 1a2b3c4d5e6f7890abcdef1234567890abcdef12"),
    ("git-sha-show", "git show abcdef1234567890abcdef1234567890abcdef12"),
    ("plain-github-url-no-userinfo", "git clone https://github.com/foo/bar.git"),
    ("ssh-key-path-not-key", "ssh -i ~/.ssh/id_rsa user@host"),
    ("base64-not-jwt", 'echo "aGVsbG8gd29ybGQ=" | base64 -d'),
    ("xoxo-not-slack", "echo xoxo-not-a-token"),
    ("psql-port-flag-not-password", "psql -p 5432 -h localhost -U admin"),
    ("ssh-port-flag-not-password", "ssh -p 2222 user@host"),
    ("empty-string", ""),
    ("noise-no-secrets", "ls -la /tmp && cat /etc/hostname"),
    ("password-shell-var", "myapp --password=$PASSWORD --user=alice"),
    ("password-shell-var-braces", "myapp --password=${PG_PASSWORD} --user=alice"),
    ("password-shell-cmdsub", "docker login -u alice -p $(get-secret) ghcr.io"),
    ("token-flag-test-value-too-short", "myapp --token=test --user=alice"),
    ("api-key-flag-low-entropy", "myapp --api-key=12345678 --user=alice"),
    ("url-no-userinfo", "curl https://api.example.com/v1/data"),
    ("plain-prose", "the quick brown fox jumps over the lazy dog 123 456"),
]


@pytest.mark.parametrize(
    "text",
    [c[1] for c in _NO_REDACT_CASES],
    ids=[c[0] for c in _NO_REDACT_CASES],
)
def test_no_false_positive(text: str) -> None:
    assert scrub(text) == text, f"unexpected redaction: scrub({text!r}) == {scrub(text)!r}"


# === IDEMPOTENCY: scrub(scrub(s)) == scrub(s) over the union of all cases ===

_IDEMPOTENT_INPUTS: list[str] = (
    [c[1] for c in _REDACT_CASES]
    + [c[1] for c in _NO_REDACT_CASES]
    + [
        "<REDACTED:JWT> already scrubbed",
        "git push https://<REDACTED:GITHUB_TOKEN>@github.com",
        "kubectl --token=<REDACTED:JWT> --server=...",
        "<REDACTED:AWS_ACCESS_KEY> mixed with <REDACTED:OPENAI_KEY>",
    ]
)


@pytest.mark.parametrize(
    "text",
    _IDEMPOTENT_INPUTS,
    ids=[f"idem-{i}" for i in range(len(_IDEMPOTENT_INPUTS))],
)
def test_idempotent(text: str) -> None:
    once = scrub(text)
    twice = scrub(once)
    assert once == twice, f"non-idempotent: once={once!r}, twice={twice!r}"


# === Edge cases ===


def test_already_redacted_input_unchanged() -> None:
    text = "<REDACTED:AWS_ACCESS_KEY> some text <REDACTED:JWT> more"
    assert scrub(text) == text


def test_unicode_surrounding_preserved() -> None:
    text = f"héllo --token={_HIGH_ENTROPY} world ✨"
    out = scrub(text)
    assert "héllo" in out
    assert "world ✨" in out
    assert _HIGH_ENTROPY not in out
    assert "<REDACTED:CREDENTIAL>" in out


def test_scrubber_version_is_set() -> None:
    assert isinstance(SCRUBBER_VERSION, str)
    assert SCRUBBER_VERSION.strip()


# === Corpus integrity: systemic canary ===

_LEAK_DETECTORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
    re.compile(r"-----BEGIN[ A-Z]+PRIVATE KEY-----"),
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@<>]+:[^/\s@<>]+@"),
)

_CORPUS_PATH = Path(__file__).parent / "fixtures" / "secrets_corpus.txt"


def _corpus_lines() -> list[str]:
    return [
        line
        for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def test_corpus_has_at_least_30_lines() -> None:
    lines = _corpus_lines()
    assert len(lines) >= 30, (
        f"secrets_corpus.txt has only {len(lines)} non-comment lines; brief requires >= 30"
    )


def test_corpus_no_known_leaks() -> None:
    """The systemic canary: scrub the whole corpus and assert no detector matches.

    If a per-pattern test misses something, this still has to fail.
    """
    leaks: list[tuple[int, str, str]] = []
    for i, line in enumerate(_corpus_lines(), start=1):
        out = scrub(line)
        for det in _LEAK_DETECTORS:
            m = det.search(out)
            if m:
                leaks.append((i, det.pattern, m.group(0)))
    assert not leaks, "corpus leaks detected:\n" + "\n".join(
        f"  line {n}: detector={p!r} matched {m!r}" for n, p, m in leaks
    )


# === Helpers ===


def _kind_marker(kind: str) -> str:
    return f"<REDACTED:{kind}>"
