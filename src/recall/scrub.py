"""Secret scrubber. The trust foundation of Recall.

Invoked at index time (so the on-disk SQLite never holds plaintext secrets)
and again at query response (defense in depth). Replaces detected secrets
with ``<REDACTED:KIND>`` markers and is idempotent: ``scrub(scrub(s)) == scrub(s)``.

Pattern coverage and entropy rules: see CLAUDE.md, "Critical correctness
requirements" section 1.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import log2

# Bumped whenever rules change in a way that changes outputs for plausible
# inputs. Stored in the meta table so the DB knows what scrubber rev produced
# its scrubbed text.
SCRUBBER_VERSION = "1"

# Flag-scoped credential entropy threshold (bits/char). Calibrated so:
#   - high-entropy token-like values (>= ~4 bits/char) trigger;
#   - low-entropy human-chosen passwords ("password123" ~ 2.5 bits/char)
#     don't trigger via entropy — they're caught by the always-redact rule
#     for password-class flags;
#   - 40-char hex git SHAs (4.0 bits/char) WOULD trigger if entropy ran over
#     free text, but this scrubber only runs entropy detection over values
#     bound to a sensitive flag, so SHAs in `git checkout <sha>` are safe.
_ENTROPY_THRESHOLD = 3.5
_ENTROPY_MIN_LENGTH = 8

_SHELL_VAR_RE = re.compile(r"^\$(?:\{[^}]*\}|\([^)]*\)|[A-Za-z_]\w*)$")
_REDACTED_RE = re.compile(r"^<REDACTED:[A-Z_]+>$")


def _kind(name: str) -> str:
    return f"<REDACTED:{name}>"


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * log2(c / n) for c in Counter(s).values())


def _is_shell_var(value: str) -> bool:
    return bool(_SHELL_VAR_RE.match(value))


def _is_already_redacted(value: str) -> bool:
    return bool(_REDACTED_RE.match(value))


def _strip_quotes(s: str) -> tuple[str, str]:
    """Return (inner, quote_char). quote_char is empty if s wasn't quoted."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1], s[0]
    return s, ""


def _fixed(kind: str) -> Callable[[re.Match[str]], str]:
    target = _kind(kind)

    def _repl(_m: re.Match[str]) -> str:
        return target

    return _repl


def _password_value_replacement(kind: str) -> Callable[[re.Match[str]], str]:
    """Always-redact replacement for password-class flag values."""
    target = _kind(kind)

    def _repl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        value = m.group(2)
        stripped, quote = _strip_quotes(value)
        if not stripped or _is_shell_var(stripped) or _is_already_redacted(stripped):
            return m.group(0)
        return f"{prefix}{quote}{target}{quote}"

    return _repl


def _credential_value_replacement(kind: str) -> Callable[[re.Match[str]], str]:
    """Entropy-scoped replacement for token/key/secret-class flag values."""
    target = _kind(kind)

    def _repl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        value = m.group(2)
        stripped, quote = _strip_quotes(value)
        if not stripped or _is_shell_var(stripped) or _is_already_redacted(stripped):
            return m.group(0)
        if len(stripped) < _ENTROPY_MIN_LENGTH:
            return m.group(0)
        if _shannon_entropy(stripped) < _ENTROPY_THRESHOLD:
            return m.group(0)
        return f"{prefix}{quote}{target}{quote}"

    return _repl


def _aws_secret_replacement(m: re.Match[str]) -> str:
    return f"{m.group(1)}{m.group(2)}{_kind('AWS_SECRET')}{m.group(2)}"


def _url_userinfo_replacement(m: re.Match[str]) -> str:
    return f"{m.group(1)}{_kind('URL_USERINFO')}{m.group(4)}"


@dataclass(frozen=True, slots=True)
class _Pattern:
    name: str
    regex: re.Pattern[str]
    replacement: Callable[[re.Match[str]], str]


# Pattern application order. Most specific first; later patterns must
# tolerate finding <REDACTED:KIND> markers from earlier passes (idempotency).
_PATTERNS: tuple[_Pattern, ...] = (
    # Multi-line SSH private key blocks. First so we don't sub-match inside.
    _Pattern(
        name="SSH_PRIVATE_KEY",
        regex=re.compile(
            r"-----BEGIN[ A-Z]+PRIVATE KEY-----.*?-----END[ A-Z]+PRIVATE KEY-----",
            re.DOTALL,
        ),
        replacement=_fixed("SSH_PRIVATE_KEY"),
    ),
    # Authorization: Bearer <token>. More specific than raw JWT — must come
    # before the JWT pattern so we tag the bearer wrapper, not the inner JWT.
    _Pattern(
        name="BEARER",
        regex=re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)"),
        replacement=_password_value_replacement("BEARER"),
    ),
    # Raw JWT (eyJ...eyJ...sig).
    _Pattern(
        name="JWT",
        regex=re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
        replacement=_fixed("JWT"),
    ),
    # GitHub tokens (classic prefixes + fine-grained PAT).
    _Pattern(
        name="GITHUB_TOKEN",
        regex=re.compile(
            r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{82}\b"
        ),
        replacement=_fixed("GITHUB_TOKEN"),
    ),
    # AWS Access Key ID.
    _Pattern(
        name="AWS_ACCESS_KEY",
        regex=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        replacement=_fixed("AWS_ACCESS_KEY"),
    ),
    # AWS Secret Access Key in key=val form (40 chars base64-ish).
    _Pattern(
        name="AWS_SECRET",
        regex=re.compile(r"(?i)(aws_secret_access_key\s*=\s*)([\"']?)([A-Za-z0-9/+=]{40})\2"),
        replacement=_aws_secret_replacement,
    ),
    # Anthropic key. Before OpenAI to disambiguate the sk- prefix.
    _Pattern(
        name="ANTHROPIC_KEY",
        regex=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
        replacement=_fixed("ANTHROPIC_KEY"),
    ),
    # OpenAI key. Negative lookahead avoids re-matching sk-ant-... already
    # caught above (and any future sk-<vendor>- subprefix we add ahead of
    # this pattern).
    _Pattern(
        name="OPENAI_KEY",
        regex=re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b"),
        replacement=_fixed("OPENAI_KEY"),
    ),
    # Slack tokens.
    _Pattern(
        name="SLACK_TOKEN",
        regex=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        replacement=_fixed("SLACK_TOKEN"),
    ),
    # Google API keys (39 chars total: AIza + exactly 35).
    _Pattern(
        name="GOOGLE_API_KEY",
        regex=re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
        replacement=_fixed("GOOGLE_API_KEY"),
    ),
    # URLs with user:password@ userinfo. Excludes < and > so the regex won't
    # cross <REDACTED:KIND> markers from earlier passes (idempotency).
    _Pattern(
        name="URL_USERINFO",
        regex=re.compile(r"(\b[a-z][a-z0-9+.-]*://)([^/\s:@<>]+):([^/\s@<>]+)(@)"),
        replacement=_url_userinfo_replacement,
    ),
    # PGPASSWORD env-var (case-sensitive — Unix env-var convention).
    _Pattern(
        name="PASSWORD",
        regex=re.compile(r"\b(PGPASSWORD\s*=\s*)([\"'][^\"']*[\"']|\S+)"),
        replacement=_password_value_replacement("PASSWORD"),
    ),
    # MYSQL_PWD env-var.
    _Pattern(
        name="PASSWORD",
        regex=re.compile(r"\b(MYSQL_PWD\s*=\s*)([\"'][^\"']*[\"']|\S+)"),
        replacement=_password_value_replacement("PASSWORD"),
    ),
    # --password / --passwd / --pass (case-insensitive flag name).
    _Pattern(
        name="PASSWORD",
        regex=re.compile(r"(?i)(--(?:password|passwd|pass)\s*[= ]\s*)([\"'][^\"']*[\"']|\S+)"),
        replacement=_password_value_replacement("PASSWORD"),
    ),
    # Command-scoped -p / -p<value>: only fires inside a mysql/mysqldump/
    # mariadb command or a `docker login` invocation. Avoids redacting
    # generic -p flags (e.g. psql port, ssh port).
    _Pattern(
        name="PASSWORD",
        regex=re.compile(
            r"(\b(?:mysql|mysqldump|mariadb|docker\s+login)\b[^\n|;&]*?\s-p\s*)"
            r"(\S[^\s|;&]*)"
        ),
        replacement=_password_value_replacement("PASSWORD"),
    ),
    # Sensitive token/key/secret/auth flag values (entropy-scoped).
    _Pattern(
        name="CREDENTIAL",
        regex=re.compile(
            r"(?i)(--(?:token|api[-_]?key|secret|access[-_]?key|auth)\s*[= ]\s*)"
            r"([\"'][^\"']*[\"']|\S+)"
        ),
        replacement=_credential_value_replacement("CREDENTIAL"),
    ),
    # URL query params: ?key= / &key= / ?token= / ?api_key= (entropy-scoped).
    _Pattern(
        name="CREDENTIAL",
        regex=re.compile(r"(?i)([?&](?:key|token|api_key)=)([^&\s'\"]+)"),
        replacement=_credential_value_replacement("CREDENTIAL"),
    ),
    # X-API-Key / X-Auth-Token HTTP headers (entropy-scoped).
    _Pattern(
        name="CREDENTIAL",
        regex=re.compile(r"(?i)(x-(?:api-key|auth-token):\s*)(\S+)"),
        replacement=_credential_value_replacement("CREDENTIAL"),
    ),
)


def scrub(text: str) -> str:
    """Scrub secrets from ``text``, replacing each with a ``<REDACTED:KIND>`` marker.

    Pure function. Idempotent: ``scrub(scrub(s)) == scrub(s)``.
    """
    for pat in _PATTERNS:
        text = pat.regex.sub(pat.replacement, text)
    return text


__all__ = ("SCRUBBER_VERSION", "scrub")
