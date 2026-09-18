"""
reviewers.py
============
The reviewer roster: who may submit a governance decision, and under what name it
is recorded.

DESIGN / WHY
  - An approval with no approver is not an audit trail. Every decision has to carry
    an identity, and that identity has to come from something the caller had to
    possess — otherwise "approved by compliance" is just a string anyone can type.
  - The mechanism is deliberately small: a gitignored `reviewers.yaml` maps a bearer
    token to an id and a role, and the server reads the token from the
    `X-Reviewer-Token` header. That is enough to make dual sign-off mean two people,
    and it is the honest limit of what this project claims. THREAT MODEL, stated
    plainly: shared-secret tokens in a local file over HTTP, with no expiry, no
    revocation list, no rotation and no transport security. It stops a reviewer from
    signing off twice under two names; it does not stop anyone who can read the file,
    sniff the connection, or write to the database.
  - Tokens are kept in memory as SHA-256 digests and compared with
    `hmac.compare_digest`, so a token never reaches a log line, a traceback or the
    audit trail, and comparison does not leak length by timing.
  - Validation is strict, like `policy.py`: a duplicate id, a reused token, an unknown
    role or a short token raises at startup rather than producing a roster that
    silently does less than it looks like it does.
  - No roster file is a legitimate configuration (a fresh clone, the test suite, the
    evaluation harness). Identity is then whatever the caller states and is recorded
    as UNVERIFIED — never as though it had been checked.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_REVIEWERS_PATH = Path(__file__).resolve().parent / "reviewers.yaml"

# Roles a decision can be recorded under. A closed set, because these strings end up
# in the audit trail and the model card, where "complaince_officer" would be permanent.
KNOWN_ROLES = (
    "ml_engineer",
    "data_scientist",
    "compliance_officer",
    "domain_expert",
    "product_owner",
)

# Short enough to type, long enough not to be guessed in a term project's threat model.
MIN_TOKEN_LENGTH = 24

# What an unauthenticated identity is recorded as. Not a role, and deliberately not
# blank: a reader of the audit trail must see that nothing was verified.
UNVERIFIED = "unverified"


class ReviewerError(ValueError):
    """The reviewer roster is unreadable or invalid."""


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_roster(raw: Any) -> dict:
    """
    Return a validated roster: {"version", "reviewers": [{"id", "role", "token_sha256"}]}.

    Raises ReviewerError on anything questionable. Plaintext tokens are dropped here
    and never stored.
    """
    if not isinstance(raw, dict):
        raise ReviewerError("the reviewer roster must be a YAML mapping")
    unknown = set(raw) - {"version", "reviewers"}
    if unknown:
        raise ReviewerError(f"unknown top-level key(s): {sorted(unknown)}")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ReviewerError("the roster needs a non-empty string 'version'")
    entries = raw.get("reviewers")
    if not isinstance(entries, list) or not entries:
        raise ReviewerError("'reviewers' must be a non-empty list")

    reviewers: list[dict] = []
    seen_ids: set[str] = set()
    seen_tokens: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        where = f"reviewers[{position}]"
        if not isinstance(entry, dict):
            raise ReviewerError(f"{where} must be a mapping")
        unknown = set(entry) - {"id", "role", "token"}
        if unknown:
            raise ReviewerError(f"unknown key(s) in {where}: {sorted(unknown)}")
        reviewer_id = entry.get("id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ReviewerError(f"{where} needs a non-empty string 'id'")
        reviewer_id = reviewer_id.strip()
        if reviewer_id in seen_ids:
            raise ReviewerError(
                f"{where}: duplicate reviewer id '{reviewer_id}'. Two reviewers sharing "
                "an id would let one person satisfy dual sign-off alone."
            )
        role = entry.get("role")
        if role not in KNOWN_ROLES:
            raise ReviewerError(
                f"{where}: role {role!r} is not one of {list(KNOWN_ROLES)}"
            )
        token = entry.get("token")
        if not isinstance(token, str) or len(token.strip()) < MIN_TOKEN_LENGTH:
            raise ReviewerError(
                f"{where}: 'token' must be a string of at least {MIN_TOKEN_LENGTH} "
                "characters (generate one with `python reviewers.py --new-token`)"
            )
        digest = token_digest(token.strip())
        if digest in seen_tokens:
            raise ReviewerError(f"{where}: this token is already used by another reviewer")
        seen_ids.add(reviewer_id)
        seen_tokens.add(digest)
        reviewers.append({"id": reviewer_id, "role": role, "token_sha256": digest})

    return {"version": version.strip(), "reviewers": reviewers}


def load_reviewers(path: str | Path | None = None) -> Optional[dict]:
    """
    Load and validate the roster, or return None when no roster file exists.

    Returns {"version", "reviewers", "source"}. A file that exists but is invalid
    raises ReviewerError: starting up with a roster that was quietly ignored would
    make every decision look authenticated when none of them were.
    """
    source = Path(path) if path is not None else DEFAULT_REVIEWERS_PATH
    if not source.exists():
        return None
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewerError(f"cannot read reviewer roster '{source}': {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ReviewerError(f"roster '{source}' is not valid YAML: {exc}") from exc
    roster = validate_roster(raw)
    return {**roster, "source": str(source)}


def identify(roster: Optional[dict], token: Optional[str]) -> Optional[dict]:
    """
    The reviewer a token belongs to, or None if it belongs to nobody.

    Compared over digests with compare_digest, so neither the token nor its length
    leaks through timing, and the token itself is never held for longer than this call.
    """
    if not roster or not token:
        return None
    presented = token_digest(token.strip())
    for reviewer in roster.get("reviewers") or []:
        if hmac.compare_digest(presented, reviewer["token_sha256"]):
            return {"reviewer_id": reviewer["id"], "reviewer_role": reviewer["role"]}
    return None


def reviewer_ids(roster: Optional[dict]) -> list[str]:
    return [r["id"] for r in (roster or {}).get("reviewers") or []]


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import secrets
    import sys

    if "--new-token" in sys.argv:
        print(secrets.token_urlsafe(32))
    else:
        roster = load_reviewers()
        if roster is None:
            print(f"No roster at {DEFAULT_REVIEWERS_PATH}. "
                  "Copy reviewers.example.yaml to reviewers.yaml to enable "
                  "authenticated reviewer identities.")
        else:
            print(f"{roster['source']} (version {roster['version']}): "
                  + ", ".join(f"{r['id']} [{r['role']}]" for r in roster["reviewers"]))
