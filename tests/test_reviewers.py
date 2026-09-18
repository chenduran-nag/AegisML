"""
Tests for the reviewer roster (reviewers.py).

The roster's whole job is to make "two reviewers" mean two people. So the things
worth pinning are the ones that would quietly defeat that: a duplicate id, a shared
token, a role typo that ends up in a model card, and a plaintext token surviving
anywhere in memory.
"""

from __future__ import annotations

import textwrap

import pytest

from reviewers import (
    KNOWN_ROLES,
    MIN_TOKEN_LENGTH,
    ReviewerError,
    identify,
    load_reviewers,
    reviewer_ids,
    token_digest,
    validate_roster,
)

TOKEN_A = "a" * MIN_TOKEN_LENGTH
TOKEN_B = "b" * MIN_TOKEN_LENGTH


def _write(tmp_path, text):
    path = tmp_path / "reviewers.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _roster(tmp_path):
    return load_reviewers(_write(tmp_path, f"""
        version: "1.0.0"
        reviewers:
          - id: a.kumar
            role: ml_engineer
            token: "{TOKEN_A}"
          - id: r.mehta
            role: compliance_officer
            token: "{TOKEN_B}"
    """))


def test_a_valid_roster_loads_with_hashed_tokens(tmp_path):
    roster = _roster(tmp_path)
    assert reviewer_ids(roster) == ["a.kumar", "r.mehta"]
    assert roster["reviewers"][0]["token_sha256"] == token_digest(TOKEN_A)
    assert all("token" not in r for r in roster["reviewers"]), \
        "a plaintext token must not survive loading"
    assert TOKEN_A not in repr(roster)


def test_a_token_identifies_its_reviewer_and_nothing_else(tmp_path):
    roster = _roster(tmp_path)
    assert identify(roster, TOKEN_A) == {"reviewer_id": "a.kumar",
                                         "reviewer_role": "ml_engineer"}
    assert identify(roster, f"  {TOKEN_B}  ") == {"reviewer_id": "r.mehta",
                                                  "reviewer_role": "compliance_officer"}
    assert identify(roster, "c" * MIN_TOKEN_LENGTH) is None
    assert identify(roster, "") is None
    assert identify(roster, None) is None
    assert identify(None, TOKEN_A) is None, "no roster identifies nobody"


def test_no_roster_file_is_a_legitimate_configuration(tmp_path):
    assert load_reviewers(tmp_path / "absent.yaml") is None


@pytest.mark.parametrize("text,match", [
    (f'version: "1"\nreviewers:\n  - id: a\n    role: ml_engineer\n    token: "{TOKEN_A}"\n'
     f'  - id: a\n    role: data_scientist\n    token: "{TOKEN_B}"\n', "duplicate reviewer id"),
    (f'version: "1"\nreviewers:\n  - id: a\n    role: ml_engineer\n    token: "{TOKEN_A}"\n'
     f'  - id: b\n    role: data_scientist\n    token: "{TOKEN_A}"\n', "already used"),
    (f'version: "1"\nreviewers:\n  - id: a\n    role: auditor\n    token: "{TOKEN_A}"\n', "role"),
    ('version: "1"\nreviewers:\n  - id: a\n    role: ml_engineer\n    token: "short"\n',
     "at least"),
    (f'version: "1"\nreviewers:\n  - id: " "\n    role: ml_engineer\n    token: "{TOKEN_A}"\n',
     "non-empty string 'id'"),
    (f'reviewers:\n  - id: a\n    role: ml_engineer\n    token: "{TOKEN_A}"\n', "version"),
    ('version: "1"\nreviewers: []\n', "non-empty list"),
    ('version: "1"\n', "non-empty list"),
    (f'version: "1"\nreviewers:\n  - id: a\n    role: ml_engineer\n    token: "{TOKEN_A}"\n'
     '    extra: 1\n', "unknown key"),
    (f'version: "1"\nteam:\n  - id: a\n', "unknown top-level"),
    ('version: "1"\nreviewers:\n  - [unclosed\n', "not valid YAML"),
])
def test_an_invalid_roster_fails_loudly(tmp_path, text, match):
    with pytest.raises(ReviewerError, match=match):
        load_reviewers(_write(tmp_path, text))


def test_the_committed_example_roster_is_a_real_template():
    """It must parse and name real roles — with tokens that are obviously placeholders."""
    import yaml

    raw = yaml.safe_load(open("reviewers.example.yaml", encoding="utf-8").read())
    assert {r["role"] for r in raw["reviewers"]} <= set(KNOWN_ROLES)
    assert len({r["id"] for r in raw["reviewers"]}) == len(raw["reviewers"])
    assert all("REPLACE-ME" in r["token"] for r in raw["reviewers"])
    # It validates, so a user who only edits the tokens gets a working roster.
    assert validate_roster(raw)["version"] == raw["version"]
