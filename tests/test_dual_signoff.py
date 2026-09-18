"""
Tests for reviewer identity and dual sign-off (Step 5).

An approval with no approver is not an audit trail, and a model the pipeline itself
found to discriminate should not be approvable by one person clicking once. These
tests drive the real graph: toy_df carries a genuine disparity on `sex`, so every
approval here is an approval of a violating model.
"""

from __future__ import annotations

import json
import os

import pytest

from audit_log import get_audit_trail
from policy import default_policy, policy_sha256, with_overrides

pytest.importorskip("langgraph", reason="langgraph not installed")

from tests.conftest import REVIEWER_A, REVIEWER_B, approve, decide  # noqa: E402
from tests.test_graph_end_to_end import _initial_state, _run_to_gate, graph  # noqa: E402,F401

ANON = {"reviewer_id": None, "reviewer_role": None, "reviewer_authenticated": False}


def _payload(env, config):
    snapshot = env.g.get_state(config)
    assert snapshot.tasks and snapshot.tasks[0].interrupts, "expected the run to be paused"
    return snapshot.tasks[0].interrupts[0].value


def _run_with_policy(env, df, policy, thread, **extra):
    config = {"configurable": {"thread_id": thread}}
    env.g.invoke({**_initial_state(df), "policy": policy, "policy_version": policy["version"],
                  "policy_sha256": policy_sha256(policy), **extra}, config=config)
    return config


def _unauditable(toy_df):
    """No protected column at all: the fairness verdict cannot be anything but None."""
    return toy_df.drop(columns=["sex", "race", "age"])


# ---------------------------------------------------------------------------
# The gate tells the reviewer what the click will do
# ---------------------------------------------------------------------------


def test_the_gate_announces_that_an_approval_needs_a_second_reviewer(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-announce")
    payload = _payload(graph, config)

    assert payload["overall_fairness_passed"] is False, "toy_df should violate on sex"
    assert "violation" in payload["dual_signoff_required_reason"]
    assert payload["awaiting_second_approval"] is False
    assert payload["first_approval"] is None
    # Approve stays offered: it is the first half of a two-person decision, not refused.
    assert "approve" in payload["allowed_decisions"]
    assert payload["approval_blocked_reason"] is None


# ---------------------------------------------------------------------------
# One approval is not an approval
# ---------------------------------------------------------------------------


def test_a_single_approval_does_not_finish_a_violating_run(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-one")
    decide(graph.g, config, "approve", "looks acceptable to me", REVIEWER_A)

    snapshot = graph.g.get_state(config)
    assert "human_approval_node" in snapshot.next, "the run must stay at the gate"
    values = snapshot.values
    assert values["awaiting_second_approval"] is True
    assert values["first_approval"]["reviewer_id"] == "test.first"
    assert values["human_decision"] is None, "a half-finished approval is not a decision"
    # Nothing on disk, no paperwork: this is not an approved model.
    assert not values.get("model_saved_path")
    assert values.get("artifacts_manifest") is None
    assert not os.path.isdir(os.path.join(graph.artifacts_dir, "t-signoff-one"))

    payload = _payload(graph, config)
    assert payload["awaiting_second_approval"] is True
    assert payload["first_approval"]["reviewer_id"] == "test.first"
    assert "Second sign-off required" in payload["question"]

    trail = get_audit_trail("t-signoff-one", db_path=graph.audit_db)
    (first,) = [e for e in trail if e["event_type"] == "signoff_first_approval"]
    assert first["details"]["first_approval"]["reviewer_id"] == "test.first"
    assert "test.first" in first["summary"] and "NOT approved" in first["summary"]
    assert not [e for e in trail if e["event_type"] == "final_outcome"]


def test_the_first_approver_cannot_supply_the_second_signoff(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-self")
    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    decide(graph.g, config, "approve", reviewer=REVIEWER_A)

    snapshot = graph.g.get_state(config)
    assert "human_approval_node" in snapshot.next, "still not approved"
    assert not snapshot.values.get("model_saved_path")
    assert snapshot.values["awaiting_second_approval"] is True

    payload = _payload(graph, config)
    assert "test.first" in payload["signoff_error"]
    assert "different reviewer" in payload["signoff_error"]

    trail = get_audit_trail("t-signoff-self", db_path=graph.audit_db)
    (rejected,) = [e for e in trail if e["event_type"] == "signoff_rejected"]
    assert rejected["details"]["attempted_by"]["reviewer_id"] == "test.first"
    # The attempt itself is on the record, as is the decision that produced it.
    assert len([e for e in trail if e["event_type"] == "human_decision"]) == 2


def test_a_refused_self_approval_does_not_count_as_a_second_approver(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-dedupe")
    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    decide(graph.g, config, "approve", reviewer=REVIEWER_A)   # refused
    decide(graph.g, config, "approve", reviewer=REVIEWER_B)

    values = graph.g.get_state(config).values
    assert values["model_saved_path"]
    assert len([d for d in values["reviewer_decisions"] if d["decision"] == "approve"]) == 3,         "every attempt stays in the history"

    final = [e for e in get_audit_trail("t-signoff-dedupe", db_path=graph.audit_db)
             if e["event_type"] == "final_outcome"][-1]
    assert [a["reviewer_id"] for a in final["details"]["approvers"]] ==         ["test.first", "test.second"], "one entry per reviewer, not per approval"

    with open(os.path.join(graph.artifacts_dir, "t-signoff-dedupe", "model_card.json"),
              encoding="utf-8") as fh:
        card = json.load(fh)
    assert [a["reviewer_id"] for a in card["human_governance"]["approvers"]] ==         ["test.first", "test.second"]
    assert len(card["human_governance"]["decision_history"]) == 3


def test_an_approval_with_no_reviewer_id_is_not_a_signoff(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-anon")
    decide(graph.g, config, "approve", reviewer=ANON)

    snapshot = graph.g.get_state(config)
    assert "human_approval_node" in snapshot.next
    assert not snapshot.values.get("model_saved_path")
    assert snapshot.values.get("first_approval") is None, "an unnamed approval holds nothing"
    assert "no reviewer id" in _payload(graph, config)["signoff_error"]

    trail = get_audit_trail("t-signoff-anon", db_path=graph.audit_db)
    (decision,) = [e for e in trail if e["event_type"] == "human_decision"]
    assert decision["details"]["reviewer_id"] is None
    assert "unidentified reviewer" in decision["summary"]


# ---------------------------------------------------------------------------
# Two different reviewers
# ---------------------------------------------------------------------------


def test_a_second_different_reviewer_completes_the_approval(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-two")
    decide(graph.g, config, "approve", "acceptable given the objective", REVIEWER_A)
    decide(graph.g, config, "approve", "confirmed, with the mitigation note", REVIEWER_B)

    snapshot = graph.g.get_state(config)
    assert snapshot.next == (), "the run should be finished"
    values = snapshot.values
    assert values["model_saved_path"] and os.path.exists(values["model_saved_path"])
    assert values["awaiting_second_approval"] is False, "nothing is being waited for now"
    assert values["first_approval"]["reviewer_id"] == "test.first", \
        "the completed run still records who signed first"

    approvals = [d for d in values["reviewer_decisions"] if d["decision"] == "approve"]
    assert [d["reviewer_id"] for d in approvals] == ["test.first", "test.second"]
    assert [d["stage"] for d in approvals] == ["first", "second"]
    assert [d["reviewer_role"] for d in approvals] == ["ml_engineer", "compliance_officer"]

    trail = get_audit_trail("t-signoff-two", db_path=graph.audit_db)
    final = [e for e in trail if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["status"] == "APPROVED"
    assert final["details"]["dual_signoff"] is True
    assert [a["reviewer_id"] for a in final["details"]["approvers"]] == \
        ["test.first", "test.second"]
    assert "test.first" in final["summary"] and "test.second" in final["summary"]


def test_the_model_card_and_aibom_name_both_approvers(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-card")
    approve(graph.g, config, "signed off")

    run_dir = os.path.join(graph.artifacts_dir, "t-signoff-card")
    with open(os.path.join(run_dir, "model_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    governance = card["human_governance"]
    assert governance["dual_signoff"] is True
    assert [a["reviewer_id"] for a in governance["approvers"]] == ["test.first", "test.second"]
    assert governance["reviewer_identities_verified"] is True
    assert [d["reviewer_id"] for d in governance["decision_history"]] == \
        ["test.first", "test.second"]

    markdown = open(os.path.join(run_dir, "model_card.md"), encoding="utf-8").read()
    assert "test.first" in markdown and "test.second" in markdown
    assert "Dual sign-off" in markdown

    with open(os.path.join(run_dir, "aibom.json"), encoding="utf-8") as fh:
        aibom = json.load(fh)
    assert [a["reviewer_id"] for a in aibom["governance"]["approvers"]] == \
        ["test.first", "test.second"]

    annex = open(os.path.join(run_dir, "technical_documentation.md"),
                 encoding="utf-8").read()
    assert "test.first and test.second" in annex


def test_an_unverified_identity_is_recorded_as_unverified(graph, toy_df):
    """No roster configured: the id is the caller's word, and the card says so."""
    config = _run_to_gate(graph, toy_df, "t-signoff-unverified")
    stated = {**REVIEWER_A, "reviewer_authenticated": False}
    approve(graph.g, config, first=stated, second={**REVIEWER_B, "reviewer_authenticated": False})

    with open(os.path.join(graph.artifacts_dir, "t-signoff-unverified", "model_card.json"),
              encoding="utf-8") as fh:
        card = json.load(fh)
    assert card["human_governance"]["reviewer_identities_verified"] is False
    assert any("identity was NOT verified" in limitation
               for limitation in card["limitations"])

    trail = get_audit_trail("t-signoff-unverified", db_path=graph.audit_db)
    decisions = [e for e in trail if e["event_type"] == "human_decision"]
    assert all(e["details"]["reviewer_authenticated"] is False for e in decisions)
    assert "UNVERIFIED" in decisions[0]["summary"]


# ---------------------------------------------------------------------------
# Every decision, not only approvals
# ---------------------------------------------------------------------------


def test_a_rejection_records_who_rejected(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-signoff-reject")
    decide(graph.g, config, "reject_model_or_fairness", "unacceptable gap", REVIEWER_B)

    trail = get_audit_trail("t-signoff-reject", db_path=graph.audit_db)
    (decision,) = [e for e in trail if e["event_type"] == "human_decision"]
    assert decision["details"]["reviewer_id"] == "test.second"
    assert decision["details"]["reviewer_role"] == "compliance_officer"
    assert "test.second" in decision["summary"]


def test_a_reroute_clears_a_pending_signoff(graph, toy_df):
    """The next gate reviews a different model, so a held first approval must lapse."""
    config = _run_to_gate(graph, toy_df, "t-signoff-lapse")
    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    assert graph.g.get_state(config).values["awaiting_second_approval"] is True

    decide(graph.g, config, "reject_model_or_fairness", reviewer=REVIEWER_B)
    values = graph.g.get_state(config).values
    assert values["awaiting_second_approval"] is False
    assert values["first_approval"] is None
    assert _payload(graph, config)["signoff_error"] is None

    # And the second reviewer alone still cannot finish it in one click.
    decide(graph.g, config, "approve", reviewer=REVIEWER_B)
    assert "human_approval_node" in graph.g.get_state(config).next


# ---------------------------------------------------------------------------
# The rule is policy, not code
# ---------------------------------------------------------------------------


def test_the_requirement_can_be_switched_off_in_policy(graph, toy_df):
    policy = with_overrides(default_policy(),
                            {"governance": {"require_dual_signoff_for_violating_approval": False}})
    config = _run_with_policy(graph, toy_df, policy, "t-signoff-off")
    assert _payload(graph, config)["dual_signoff_required_reason"] is None

    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    values = graph.g.get_state(config).values
    assert values["model_saved_path"], "one approval is enough under this policy"
    final = [e for e in get_audit_trail("t-signoff-off", db_path=graph.audit_db)
             if e["event_type"] == "final_outcome"][-1]
    assert final["details"]["dual_signoff"] is False
    assert [a["reviewer_id"] for a in final["details"]["approvers"]] == ["test.first"]


def test_dual_signoff_is_the_override_path_for_unmeasured_fairness(graph, toy_df):
    """
    With the override off, an unmeasured verdict cannot be approved at all. With it on,
    the block becomes a two-person decision rather than a refusal.
    """
    policy = with_overrides(default_policy(),
                            {"governance": {"dual_signoff_can_override_approval_block": True}})
    config = _run_with_policy(graph, _unauditable(toy_df), policy, "t-signoff-override")

    payload = _payload(graph, config)
    assert payload["overall_fairness_passed"] is None
    assert payload["approval_blocked_reason"] is None, "the block is delegated, not applied"
    assert "not measured" in payload["dual_signoff_required_reason"]
    assert "approve" in payload["allowed_decisions"]

    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    assert "human_approval_node" in graph.g.get_state(config).next
    assert not graph.g.get_state(config).values.get("model_saved_path")

    decide(graph.g, config, "approve", reviewer=REVIEWER_B)
    values = graph.g.get_state(config).values
    assert values["model_saved_path"], "two reviewers may sign off what one may not"
    assert values.get("unresolved_approval_blocked") is not True


def test_without_the_override_no_pair_of_reviewers_can_approve_unmeasured_fairness(graph, toy_df):
    config = _run_to_gate(graph, _unauditable(toy_df), "t-signoff-no-override")
    payload = _payload(graph, config)
    assert payload["approval_blocked_reason"]
    assert payload["dual_signoff_required_reason"] is None
    assert "approve" not in payload["allowed_decisions"]

    decide(graph.g, config, "approve", reviewer=REVIEWER_A)
    values = graph.g.get_state(config).values
    assert values["unresolved_approval_blocked"] is True
    assert not values.get("model_saved_path")
