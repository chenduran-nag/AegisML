"""
A finished run must still be reviewable.

Once the graph ends there is no live interrupt, so the status API used to return
`review_payload: null` and the dashboard's evaluation tabs rendered empty for every
completed run. The gate node now keeps the payload the reviewer decided on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from langgraph.types import Command  # noqa: E402

from tests.test_graph_end_to_end import _run_to_gate, graph  # noqa: E402,F401


def test_a_completed_run_keeps_the_payload_the_reviewer_decided_on(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-last-payload")
    live = graph.g.get_state(config).tasks[0].interrupts[0].value
    assert not graph.g.get_state(config).values.get("last_review_payload"), \
        "nothing is stored before the reviewer decides"

    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)
    values = graph.g.get_state(config).values

    kept = values["last_review_payload"]
    assert kept["selected_model_name"] == live["selected_model_name"]
    assert kept["leaderboard"] == live["leaderboard"]
    assert kept["fairness_report"] == live["fairness_report"]


def test_after_a_reroute_the_kept_payload_is_the_latest_gate(graph, toy_df):
    config = _run_to_gate(graph, toy_df, "t-last-payload-reroute")
    graph.g.invoke(Command(resume={"decision": "reject_model_or_fairness", "human_feedback": ""}),
                   config=config)
    second = graph.g.get_state(config).tasks[0].interrupts[0].value
    graph.g.invoke(Command(resume={"decision": "approve", "human_feedback": ""}), config=config)

    kept = graph.g.get_state(config).values["last_review_payload"]
    assert kept["selected_model_name"] == second["selected_model_name"]
