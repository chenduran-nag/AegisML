"""
server.py
=========
FastAPI REST Server for the AI-Governed Multi-Agent Data Science Platform.

Provides RESTful API endpoints for:
  - Starting initial pipeline execution (/api/pipeline/start)
  - Submitting human governance decisions (/api/pipeline/resume)
  - Inspecting pipeline status (/api/pipeline/status/{thread_id})
  - Retrieving audit log trail (/api/pipeline/audit/{thread_id})
  - Serving static Web UI dashboard (/)
"""

from __future__ import annotations

import hashlib
import os
import uuid
import io
import pandas as pd
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Auto-load .env if GROQ_API_KEY is not already in environment
if "GROQ_API_KEY" not in os.environ:
    env_paths = [".env", os.path.join(os.path.dirname(__file__), ".env")]
    for path in env_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[7:]
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")

from langgraph.types import Command
from graph_state import df_to_bytes
from pipeline_graph import graph
from audit_log import get_audit_trail, verify_audit_chain
from compliance_artifacts import verify_artifacts

app = FastAPI(
    title="AI Multi-Agent Governance API",
    description="REST backend for LangGraph Multi-Agent Data Science Governance Platform",
    version="1.0.0",
)

# NOTE: allow_credentials=True is INVALID alongside allow_origins=["*"] — the
# spec forbids the wildcard on credentialed requests and browsers reject the
# response outright. This API uses no cookies or auth headers, so credentials are
# simply off. If authentication is added later, replace the wildcard with an
# explicit origin list and only then re-enable credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str
    human_feedback: Optional[str] = None


def _build_pipeline_response(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    next_nodes = list(snapshot.next)

    is_paused = "human_approval_node" in next_nodes
    payload = None
    if is_paused and snapshot.tasks and snapshot.tasks[0].interrupts:
        payload = snapshot.tasks[0].interrupts[0].value

    values = snapshot.values or {}
    fairness_result = values.get("fairness_result") or {}

    # The real path written by audit_log_node, or None. Never synthesise a path
    # here: showing a reviewer an artifact location that does not exist on disk
    # is worse than showing nothing.
    saved_path = values.get("model_saved_path")

    manifest = values.get("artifacts_manifest") or {}

    return {
        "thread_id": thread_id,
        "status": "paused" if is_paused else ("completed" if not next_nodes else "running"),
        "next_nodes": next_nodes,
        "review_payload": payload,
        "model_saved_path": saved_path,
        "artifacts": sorted((manifest.get("files") or {}).keys()),
        "values": {
            "unresolved_human_rejection": values.get("unresolved_human_rejection", False),
            "unresolved_quality_issue": values.get("unresolved_quality_issue", False),
            "unresolved_training_failure": values.get("unresolved_training_failure", False),
            "human_decision": values.get("human_decision"),
            "human_feedback": values.get("human_feedback"),
            "retry_count": values.get("retry_count", 0),
            "rejection_reroute_count": values.get("rejection_reroute_count", 0),
            "model_saved_path": saved_path,
            "model_save_error": values.get("model_save_error"),
            "dataset_sha256": values.get("dataset_sha256"),
            "selected_model_name": (values.get("training_result") or {}).get("selected_model_name"),
            "overall_fairness_passed": fairness_result.get("overall_fairness_passed"),
            "fairness_evaluated": fairness_result.get("fairness_evaluated", False),
            "fairness_coverage": fairness_result.get("fairness_coverage"),
            "protected_attributes_unaudited": fairness_result.get("protected_attributes_unaudited", []),
        },
    }


@app.post("/api/pipeline/start")
async def start_pipeline(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...),
    business_objective: Optional[str] = Form(None),
):
    """
    Ingest uploaded CSV, construct initial state, and invoke pipeline graph
    until paused at human_approval_node or completed. Exploratory analysis runs
    inside the graph as data_analysis_node.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()

    # Hash the uploaded bytes, not the parsed frame. This digest is the dataset's
    # provenance anchor in the AI Bill of Materials, so it has to identify the file
    # the user actually supplied.
    dataset_sha256 = hashlib.sha256(contents).hexdigest()

    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {exc}")

    if target_column not in df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in CSV headers: {list(df_raw.columns)}"
        )

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set in environment or .env file.",
        )

    thread_id = f"run-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "df_bytes": df_to_bytes(df_raw),
        "dataset_sha256": dataset_sha256,
        "target_column": target_column,
        "task_type": task_type,
        "business_objective": business_objective or "",
        "retry_count": 0,
        "unresolved_quality_issue": False,
        "last_failure_reason": None,
        "human_decision": None,
        "human_feedback": None,
        "rejection_reroute_count": 0,
        "unresolved_human_rejection": False,
        "rejected_models": [],
    }

    # graph.invoke() is synchronous and runs the whole pipeline — model training
    # included. Calling it directly from an async handler would block the event
    # loop for its entire duration, freezing every other request (including the
    # dashboard's own status polls). run_in_threadpool hands it to a worker.
    try:
        await run_in_threadpool(graph.invoke, initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {exc}")

    res = _build_pipeline_response(thread_id)
    values = graph.get_state(config).values or {}
    res["eda_report"] = {
        **(values.get("eda_report") or {}),
        "findings": values.get("eda_findings") or [],
    }
    return res


@app.get("/api/pipeline/eda/{thread_id}")
async def get_eda_report(thread_id: str):
    """
    Get Data Analysis Agent EDA report for a given run thread_id.
    """
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    report = values.get("eda_report")
    if not report:
        raise HTTPException(status_code=404, detail="EDA report not found for this run")
    return {**report, "findings": values.get("eda_findings") or []}


@app.post("/api/pipeline/resume")
async def resume_pipeline(req: ResumeRequest):
    """
    Submit human governance decision to resume graph execution.
    """
    allowed = ("approve", "reject_data_quality", "reject_model_or_fairness", "reject_and_mitigate")
    if req.decision not in allowed:
        raise HTTPException(status_code=400, detail=f"Decision must be one of {allowed}")

    config = {"configurable": {"thread_id": req.thread_id}}
    resume_payload = {
        "decision": req.decision,
        "human_feedback": req.human_feedback or "",
    }

    try:
        await run_in_threadpool(
            graph.invoke, Command(resume=resume_payload), config=config
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error resuming graph execution: {exc}")

    return _build_pipeline_response(req.thread_id)


@app.get("/api/pipeline/status/{thread_id}")
async def get_pipeline_status(thread_id: str):
    """
    Get current snapshot status for a given thread_id.
    """
    return _build_pipeline_response(thread_id)


@app.get("/api/pipeline/artifacts/{thread_id}")
async def get_artifacts(thread_id: str):
    """
    List the compliance artifacts generated for a run and re-verify their digests
    against the hash-chained audit log.
    """
    result = verify_artifacts(thread_id)
    if result["verified"] is None:
        raise HTTPException(
            status_code=404,
            detail="No compliance artifacts were generated for this run "
                   "(they are produced only on the approve path).",
        )
    return {"thread_id": thread_id, **result}


@app.get("/api/pipeline/artifact/{thread_id}/{filename}")
async def get_artifact_file(thread_id: str, filename: str):
    """
    Return the text of one generated artifact.

    Only names appearing in the audit-logged manifest for this run are served, so
    the path cannot be steered outside the run's own artifact directory.
    """
    result = verify_artifacts(thread_id)
    match = next((f for f in result["files"] if f["file"] == filename), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' is not a recorded artifact for this run.",
        )
    if match["status"] != "OK":
        raise HTTPException(
            status_code=409,
            detail=f"Artifact '{filename}' failed integrity verification "
                   f"({match['status']}); refusing to serve it.",
        )
    with open(match["path"], encoding="utf-8") as fh:
        return {"thread_id": thread_id, "filename": filename, "content": fh.read()}


@app.get("/api/pipeline/audit/{thread_id}")
async def get_audit_log(thread_id: str):
    """
    Retrieve the chronological audit log trail for a given thread_id, together
    with the result of re-verifying its hash chain.
    """
    trail = get_audit_trail(thread_id)
    chain = verify_audit_chain(thread_id)
    return {"thread_id": thread_id, "entries": trail, "chain": chain}


# Serve static web frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
