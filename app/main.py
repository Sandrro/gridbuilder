"""FastAPI application exposing training and inference utilities."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .process_manager import ProcessManager


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
TRAIN_SCRIPT = REPO_ROOT / "train.py"
INFER_SCRIPT = REPO_ROOT / "infer.py"

process_manager = ProcessManager(LOG_DIR)
_JOB_METADATA: Dict[str, Dict[str, Optional[str]]] = {}
app = FastAPI(title="GridBuilder Service", version="0.1.0")


class CLIRequest(BaseModel):
    """Request payload describing CLI arguments."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = Field(default=None, description="Optional identifier for the spawned job")


class TrainRequest(CLIRequest):
    pass


class InferRequest(CLIRequest):
    pass


def _normalize_job_id(prefix: str, requested: Optional[str]) -> str:
    if requested:
        return requested
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _build_cli_command(script: Path, parameters: Dict[str, Any]) -> List[str]:
    command: List[str] = [sys.executable, str(script)]
    for key, value in parameters.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            for item in value:
                command.extend([flag, str(item)])
            continue
        command.extend([flag, str(value)])
    return command


def _validate_training_parameters(parameters: Dict[str, Any]) -> None:
    required = {"descriptions", "grid", "out_dir"}
    missing = [key for key in required if parameters.get(key) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required training parameters: {', '.join(missing)}")


def _validate_inference_parameters(parameters: Dict[str, Any]) -> None:
    required = {"model", "vocab", "config", "zone_type", "budget_json", "out_parquet"}
    missing = [key for key in required if parameters.get(key) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required inference parameters: {', '.join(missing)}")


@app.post("/train")
def start_training(request: TrainRequest, background: BackgroundTasks) -> Dict[str, Any]:
    _validate_training_parameters(request.parameters)
    job_id = _normalize_job_id("train", request.job_id)
    command = _build_cli_command(TRAIN_SCRIPT, request.parameters)
    resolved_out_dir = _resolve_subpath(str(request.parameters["out_dir"]))
    tensorboard_value = request.parameters.get("tensorboard_logdir")
    if tensorboard_value is None:
        tensorboard_value = "tensorboard"
    tensorboard_log_path: Optional[Path]
    if tensorboard_value:
        tensorboard_log_path = resolved_out_dir / Path(str(tensorboard_value))
    else:
        tensorboard_log_path = None
    try:
        managed = process_manager.start(job_id=job_id, command=command, cwd=REPO_ROOT)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background.add_task(lambda: managed.process.wait())
    metadata = _JOB_METADATA.setdefault(job_id, {})
    metadata["tensorboard_log"] = str(tensorboard_log_path) if tensorboard_log_path else None
    return {
        "job_id": job_id,
        "command": managed.command,
        "log_path": str(managed.log_path),
        "status": managed.status(),
    }


@app.post("/infer")
def start_inference(request: InferRequest, background: BackgroundTasks) -> Dict[str, Any]:
    _validate_inference_parameters(request.parameters)
    job_id = _normalize_job_id("infer", request.job_id)
    command = _build_cli_command(INFER_SCRIPT, request.parameters)
    try:
        managed = process_manager.start(job_id=job_id, command=command, cwd=REPO_ROOT)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background.add_task(lambda: managed.process.wait())
    return {
        "job_id": job_id,
        "command": managed.command,
        "log_path": str(managed.log_path),
        "status": managed.status(),
    }


@app.get("/logs/{job_id}")
def read_logs(job_id: str, limit: int = Query(200, ge=1, le=2000)) -> Dict[str, Any]:
    try:
        lines = process_manager.tail(job_id, limit=limit)
        status = process_manager.status(job_id)
    except KeyError:
        log_path = LOG_DIR / f"{job_id}.log"
        if not log_path.exists():
            raise HTTPException(status_code=404, detail=f"Logs for job '{job_id}' not found")
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            from collections import deque

            lines = list(deque(handle, maxlen=limit))
        status = {"job_id": job_id, "status": "unknown", "returncode": None, "command": [], "log_path": str(log_path)}
    return {"job_id": job_id, "lines": lines, "status": status}


def _resolve_subpath(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise HTTPException(status_code=400, detail="Requested path is outside the repository")
    return resolved


@app.get("/tensorboard")
def tensorboard_scalars(run: str = Query(..., description="Relative path to the TensorBoard run directory"), tags: Optional[List[str]] = Query(None)) -> Dict[str, Any]:
    log_dir = _resolve_subpath(run)
    if not log_dir.exists():
        raise HTTPException(status_code=404, detail=f"TensorBoard directory '{log_dir}' does not exist")

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="tensorboard is not installed") from exc

    accumulator = EventAccumulator(str(log_dir))
    try:
        accumulator.Reload()
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to read TensorBoard logs: {exc}") from exc

    available = accumulator.Tags().get("scalars", [])
    selected_tags = tags or available

    scalars: Dict[str, List[Dict[str, float]]] = {}
    for tag in selected_tags:
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        scalars[tag] = [
            {"step": event.step, "value": float(event.value), "wall_time": float(event.wall_time)}
            for event in events
        ]

    return {"run": str(log_dir), "available_tags": available, "scalars": scalars}


@app.get("/processes")
def list_processes() -> Dict[str, Any]:
    jobs = []
    for job_id in process_manager.known_jobs():
        status = dict(process_manager.status(job_id))
        metadata = _JOB_METADATA.get(job_id, {})
        status.update(metadata)
        status.setdefault("tensorboard_log", None)
        jobs.append(status)
    return {"jobs": jobs}


__all__ = ["app"]
