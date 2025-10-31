"""FastAPI application exposing training and inference utilities."""
from __future__ import annotations

import logging
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
_JOB_METADATA: Dict[str, Dict[str, Any]] = {}
app = FastAPI(title="GridBuilder Service", version="0.1.0")


class CLIRequest(BaseModel):
    """Request payload describing CLI arguments."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = Field(default=None, description="Optional identifier for the spawned job")


class HFUploadConfig(BaseModel):
    """Configuration for uploading training artifacts to HuggingFace Hub."""

    repo_id: str = Field(..., description="Target HuggingFace repository identifier")
    token: str = Field(..., description="Token used for authentication")
    branch: str = Field("main", description="Repository branch to push to")
    path_in_repo: str = Field("", description="Path inside the repository where files will be uploaded")
    private: bool = Field(False, description="Create the repository as private when allowed")
    allow_create: bool = Field(False, description="Create the repository if it does not already exist")
    commit_message: str = Field(
        "Add GridBuilder training artifacts",
        description="Commit message to use for the upload",
    )
    description: Optional[str] = Field(
        None, description="Optional README.md contents describing the model"
    )
    repo_type: str = Field("model", description="Type of repository to upload to")


class HFModelDownloadConfig(BaseModel):
    """Configuration describing how to download a model snapshot from HuggingFace Hub."""

    repo_id: str = Field(..., description="Source HuggingFace repository identifier")
    token: Optional[str] = Field(None, description="Authentication token for private repositories")
    revision: Optional[str] = Field(None, description="Branch, tag, or commit to download")
    subfolder: Optional[str] = Field(None, description="Optional subfolder within the repository")
    local_dir: Optional[str] = Field(
        None, description="Directory where the snapshot should be stored"
    )
    allow_patterns: Optional[List[str]] = Field(
        None, description="Glob patterns of files to include when downloading"
    )
    ignore_patterns: Optional[List[str]] = Field(
        None, description="Glob patterns of files to exclude when downloading"
    )
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from CLI parameter names to relative paths inside the snapshot",
    )
    repo_type: str = Field("model", description="Repository type to download from")


class TrainRequest(CLIRequest):
    hf_upload: Optional[HFUploadConfig] = Field(
        default=None,
        description="Optional configuration for uploading artifacts to HuggingFace Hub",
    )


class InferRequest(CLIRequest):
    hf_model: Optional[HFModelDownloadConfig] = Field(
        default=None,
        description="Optional configuration for downloading model artifacts from HuggingFace Hub",
    )


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


def _sanitize_command(command: List[str], sensitive_flags: Iterable[str]) -> List[str]:
    sanitized = list(command)
    indices = {flag: [] for flag in sensitive_flags}
    for idx, token in enumerate(command):
        if token in indices:
            indices[token].append(idx)
    for positions in indices.values():
        for pos in positions:
            if pos + 1 < len(sanitized):
                sanitized[pos + 1] = "***"
    return sanitized


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_readme(out_dir: Path, description: str) -> None:
    try:
        _ensure_directory(out_dir)
        (out_dir / "README.md").write_text(description, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write README.md: {exc}") from exc


def _perform_hf_upload(out_dir: Path, config: HFUploadConfig) -> Optional[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required to upload artifacts to HuggingFace Hub") from exc

    api = HfApi()
    if config.allow_create:
        logging.info("Ensuring HuggingFace repository %s exists", config.repo_id)
        api.create_repo(
            repo_id=config.repo_id,
            token=config.token,
            repo_type=config.repo_type,
            private=config.private,
            exist_ok=True,
        )

    logging.info(
        "Uploading artifacts from %s to HuggingFace Hub repo %s (branch %s)",
        out_dir,
        config.repo_id,
        config.branch,
    )
    commit_info = api.upload_folder(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        folder_path=str(out_dir),
        path_in_repo=config.path_in_repo,
        token=config.token,
        commit_message=config.commit_message,
        revision=config.branch,
    )
    logging.info("Upload to HuggingFace Hub completed successfully")
    return getattr(commit_info, "commit_hash", None)


def _download_hf_snapshot(job_id: str, config: HFModelDownloadConfig) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required to download artifacts from HuggingFace Hub") from exc

    if config.local_dir:
        base_dir = Path(config.local_dir)
        if not base_dir.is_absolute():
            base_dir = REPO_ROOT / base_dir
    else:
        base_dir = REPO_ROOT / "models" / "hf_downloads" / job_id
    _ensure_directory(base_dir)

    snapshot_path = snapshot_download(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        revision=config.revision,
        token=config.token,
        subfolder=config.subfolder,
        local_dir=str(base_dir),
        local_dir_use_symlinks=False,
        allow_patterns=config.allow_patterns,
        ignore_patterns=config.ignore_patterns,
    )
    return Path(snapshot_path)


def _monitor_training_job(
    job_id: str,
    managed: "ManagedProcess",
    upload_config: Optional[HFUploadConfig],
    out_dir: Path,
) -> None:
    returncode = managed.process.wait()
    if not upload_config:
        return

    metadata = _JOB_METADATA.setdefault(job_id, {})
    if returncode != 0:
        metadata["hf_upload_status"] = f"skipped: training failed (returncode {returncode})"
        return

    try:
        commit_hash = _perform_hf_upload(out_dir, upload_config)
    except Exception as exc:  # pragma: no cover - network failures are environment specific
        logging.exception("Failed to upload artifacts for job %s: %s", job_id, exc)
        metadata["hf_upload_status"] = f"error: {exc}"
    else:
        metadata["hf_upload_status"] = "success"
        if commit_hash:
            metadata["hf_upload_commit"] = commit_hash


def _wait_for_completion(managed: "ManagedProcess") -> None:
    managed.process.wait()


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
    parameters = dict(request.parameters)
    _validate_training_parameters(parameters)
    job_id = _normalize_job_id("train", request.job_id)
    resolved_out_dir = _resolve_subpath(str(parameters["out_dir"]))

    metadata = _JOB_METADATA.setdefault(job_id, {})

    if request.hf_upload:
        metadata["hf_upload_repo"] = request.hf_upload.repo_id
        metadata["hf_upload_branch"] = request.hf_upload.branch
        metadata["hf_upload_path_in_repo"] = request.hf_upload.path_in_repo
        metadata["hf_upload_status"] = "pending"
        if request.hf_upload.description:
            _write_readme(resolved_out_dir, request.hf_upload.description)

    command = _build_cli_command(TRAIN_SCRIPT, parameters)
    tensorboard_value = parameters.get("tensorboard_logdir")
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
    sanitized = _sanitize_command(managed.command, ["--hf-token", "--token"])
    managed.command = sanitized
    background.add_task(_monitor_training_job, job_id, managed, request.hf_upload, resolved_out_dir)
    metadata["tensorboard_log"] = str(tensorboard_log_path) if tensorboard_log_path else None
    return {
        "job_id": job_id,
        "command": sanitized,
        "log_path": str(managed.log_path),
        "status": managed.status(),
    }


@app.post("/infer")
def start_inference(request: InferRequest, background: BackgroundTasks) -> Dict[str, Any]:
    parameters = dict(request.parameters)
    job_id = _normalize_job_id("infer", request.job_id)
    metadata = _JOB_METADATA.setdefault(job_id, {})

    if request.hf_model:
        metadata["hf_download_repo"] = request.hf_model.repo_id
        metadata["hf_download_revision"] = request.hf_model.revision
        try:
            snapshot_path = _download_hf_snapshot(job_id, request.hf_model)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network failures vary
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download model from HuggingFace Hub: {exc}",
            ) from exc

        file_map = dict(request.hf_model.files) if request.hf_model.files else {
            "model": "model.pt",
            "vocab": "vocab.json",
            "config": "inference_config.json",
        }
        resolved_files: Dict[str, str] = {}
        for param_key, relative_path in file_map.items():
            resolved_path = snapshot_path / relative_path
            if not resolved_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Expected file '{relative_path}' for parameter '{param_key}' "
                        f"not found in HuggingFace snapshot {snapshot_path}"
                    ),
                )
            resolved_files[param_key] = str(resolved_path)
        parameters.update(resolved_files)
        metadata["hf_download_path"] = str(snapshot_path)
        metadata["hf_download_files"] = dict(resolved_files)

    _validate_inference_parameters(parameters)
    command = _build_cli_command(INFER_SCRIPT, parameters)
    try:
        managed = process_manager.start(job_id=job_id, command=command, cwd=REPO_ROOT)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    sanitized = _sanitize_command(managed.command, ["--hf-token", "--token"])
    managed.command = sanitized
    background.add_task(_wait_for_completion, managed)
    return {
        "job_id": job_id,
        "command": sanitized,
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
