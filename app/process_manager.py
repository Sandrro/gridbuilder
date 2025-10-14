"""Utilities for launching and tracking long running CLI processes."""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional


@dataclass
class ManagedProcess:
    """Holds metadata for a spawned subprocess."""

    job_id: str
    command: List[str]
    process: subprocess.Popen[str]
    log_path: Path
    started_at: float = field(default_factory=time.time)
    _buffer: Deque[str] = field(default_factory=lambda: deque(maxlen=1000), init=False)

    def status(self) -> str:
        """Return human friendly status for the subprocess."""
        return "running" if self.process.poll() is None else "finished"

    def returncode(self) -> Optional[int]:
        """Return the process return code when finished."""
        return self.process.poll()


class ProcessManager:
    """Tracks subprocesses and persists their combined stdout/stderr to disk."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: Dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def start(self, job_id: str, command: Iterable[str], cwd: Optional[Path] = None) -> ManagedProcess:
        """Start a subprocess and begin streaming its output to disk."""
        with self._lock:
            if job_id in self._processes and self._processes[job_id].status() == "running":
                raise RuntimeError(f"Job '{job_id}' is already running")

            log_path = self.log_dir / f"{job_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")

            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            managed = ManagedProcess(job_id=job_id, command=list(command), process=process, log_path=log_path)
            self._processes[job_id] = managed

            thread = threading.Thread(target=self._pump_output, args=(managed,), daemon=True)
            thread.start()
        return managed

    def _pump_output(self, managed: ManagedProcess) -> None:
        assert managed.process.stdout is not None
        with managed.log_path.open("a", encoding="utf-8") as log_file:
            for line in managed.process.stdout:
                log_file.write(line)
                log_file.flush()
                managed._buffer.append(line.rstrip("\n"))
        managed.process.wait()

    def get(self, job_id: str) -> ManagedProcess:
        try:
            return self._processes[job_id]
        except KeyError as exc:
            raise KeyError(f"Job '{job_id}' not found") from exc

    def tail(self, job_id: str, limit: int = 200) -> List[str]:
        managed = self.get(job_id)
        buffer = list(managed._buffer)
        if len(buffer) >= limit:
            return buffer[-limit:]

        if managed.log_path.exists():
            with managed.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = deque(handle, maxlen=limit)
            return list(lines)
        return buffer

    def status(self, job_id: str) -> Dict[str, Optional[str]]:
        managed = self.get(job_id)
        return {
            "job_id": managed.job_id,
            "status": managed.status(),
            "returncode": None if managed.returncode() is None else str(managed.returncode()),
            "command": managed.command,
            "log_path": str(managed.log_path),
        }

    def known_jobs(self) -> List[str]:
        return list(self._processes.keys())


__all__ = ["ManagedProcess", "ProcessManager"]
