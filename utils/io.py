"""Utility helpers for input/output operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create the directory *path* if it does not exist and return it."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(obj: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    """Serialize *obj* to *path* with UTF-8 encoding."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load a JSON file from *path* and return the decoded dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
