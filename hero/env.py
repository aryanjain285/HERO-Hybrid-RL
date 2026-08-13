"""Load credentials from a dotenv file into the environment.

Secrets stay in the caller's file and are never returned, logged, or written to a
manifest. Only variable names are ever reported.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env_file"]


def load_env_file(path: str | Path, *, override: bool = False) -> tuple[str, ...]:
    """Set variables from a ``KEY=value`` file.

    Args:
        path: Dotenv file. ``export`` prefixes and surrounding quotes are handled.
        override: Replace variables already present in the environment.

    Returns:
        The names of the variables set, never their values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"env file not found: {file}")

    applied: list[str] = []
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not name or (not override and name in os.environ):
            continue
        os.environ[name] = value
        applied.append(name)
    return tuple(applied)
