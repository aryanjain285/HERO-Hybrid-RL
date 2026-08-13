"""Benchmark loading for the verifier study and evaluations.

MATH-500 stands in for the paper's HardVerify-Math, which is not published as a
standalone dataset. Level-5 problems are the closest available proxy for its
difficulty profile, and are also where answer formats are most irregular -- the
condition that produces verifier false negatives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Problem", "load_math500"]

MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_FILE = "test.jsonl"


@dataclass(frozen=True)
class Problem:
    """One benchmark item.

    Args:
        uid: Stable identifier, used to key rollout groups.
        question: Problem statement.
        answer: Reference final answer.
        level: Difficulty, 1-5 for MATH-500.
        subject: Topic label.
    """

    uid: str
    question: str
    answer: str
    level: int
    subject: str


def load_math500(
    *,
    limit: int | None = None,
    min_level: int = 1,
    cache_dir: str | Path | None = None,
) -> tuple[Problem, ...]:
    """Load MATH-500, optionally filtered by difficulty.

    Args:
        limit: Keep at most this many problems, after filtering.
        min_level: Discard problems below this level.
        cache_dir: HuggingFace cache location; defaults to the standard one.

    Returns:
        Problems in dataset order, so a given (limit, min_level) is reproducible.

    Raises:
        ValueError: On an invalid level bound.
        RuntimeError: If the dataset cannot be fetched.
    """
    if not 1 <= min_level <= 5:
        raise ValueError(f"min_level must lie in 1..5; got {min_level}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive; got {limit}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to load MATH-500") from exc

    try:
        path = hf_hub_download(
            MATH500_REPO,
            MATH500_FILE,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        raise RuntimeError(f"could not fetch {MATH500_REPO}: {exc}") from exc

    problems = []
    with open(path, encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            level = int(row.get("level", 0))
            if level < min_level:
                continue
            problems.append(
                Problem(
                    uid=str(row["unique_id"]),
                    question=str(row["problem"]),
                    answer=str(row["answer"]),
                    level=level,
                    subject=str(row.get("subject", "")),
                )
            )
            if limit is not None and len(problems) >= limit:
                break
    if not problems:
        raise RuntimeError(f"no problems matched min_level={min_level}")
    return tuple(problems)
