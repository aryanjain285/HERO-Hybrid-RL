"""Verifier study: the paper's Table 1 methodology, executed locally.

Generates responses to real problems, scores each with every rule verifier, and
labels true correctness with an LLM judge using the paper's template. Verifier
recall, precision, false-positive rate, and accuracy are then measured against
those labels, which is how Table 1 is constructed.

The judge is the label source, so its own reliability bounds every number here.
Labels are retained per response so a manual audit can re-adjudicate them (A-5).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from hero.data import Problem
from hero.llm import OllamaClient
from hero.verifiers import Verdict, VerificationResult

__all__ = [
    "ResponseRecord",
    "StudyResult",
    "VerifierMetrics",
    "Verifier",
    "generate_responses",
    "label_responses",
    "score_verifiers",
]

SOLVE_PROMPT = (
    "Solve the following problem. Reason step by step, then give the final "
    "answer inside \\boxed{{}}.\n\nProblem: {question}"
)


class Verifier(Protocol):
    """Structural type for rule-based verifiers."""

    name: str

    def verify(self, response: str, reference: str) -> VerificationResult: ...


@dataclass
class ResponseRecord:
    """One generated response with its verdicts and label."""

    uid: str
    question: str
    reference: str
    level: int
    sample_index: int
    response: str
    tokens: int
    verdicts: dict[str, str] = field(default_factory=dict)
    """Verifier name -> verdict value."""
    judge_label: bool | None = None
    judge_raw: str = ""

    @property
    def is_labelled(self) -> bool:
        return self.judge_label is not None


@dataclass(frozen=True)
class VerifierMetrics:
    """Table 1 columns for one verifier, computed against judge labels."""

    name: str
    recall: float
    precision: float
    false_positive_rate: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    errors: int
    """Checker failures or timeouts, excluded from the confusion matrix."""

    @property
    def support(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )


@dataclass
class StudyResult:
    """Complete study output."""

    records: list[ResponseRecord]
    metrics: list[VerifierMetrics]
    base_rate: float
    """Fraction of labelled responses the judge called correct."""
    abstentions: int

    def to_json(self, path: str | Path) -> None:
        """Persist everything needed to recompute metrics or audit labels."""
        payload = {
            "base_rate": self.base_rate,
            "abstentions": self.abstentions,
            "metrics": [asdict(m) for m in self.metrics],
            "records": [asdict(r) for r in self.records],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_responses(
    client: OllamaClient,
    problems: tuple[Problem, ...],
    *,
    samples_per_problem: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 768,
    progress: bool = True,
) -> list[ResponseRecord]:
    """Sample responses for each problem.

    Temperature defaults above zero so that repeated samples differ in format,
    which is what exercises verifier recall. Seeds are derived from the sample
    index, making a run reproducible.
    """
    if samples_per_problem < 1:
        raise ValueError(f"samples_per_problem must be positive; got {samples_per_problem}")

    records: list[ResponseRecord] = []
    total = len(problems) * samples_per_problem
    for i, problem in enumerate(problems):
        for k in range(samples_per_problem):
            generation = client.generate(
                SOLVE_PROMPT.format(question=problem.question),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=k,
            )
            records.append(
                ResponseRecord(
                    uid=problem.uid,
                    question=problem.question,
                    reference=problem.answer,
                    level=problem.level,
                    sample_index=k,
                    response=generation.text,
                    tokens=generation.tokens,
                )
            )
            if progress:
                done = i * samples_per_problem + k + 1
                print(
                    f"  generated {done}/{total} "
                    f"({generation.tokens} tok, {generation.tokens_per_s:.0f} tok/s)",
                    flush=True,
                )
    return records


def apply_verifiers(
    records: list[ResponseRecord], verifiers: tuple[Verifier, ...]
) -> None:
    """Attach every verifier's verdict to each record, in place."""
    for record in records:
        for verifier in verifiers:
            result = verifier.verify(record.response, record.reference)
            record.verdicts[verifier.name] = str(result.verdict)


def label_responses(
    client: OllamaClient, records: list[ResponseRecord], *, progress: bool = True
) -> int:
    """Label true correctness with the judge. Returns the abstention count."""
    abstentions = 0
    for i, record in enumerate(records, 1):
        verdict = client.judge_equivalence(
            record.question, record.reference, record.response
        )
        record.judge_label = verdict.equivalent
        record.judge_raw = verdict.raw
        if verdict.abstained:
            abstentions += 1
        if progress:
            label = "abstain" if verdict.abstained else str(verdict.equivalent)
            print(f"  judged {i}/{len(records)} -> {label}", flush=True)
    return abstentions


def score_verifiers(
    records: list[ResponseRecord], verifier_names: tuple[str, ...]
) -> list[VerifierMetrics]:
    """Compute Table 1 metrics for each verifier against the judge labels.

    Unlabelled responses are excluded. Verifier ERROR verdicts are counted
    separately rather than folded into the negatives, so a broken checker is
    distinguishable from a strict one.
    """
    labelled = [r for r in records if r.is_labelled]
    metrics: list[VerifierMetrics] = []

    for name in verifier_names:
        tp = fp = tn = fn = errors = 0
        for record in labelled:
            verdict = record.verdicts.get(name)
            if verdict == Verdict.ERROR:
                errors += 1
                continue
            predicted = verdict == Verdict.CORRECT
            actual = bool(record.judge_label)
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1

        metrics.append(
            VerifierMetrics(
                name=name,
                recall=_ratio(tp, tp + fn),
                precision=_ratio(tp, tp + fp),
                false_positive_rate=_ratio(fp, fp + tn),
                accuracy=_ratio(tp + tn, tp + tn + fp + fn),
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                errors=errors,
            )
        )
    return metrics


def _ratio(numerator: int, denominator: int) -> float:
    """Percentage, or 0.0 when undefined, matching the paper's presentation."""
    return 100.0 * numerator / denominator if denominator else 0.0
