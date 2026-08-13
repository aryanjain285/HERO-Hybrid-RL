"""Rule-based math answer verifiers.

Supplies the training reward signal, the data filter, and easy-to-verify
evaluation scoring. Three implementations span the precision/recall trade-off the
paper measures in Table 1:

* :class:`ExactMatchVerifier` -- strict normalised equality (high precision).
* :class:`NormalisedMatchVerifier` -- LaTeX cleanup, numeric tolerance, unordered
  set comparison. The project's training-time checker (D-09).
* :class:`SymbolicVerifier` -- adds SymPy equivalence for algebraically equal but
  textually different answers.

These occupy the same roles as the paper's three rule-based checkers but are not
reimplementations of them, so recall figures are not directly comparable; all
three here extract from ``\\boxed{}`` and normalise, which the paper's
``math_reward.py`` does not.

Verification never raises: parse failures and timeouts are :class:`Verdict`
values, so a checker cannot take down a training run.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

__all__ = [
    "ExactMatchVerifier",
    "NormalisedMatchVerifier",
    "SymbolicVerifier",
    "Verdict",
    "VerificationResult",
    "extract_answers",
    "normalise",
]


class Verdict(StrEnum):
    """Outcome of a verification attempt."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    NO_ANSWER = "no_answer"
    """Nothing extractable: a formatting failure rather than a wrong answer."""
    ERROR = "error"
    """Checker failure or timeout, kept distinct so verifier breakage stays visible."""

    @property
    def is_pass(self) -> bool:
        """Whether this scores a reward of 1. Only CORRECT does (D-04)."""
        return self is Verdict.CORRECT


@dataclass(frozen=True)
class VerificationResult:
    """A verdict with the context needed to audit it.

    Attributes:
        verdict: The outcome.
        strategy: Which comparison decided it, for failure taxonomies.
        extracted: Candidate answers found in the response.
        reference: The ground-truth answer.
        detail: Optional discriminator, e.g. ``subset`` or a timeout bound.
    """

    verdict: Verdict
    strategy: str = ""
    extracted: tuple[str, ...] = ()
    reference: tuple[str, ...] = ()
    detail: str = ""

    @property
    def is_pass(self) -> bool:
        return self.verdict.is_pass


_BOXED = re.compile(r"\\boxed\s*\{")
_FINAL_ANSWER = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _balanced_braces(text: str, start: int) -> str | None:
    """Return the contents of the brace group opening at ``start``.

    Brace matching rather than a regex, because nested LaTeX groups such as
    ``\\boxed{\\frac{1}{2}}`` truncate under a non-greedy pattern.
    """
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def extract_answers(response: str) -> tuple[str, ...]:
    """Extract candidate answers, most reliable cue first.

    Returns every ``\\boxed{}`` span when any are present, since answers are often
    split across several boxes.
    """
    if not response or not response.strip():
        return ()

    boxed = []
    for match in _BOXED.finditer(response):
        content = _balanced_braces(response, match.end() - 1)
        if content and content.strip():
            boxed.append(content.strip())
    if boxed:
        return tuple(boxed)

    if phrase := _FINAL_ANSWER.search(response):
        if candidate := phrase.group(1).strip().rstrip("."):
            return (candidate,)

    numbers = _NUMBER.findall(response)
    return (numbers[-1],) if numbers else ()


_STRIP_WRAPPERS = (
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\ ", " "),
    (r"\$", ""),
    (r"\\text\s*\{([^}]*)\}", r"\1"),
    (r"\\mathrm\s*\{([^}]*)\}", r"\1"),
    (r"\\mbox\s*\{([^}]*)\}", r"\1"),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
)
_UNITS = re.compile(
    r"\b(?:cm|mm|km|kg|g|m|s|ms|hours?|hrs?|minutes?|mins?|seconds?|secs?|"
    r"degrees?|units?|dollars?|percent)\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    """Canonicalise an answer for comparison.

    Strips LaTeX presentation markup, units, and answer decorations -- the
    differences the paper attributes verifier false negatives to.
    """
    s = text.strip()
    for pattern, replacement in _STRIP_WRAPPERS:
        s = re.sub(pattern, replacement, s)
    for _ in range(4):  # nested fractions
        new = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = _UNITS.sub("", s)
    s = re.sub(r"^[a-zA-Z]\s*(?:\(\s*[a-zA-Z]\s*\))?\s*=\s*", "", s)  # drop "x =" prefix
    s = s.replace("\\%", "").replace("%", "")  # escaped form first, or a "\" survives
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    return re.sub(r"\s+", "", s).rstrip(".").lower()


def _as_number(text: str) -> float | None:
    """Parse a normalised scalar, tolerating fractions."""
    s = text.replace("(", "").replace(")", "")
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(s)
    except ValueError:
        return None


def _split_set(text: str) -> tuple[str, ...]:
    """Split on top-level commas, so ``(6,3),(9,3)`` yields two tuples not four ints."""
    depth = 0
    parts: list[str] = []
    current: list[str] = []
    for ch in text.strip().strip("{}"):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return tuple(p for p in (part.strip() for part in parts) if p)


class ExactMatchVerifier:
    """Strict normalised equality. Precision ceiling, recall floor of the study."""

    name = "exact_match"

    def verify(self, response: str, reference: str) -> VerificationResult:
        extracted = extract_answers(response)
        if not extracted:
            return VerificationResult(Verdict.NO_ANSWER, "no_extraction")
        ref = normalise(reference)
        verdict = (
            Verdict.CORRECT
            if any(normalise(c) == ref for c in extracted)
            else Verdict.INCORRECT
        )
        return VerificationResult(verdict, "exact", extracted, (reference,))


class NormalisedMatchVerifier:
    """Normalised equality with numeric tolerance and unordered set comparison.

    Accepts an answer whose extracted components match the reference components as
    sets, covering multi-box answers and reordered tuple lists.
    """

    name = "normalised_match"

    def __init__(self, rel_tol: float = 1e-6) -> None:
        if rel_tol < 0:
            raise ValueError(f"rel_tol must be non-negative; got {rel_tol}")
        self.rel_tol = rel_tol

    def _scalar_equal(self, a: str, b: str) -> bool:
        if a == b:
            return True
        x, y = _as_number(a), _as_number(b)
        if x is None or y is None:
            return False
        return abs(x - y) <= self.rel_tol * max(1.0, abs(x), abs(y))

    def _components(self, values: tuple[str, ...]) -> set[str]:
        return {part for value in values for part in _split_set(normalise(value))}

    def _matches_with_tolerance(self, got: set[str], want: set[str]) -> bool:
        if not want or len(want) != len(got):
            return False
        remaining = list(want)
        for item in got:
            match = next((w for w in remaining if self._scalar_equal(item, w)), None)
            if match is None:
                return False
            remaining.remove(match)
        return True

    def verify(self, response: str, reference: str) -> VerificationResult:
        extracted = extract_answers(response)
        if not extracted:
            return VerificationResult(Verdict.NO_ANSWER, "no_extraction")

        ref_norm = normalise(reference)
        if any(self._scalar_equal(normalise(c), ref_norm) for c in extracted):
            return VerificationResult(Verdict.CORRECT, "scalar", extracted, (reference,))

        got = self._components(extracted)
        want = self._components((reference,))
        if got == want:
            return VerificationResult(Verdict.CORRECT, "set_equal", extracted, (reference,))
        if self._matches_with_tolerance(got, want):
            return VerificationResult(
                Verdict.CORRECT, "set_equal_tol", extracted, (reference,)
            )

        detail = "subset" if got < want else ("superset" if got > want else "mismatch")
        return VerificationResult(
            Verdict.INCORRECT, "set_compare", extracted, (reference,), detail
        )


def _sympy_equivalent(a: str, b: str) -> bool:
    """Subprocess entry point: True when two expressions are symbolically equal."""
    from sympy import simplify
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application,)
    left = parse_expr(a, transformations=transformations)
    right = parse_expr(b, transformations=transformations)
    return bool(simplify(left - right) == 0)


class SymbolicVerifier:
    """Normalised matching, then SymPy equivalence for the remainder.

    SymPy runs in a subprocess under a timeout because simplification can hang
    uninterruptibly, and a thread could not be cancelled. A timeout yields
    :attr:`Verdict.ERROR`, scored 0 upstream but counted separately (D-04).
    """

    name = "symbolic"

    def __init__(self, timeout_s: float = 5.0, rel_tol: float = 1e-6) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive; got {timeout_s}")
        self.timeout_s = timeout_s
        self._fallback = NormalisedMatchVerifier(rel_tol=rel_tol)
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def close(self) -> None:
        """Release the subprocess. Safe to call repeatedly."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> SymbolicVerifier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def verify(self, response: str, reference: str) -> VerificationResult:
        base = self._fallback.verify(response, reference)
        if base.verdict is not Verdict.INCORRECT:
            return base

        ref = normalise(reference)
        for candidate in base.extracted:
            try:
                future = self._executor().submit(_sympy_equivalent, normalise(candidate), ref)
                if future.result(timeout=self.timeout_s):
                    return VerificationResult(
                        Verdict.CORRECT, "sympy", base.extracted, (reference,)
                    )
            except FutureTimeout:
                self.close()  # the worker is wedged; discard it
                return VerificationResult(
                    Verdict.ERROR,
                    "sympy_timeout",
                    base.extracted,
                    (reference,),
                    f"exceeded {self.timeout_s}s",
                )
            except Exception:
                continue  # unparseable candidate is ordinary
        return VerificationResult(
            Verdict.INCORRECT, "sympy", base.extracted, (reference,), base.detail
        )
