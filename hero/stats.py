"""Confidence intervals for benchmark metrics.

PRD 9.2 requires intervals on every headline number and forbids reporting bare
deltas. Audit A-6 is the reason: HardVerify-Math has 250 items scored at N=1, so a
single-draw binomial standard error near p=0.6 is roughly 3 points -- comparable
to several of the paper's reported gaps.

Two estimators, chosen by what is being measured:

* :func:`wilson_interval` for a single proportion. Preferred over the normal
  approximation because it stays inside [0, 1] and remains usable at p=0 or p=1,
  exactly where a small verifier study lands.
* :func:`bootstrap_interval` for anything else, including paired differences
  between two methods on the same items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Interval",
    "bootstrap_interval",
    "paired_difference_interval",
    "wilson_interval",
]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval, all on a 0-100 scale."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely one side of zero.

        The test for whether a difference is distinguishable from noise.
        """
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:.1f} [{self.low:.1f}, {self.high:.1f}]"


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> Interval:
    """Wilson score interval for a proportion, as a percentage.

    Args:
        successes: Count of successes.
        total: Number of trials.
        confidence: Two-sided confidence level.

    Returns:
        The interval, or a degenerate ``[0, 100]`` when ``total`` is 0.

    Raises:
        ValueError: If counts are negative or ``successes`` exceeds ``total``.
    """
    if successes < 0 or total < 0:
        raise ValueError(f"counts must be non-negative; got {successes}/{total}")
    if successes > total:
        raise ValueError(f"successes {successes} exceeds total {total}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")
    if total == 0:
        return Interval(0.0, 0.0, 100.0, confidence)

    z = _normal_quantile(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return Interval(
        100.0 * p,
        100.0 * max(0.0, centre - spread),
        100.0 * min(1.0, centre + spread),
        confidence,
    )


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for any statistic of a sample.

    Args:
        values: Per-item observations. Resampling is over items, which is the unit
            of independence in a benchmark.
        statistic: Applied to each resample.
        resamples: Bootstrap replicates.
        confidence: Two-sided confidence level.
        seed: Fixed so a reported interval is reproducible.

    Raises:
        ValueError: On an empty sample or invalid parameters.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1:
        raise ValueError(f"resamples must be positive; got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    replicates = np.array([statistic(data[row]) for row in indices])
    tail = (1.0 - confidence) / 2.0
    return Interval(
        float(statistic(data)),
        float(np.quantile(replicates, tail)),
        float(np.quantile(replicates, 1.0 - tail)),
        confidence,
    )


def paired_difference_interval(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Pairing matters: two methods evaluated on the same items share item difficulty,
    so an unpaired comparison discards that and overstates uncertainty.

    Raises:
        ValueError: If the samples differ in length.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"paired samples must align; got {left.shape} vs {right.shape}")
    return bootstrap_interval(
        left - right, resamples=resamples, confidence=confidence, seed=seed
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, via the inverse error function."""
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    """Inverse error function.

    Uses Giles' rational approximation, then two Newton refinements against
    ``math.erf``, which brings the result to double precision. Avoids a scipy
    dependency for what is a handful of z-values.
    """
    if x <= -1.0 or x >= 1.0:
        raise ValueError(f"erfinv domain is (-1, 1); got {x}")
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        coefficients = (
            2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
            0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941,
        )
    else:
        w = math.sqrt(w) - 3.0
        coefficients = (
            -0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
            0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682,
        )
    result = 0.0
    for c in coefficients:
        result = result * w + c
    result *= x
    for _ in range(2):
        error = math.erf(result) - x
        derivative = 2.0 / math.sqrt(math.pi) * math.exp(-result * result)
        if derivative == 0.0:
            break
        result -= error / derivative
    return result
