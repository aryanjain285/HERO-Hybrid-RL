"""Tests for the interval estimators.

The normal-quantile and Wilson implementations are checked against published
values, because a hand-rolled erfinv is exactly the kind of code that silently
returns plausible-but-wrong numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hero.stats import (
    Interval,
    _normal_quantile,
    bootstrap_interval,
    paired_difference_interval,
    wilson_interval,
)


class TestNormalQuantile:
    @pytest.mark.parametrize(
        "p, expected",
        [
            (0.975, 1.959964),
            (0.995, 2.575829),
            (0.95, 1.644854),
            (0.5, 0.0),
            (0.025, -1.959964),
        ],
    )
    def test_against_published_values(self, p, expected):
        assert _normal_quantile(p) == pytest.approx(expected, abs=1e-5)

    def test_round_trips_through_erf(self):
        for p in (0.6, 0.8, 0.9, 0.99, 0.999):
            z = _normal_quantile(p)
            recovered = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            assert recovered == pytest.approx(p, abs=1e-9)


class TestWilsonInterval:
    def test_matches_an_independent_derivation(self):
        """Recomputes the Wilson formula here rather than trusting a copied number.

        For 20/100 at 95% this gives [13.3, 28.9]. The commonly quoted 29.2 upper
        bound belongs to Agresti-Coull, not Wilson, which is worth pinning so the
        two are not conflated later.
        """
        n, k, z = 100, 20, 1.959963985
        p = k / n
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        spread = (
            z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        )

        interval = wilson_interval(k, n)
        assert interval.point == pytest.approx(100 * p)
        assert interval.low == pytest.approx(100 * (centre - spread), abs=1e-6)
        assert interval.high == pytest.approx(100 * (centre + spread), abs=1e-6)
        assert (interval.low, interval.high) == pytest.approx((13.3, 28.9), abs=0.1)

    def test_stays_within_bounds_at_the_extremes(self):
        """The normal approximation would leave [0, 100] here; Wilson must not."""
        perfect = wilson_interval(23, 23)
        assert perfect.point == 100.0
        assert perfect.high == pytest.approx(100.0)
        assert 0.0 < perfect.low < 100.0

        zero = wilson_interval(0, 23)
        assert zero.point == 0.0
        assert zero.low == pytest.approx(0.0)
        assert 0.0 < zero.high < 100.0

    def test_interval_narrows_with_sample_size(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(500, 1000)
        assert large.width < small.width

    def test_a6_scale_check(self):
        """Audit A-6: 250 items at p=0.6 has roughly a 3-point standard error.

        The half-width should therefore be about 6 points at 95%.
        """
        interval = wilson_interval(150, 250)
        assert 10.0 < interval.width < 14.0

    def test_empty_sample_is_maximally_uncertain(self):
        interval = wilson_interval(0, 0)
        assert (interval.low, interval.high) == (0.0, 100.0)

    @pytest.mark.parametrize(
        "successes, total",
        [(-1, 10), (5, -1), (11, 10)],
    )
    def test_invalid_counts_rejected(self, successes, total):
        with pytest.raises(ValueError):
            wilson_interval(successes, total)

    def test_confidence_must_be_a_probability(self):
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(5, 10, confidence=1.5)

    def test_higher_confidence_widens(self):
        assert wilson_interval(50, 100, confidence=0.99).width > wilson_interval(
            50, 100, confidence=0.95
        ).width


class TestBootstrap:
    def test_recovers_the_mean(self):
        values = np.linspace(0.0, 100.0, 200)
        interval = bootstrap_interval(values)
        assert interval.point == pytest.approx(values.mean())
        assert interval.low < interval.point < interval.high

    def test_deterministic_for_a_fixed_seed(self):
        values = [1.0, 5.0, 9.0, 3.0, 7.0]
        first = bootstrap_interval(values, seed=42, resamples=500)
        second = bootstrap_interval(values, seed=42, resamples=500)
        assert (first.low, first.high) == (second.low, second.high)

    def test_zero_variance_sample_gives_a_point_interval(self):
        interval = bootstrap_interval([4.0] * 20)
        assert interval.low == interval.high == pytest.approx(4.0)

    def test_supports_other_statistics(self):
        interval = bootstrap_interval([1.0, 2.0, 3.0, 100.0], statistic=np.median)
        assert 1.0 <= interval.point <= 100.0

    def test_empty_sample_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_interval([])

    def test_invalid_resamples_rejected(self):
        with pytest.raises(ValueError, match="resamples"):
            bootstrap_interval([1.0], resamples=0)


class TestPairedDifference:
    def test_detects_a_consistent_improvement(self):
        """Pairing is what makes a small but consistent gain detectable."""
        a = [0.62, 0.71, 0.55, 0.80, 0.66] * 8
        b = [0.60, 0.69, 0.53, 0.78, 0.64] * 8
        interval = paired_difference_interval(a, b)
        assert interval.point == pytest.approx(0.02, abs=1e-9)
        assert interval.excludes_zero

    def test_no_difference_spans_zero(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.5, 0.1, 100)
        b = a + rng.normal(0.0, 0.1, 100)
        assert not paired_difference_interval(a, b).excludes_zero

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="align"):
            paired_difference_interval([1.0, 2.0], [1.0])


class TestInterval:
    def test_formatting_is_report_ready(self):
        assert str(Interval(60.9, 40.8, 77.8)) == "60.9 [40.8, 77.8]"

    def test_excludes_zero_both_directions(self):
        assert Interval(5.0, 1.0, 9.0).excludes_zero
        assert Interval(-5.0, -9.0, -1.0).excludes_zero
        assert not Interval(0.5, -1.0, 2.0).excludes_zero

    def test_width(self):
        assert Interval(50.0, 40.0, 65.0).width == pytest.approx(25.0)

    def test_non_overlap_is_detectable(self):
        """The claim actually made in the M0 report: 60.9 vs 100.0 recall."""
        raw = Interval(60.9, 40.8, 77.8)
        symbolic = Interval(100.0, 85.7, 100.0)
        assert raw.high < symbolic.low
