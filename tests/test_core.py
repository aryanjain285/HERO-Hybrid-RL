"""Property tests for HERO's reward core.

Each test pins a claim the audit or reproduction plan depends on, so a later
refactor or verl upgrade fails the suite rather than the science.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from hero.core import (
    VERL_ADV_EPSILON,
    GroupOutcome,
    HeroConfig,
    HeroConfigError,
    RunningMeanDispersion,
    grpo_advantage,
    group_dispersion,
    shape_group,
    stratified_normalise,
    variance_weight,
)

EASY = HeroConfig(alpha=0.05, beta=0.05)
HARD = HeroConfig(alpha=0.10, beta=0.10)


def make_group(
    rng: np.random.Generator, n: int, n_correct: int, scale: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic rollout group. ``scale`` sets RM score dispersion."""
    r_rule = np.zeros(n, dtype=int)
    r_rule[:n_correct] = 1
    r_rm = np.where(
        r_rule == 1,
        rng.normal(6.0, scale, n),
        rng.normal(2.0, scale, n),
    )
    return r_rule, r_rm


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260813)


# --------------------------------------------------------------------------- #
# Config preconditions
# --------------------------------------------------------------------------- #
class TestHeroConfig:
    def test_published_settings_are_admissible(self):
        for a in (0.05, 0.1, 0.2):
            cfg = HeroConfig(alpha=a, beta=a)
            assert cfg.incorrect_band[1] < cfg.correct_band[0]

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_alpha_out_of_range_rejected(self, bad):
        with pytest.raises(HeroConfigError, match="alpha"):
            HeroConfig(alpha=bad)

    def test_band_overlap_rejected(self):
        """P1 needs alpha + beta < 1; the paper only says (0, 1] (audit A-15)."""
        with pytest.raises(HeroConfigError, match="P1"):
            HeroConfig(alpha=0.6, beta=0.6)
        with pytest.raises(HeroConfigError, match="P1"):
            HeroConfig(alpha=0.5, beta=0.5)  # exactly touching is still rejected

    def test_weight_bounds_validated(self):
        with pytest.raises(HeroConfigError, match="w_min"):
            HeroConfig(w_min=0.0)
        with pytest.raises(HeroConfigError, match="w_max"):
            HeroConfig(w_min=2.0, w_max=1.0)


# --------------------------------------------------------------------------- #
# Eq. 3 -- stratified normalisation
# --------------------------------------------------------------------------- #
class TestStratifiedNormalise:
    def test_band_containment(self, rng):
        """Every rollout lands inside its label's band. Property P1, half 1."""
        for _ in range(500):
            n = int(rng.integers(2, 17))
            r_rule, r_rm = make_group(rng, n, int(rng.integers(0, n + 1)))
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            inc, cor = EASY.incorrect_band, EASY.correct_band
            assert (r_hat[r_rule == 0] >= inc[0] - 1e-12).all()
            assert (r_hat[r_rule == 0] <= inc[1] + 1e-12).all()
            assert (r_hat[r_rule == 1] >= cor[0] - 1e-12).all()
            assert (r_hat[r_rule == 1] <= cor[1] + 1e-12).all()

    def test_ordering_preservation(self, rng):
        """Property P1: no correct rollout is ever outscored by an incorrect one."""
        for _ in range(500):
            n = int(rng.integers(2, 17))
            n_correct = int(rng.integers(1, n))  # force a mixed group
            r_rule, r_rm = make_group(rng, n, n_correct)
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            assert r_hat[r_rule == 1].min() > r_hat[r_rule == 0].max()

    def test_rank_preservation_within_band(self, rng):
        """Within a band, r_hat is monotone in the RM score."""
        for _ in range(200):
            n = 8
            r_rule, r_rm = make_group(rng, n, 0)
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            assert np.array_equal(np.argsort(r_rm), np.argsort(r_hat))

    def test_affine_invariance_of_rm_scores(self, rng):
        """RM location and scale do not reach the policy; only ranking does.

        So RM calibration cannot affect ``r_hat``, and any calibration benefit
        must route through ``sigma_u`` -- consistent with the 72B RM barely
        moving the paper's numbers (Table 8).

        Invariance is exact only as epsilon -> 0, since ``(hi - lo)`` scales with
        the RM gain but epsilon does not. The per-group bound is asserted rather
        than a blanket tolerance, because a two-member band can have an
        arbitrarily small range and hence an arbitrarily large residual.
        """
        for _ in range(300):
            r_rule, r_rm = make_group(rng, 8, int(rng.integers(0, 9)))
            base = stratified_normalise(r_rule, r_rm, EASY)
            for gain, offset in ((3.0, 0.0), (0.1, 0.0), (1.0, 100.0), (7.5, -4.2)):
                moved = stratified_normalise(r_rule, gain * r_rm + offset, EASY)
                bound = self._epsilon_affine_bound(r_rule, r_rm, gain, EASY)
                assert np.abs(moved - base).max() <= bound + 1e-12

    @staticmethod
    def _epsilon_affine_bound(
        r_rule: np.ndarray, r_rm: np.ndarray, gain: float, cfg: HeroConfig
    ) -> float:
        """Largest r_hat deviation the min-max epsilon can produce under scaling."""
        eps = cfg.minmax_epsilon
        bound = 0.0
        for label, span in ((0, 2 * cfg.alpha), (1, 2 * cfg.beta)):
            m = r_rule == label
            if m.sum() < 2:
                continue
            R = float(r_rm[m].max() - r_rm[m].min())
            if R == 0.0:
                continue
            shrink_base = 1.0 / (1.0 + eps / R)
            shrink_moved = 1.0 / (1.0 + eps / (gain * R))
            bound = max(bound, span * abs(shrink_moved - shrink_base))
        return bound

    def test_epsilon_breaks_affine_invariance_by_a_known_bound(self):
        """Pin the residual: |dz| <= epsilon / (hi - lo), so |dr_hat| <= 2*alpha*that.

        Stated as a test so that anyone tempted to raise ``minmax_epsilon`` for
        'stability' sees immediately what it costs. At epsilon = 1e-2 with a
        narrow-range RM the band structure is measurably distorted.
        """
        r_rule = np.zeros(4, dtype=int)
        r_rm = np.array([0.0, 0.25, 0.5, 1.0])  # range exactly 1.0
        for eps in (1e-6, 1e-3, 1e-2):
            cfg = HeroConfig(alpha=0.05, beta=0.05, minmax_epsilon=eps)
            r_hat = stratified_normalise(r_rule, r_rm, cfg)
            span = r_hat.max() - r_hat.min()
            ideal = 2 * cfg.alpha
            rm_range = float(r_rm.max() - r_rm.min())
            # Realised band span falls short of 2*alpha by exactly the
            # epsilon-induced shrinkage factor range/(range + eps).
            assert span == pytest.approx(ideal * rm_range / (rm_range + eps), rel=1e-12)
            assert ideal - span <= ideal * eps / rm_range + 1e-15

    def test_singleton_pinned_to_band_floor(self):
        """Eq. 3 as written: max == min gives z = 0, i.e. the band floor (D-05)."""
        r_hat = stratified_normalise(np.array([1, 0, 0]), np.array([9.0, 1.0, 2.0]), EASY)
        assert r_hat[0] == pytest.approx(1.0 - EASY.beta)  # lone correct -> floor

    def test_singleton_midpoint_alternative(self):
        """singleton_z=0.5 places a lone rollout at the band centre instead."""
        cfg = HeroConfig(alpha=0.05, beta=0.05, singleton_z=0.5)
        r_hat = stratified_normalise(np.array([1, 0, 0]), np.array([9.0, 1.0, 2.0]), cfg)
        assert r_hat[0] == pytest.approx(1.0)

    def test_all_tied_rm_scores_give_no_intra_band_signal(self):
        """Tied RM scores collapse the band: no gradient is manufactured."""
        r_hat = stratified_normalise(np.zeros(8, dtype=int), np.full(8, 3.3), EASY)
        assert np.allclose(r_hat, -EASY.alpha)
        assert grpo_advantage(r_hat) == pytest.approx(np.zeros(8), abs=1e-12)

    def test_uniform_group_always_spans_the_full_band(self, rng):
        """min-max forces the band to be fully occupied -- the root of A-1b.

        The realised spread of a uniform group is 2*alpha regardless of how
        weakly the RM discriminates, which is exactly why group standardisation
        can divide alpha out (see test_alpha_is_inert_in_uniform_groups).
        """
        for _ in range(200):
            r_rule, r_rm = make_group(rng, 8, 0)
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            assert r_hat.max() - r_hat.min() == pytest.approx(2 * EASY.alpha, rel=1e-4)

    @pytest.mark.parametrize(
        "r_rule, r_rm, match",
        [
            (np.array([0, 1]), np.array([1.0]), "shape mismatch"),
            (np.array([], dtype=int), np.array([]), "non-empty"),
            (np.array([0, 2]), np.array([1.0, 2.0]), "only 0/1"),
            (np.array([0, 1]), np.array([1.0, np.nan]), "non-finite"),
        ],
    )
    def test_malformed_input_rejected(self, r_rule, r_rm, match):
        with pytest.raises(ValueError, match=match):
            stratified_normalise(r_rule, r_rm, EASY)


# --------------------------------------------------------------------------- #
# Eq. 4 -- variance weighting
# --------------------------------------------------------------------------- #
class TestVarianceWeight:
    def test_bounds_and_monotonicity(self):
        cfg = EASY
        ws = [variance_weight(s, 1.0, cfg) for s in np.linspace(-5, 7, 200)]
        assert all(cfg.w_min <= w <= cfg.w_max for w in ws)
        assert all(b >= a for a, b in zip(ws, ws[1:]))

    def test_midpoint_at_equality(self):
        """sigma_u == sigma_bar sits exactly halfway between the bounds."""
        assert variance_weight(1.0, 1.0, EASY) == pytest.approx(
            (EASY.w_min + EASY.w_max) / 2
        )

    def test_no_overflow_at_extreme_dispersion(self):
        """Saturation is routine, so the logistic must not overflow (audit A-16)."""
        assert variance_weight(1e6, 0.0, EASY) == pytest.approx(EASY.w_max)
        assert variance_weight(-1e6, 0.0, EASY) == pytest.approx(EASY.w_min)

    def test_saturation_with_published_k_and_realistic_dispersion(self):
        """With k=6, a dispersion gap of ~1 raw RM unit already saturates.

        AceMath raw scores run to ~40 during training (paper Fig. 6), so
        per-group dispersion gaps of O(1) are ordinary. The logistic is then a
        hard two-level gate, not a smooth weighting -- so the paper's
        'bounded monotone' framing understates how discrete the mechanism is.
        """
        span = EASY.w_max - EASY.w_min
        # One raw unit above the running mean already reaches 99.75% of w_max.
        assert variance_weight(1.0, 0.0, EASY) > EASY.w_min + 0.997 * span
        assert variance_weight(-1.0, 0.0, EASY) < EASY.w_min + 0.003 * span

    def test_dispersion_source_switch(self, rng):
        """D-02: raw-score dispersion is O(RM scale); normalised is O(alpha)."""
        r_rule, r_rm = make_group(rng, 8, 4)
        r_hat = stratified_normalise(r_rule, r_rm, EASY)
        raw = group_dispersion(r_rm, r_hat, HeroConfig(sigma_on_raw_rm=True))
        norm = group_dispersion(r_rm, r_hat, HeroConfig(sigma_on_raw_rm=False))
        assert raw > 1.0
        assert norm < 1.0

    def test_dispersion_of_singleton_group_is_zero(self):
        assert group_dispersion(np.array([1.0]), np.array([0.0]), EASY) == 0.0


# --------------------------------------------------------------------------- #
# sigma_bar bookkeeping
# --------------------------------------------------------------------------- #
class TestRunningMeanDispersion:
    def test_undefined_before_warmup(self):
        ema = RunningMeanDispersion()
        assert not ema.is_warm
        with pytest.raises(RuntimeError, match="undefined"):
            _ = ema.value

    def test_warm_start_is_first_batch_mean(self):
        ema = RunningMeanDispersion(momentum=0.9)
        for s in (1.0, 2.0, 3.0):
            ema.observe(s)
        assert ema.end_batch() == pytest.approx(2.0)

    def test_ema_update(self):
        ema = RunningMeanDispersion(momentum=0.9)
        ema.observe(2.0)
        ema.end_batch()
        ema.observe(12.0)
        assert ema.end_batch() == pytest.approx(0.9 * 2.0 + 0.1 * 12.0)

    def test_reads_are_frozen_within_a_batch(self):
        """Order-independence: no group's weight may depend on visit order.

        A group-by-group update would make rewards depend on how the batch was
        sharded across data-parallel ranks, i.e. unreproducible runs.
        """
        ema = RunningMeanDispersion()
        ema.observe(5.0)
        ema.end_batch()
        frozen = ema.value
        for s in (100.0, 0.01, 42.0):
            ema.observe(s)
            assert ema.value == frozen
        ema.end_batch()
        assert ema.value != frozen

    def test_end_batch_without_observations_is_an_error(self):
        with pytest.raises(RuntimeError, match="no observed groups"):
            RunningMeanDispersion().end_batch()

    def test_permutation_invariance(self, rng):
        a = RunningMeanDispersion()
        b = RunningMeanDispersion()
        vals = rng.normal(3.0, 1.0, 64)
        for v in vals:
            a.observe(v)
        for v in rng.permutation(vals):
            b.observe(v)
        assert a.end_batch() == pytest.approx(b.end_batch())


# --------------------------------------------------------------------------- #
# GRPO advantage fidelity to verl
# --------------------------------------------------------------------------- #
class TestGrpoAdvantage:
    def test_zero_mean(self, rng):
        for _ in range(200):
            a = grpo_advantage(rng.normal(0, 1, 8))
            assert a.sum() == pytest.approx(0.0, abs=1e-9)

    def test_unit_scale_when_normalised(self, rng):
        a = grpo_advantage(rng.normal(0, 5, 64))
        assert a.std(ddof=1) == pytest.approx(1.0, rel=1e-5)

    def test_mean_only_preserves_reward_scale(self, rng):
        r = rng.normal(0, 5, 64)
        a = grpo_advantage(r, norm_by_std=False)
        assert a.std(ddof=1) == pytest.approx(r.std(ddof=1))

    def test_constant_group_yields_no_gradient(self):
        assert np.allclose(grpo_advantage(np.full(8, 0.7)), 0.0)

    def test_verl_singleton_quirk_replicated(self):
        """verl sets mean=0/std=1 for |group|==1, so A == r, not 0."""
        assert grpo_advantage(np.array([0.95])) == pytest.approx(np.array([0.95]))

    def test_epsilon_matches_verl(self):
        assert VERL_ADV_EPSILON == 1e-6


# --------------------------------------------------------------------------- #
# The audit claims themselves -- these are the load-bearing tests
# --------------------------------------------------------------------------- #
class TestAuditA1WeightInvariance:
    """A-1: the variance weight cancels under canonical GRPO."""

    def test_weight_cancels_under_std_normalisation(self, rng):
        worst = 0.0
        for _ in range(2000):
            r_rule, r_rm = make_group(rng, 8, int(rng.integers(0, 9)))
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            for w in (0.4, 1.0, 1.7, 3.0):
                a = grpo_advantage(r_hat, norm_by_std=True)
                b = grpo_advantage(w * r_hat, norm_by_std=True)
                scale = max(np.abs(a).max(), 1e-12)
                worst = max(worst, np.abs(b - a).max() / scale)
        # Pure float noise from the 1e-6 denominator epsilon.
        assert worst < 1e-3, f"weight is not cancelling; worst relative shift {worst}"

    def test_weight_is_live_without_std_normalisation(self, rng):
        r_rule, r_rm = make_group(rng, 8, 3)
        r_hat = stratified_normalise(r_rule, r_rm, EASY)
        a = grpo_advantage(r_hat, norm_by_std=False)
        for w in (0.4, 3.0):
            b = grpo_advantage(w * r_hat, norm_by_std=False)
            np.testing.assert_allclose(b, w * a, rtol=1e-12)

    def test_epsilon_cannot_explain_the_ablation(self):
        """PRD candidate explanation (3) quantified and killed.

        A_i(w)/A_i(1) = w(sigma + eps)/(w*sigma + eps). At realistic group
        dispersion the deviation from 1 is O(1e-5) -- five orders of magnitude
        too small to move a benchmark average by 3.8 points.
        """
        for sigma in (0.005, 0.02, 0.03):
            for w in (0.4, 3.0):
                ratio = w * (sigma + VERL_ADV_EPSILON) / (w * sigma + VERL_ADV_EPSILON)
                assert abs(ratio - 1.0) < 1e-3


class TestAuditA1bAlphaInvariance:
    """A-1b (new): the band width is inert in uniform groups under canonical GRPO."""

    @pytest.mark.parametrize("n_correct", [0, 8])
    def test_alpha_is_inert_in_uniform_groups(self, rng, n_correct):
        worst = 0.0
        for _ in range(1000):
            r_rule, r_rm = make_group(rng, 8, n_correct)
            a = grpo_advantage(stratified_normalise(r_rule, r_rm, EASY), True)
            for cfg in (HARD, HeroConfig(alpha=0.2, beta=0.2)):
                b = grpo_advantage(stratified_normalise(r_rule, r_rm, cfg), True)
                worst = max(worst, np.abs(b - a).max() / max(np.abs(a).max(), 1e-12))
        assert worst < 1e-3, f"alpha should be inert here; worst shift {worst}"

    @pytest.mark.parametrize("n_correct", [0, 8])
    def test_alpha_scales_gradient_linearly_without_std_norm(self, rng, n_correct):
        r_rule, r_rm = make_group(rng, 8, n_correct)
        a = grpo_advantage(stratified_normalise(r_rule, r_rm, EASY), False)
        b = grpo_advantage(stratified_normalise(r_rule, r_rm, HARD), False)
        np.testing.assert_allclose(b, 2.0 * a, rtol=1e-6)

    def test_alpha_is_live_in_mixed_groups_even_with_std_norm(self, rng):
        """The honest counterpart: alpha is NOT globally inert.

        In mixed groups it trades between-band separation against within-band
        ranking, a real effect of order 10% of advantage scale. So the range
        ablation can act -- just not through the uniform groups the paper
        credits for the mixed-regime gain.
        """
        shifts = []
        for _ in range(500):
            r_rule, r_rm = make_group(rng, 8, 4)
            a = grpo_advantage(stratified_normalise(r_rule, r_rm, EASY), True)
            b = grpo_advantage(
                stratified_normalise(r_rule, r_rm, HeroConfig(alpha=0.2, beta=0.2)), True
            )
            shifts.append(np.abs(b - a).max() / np.abs(a).max())
        assert 0.01 < float(np.mean(shifts)) < 1.0


class TestUniformGroupAmplification:
    """A-17 (new): standardisation erases the band's containment of the RM."""

    def test_uniform_groups_reach_mixed_group_gradient_scale(self, rng):
        uni, mixed = [], []
        for _ in range(2000):
            r0, m0 = make_group(rng, 8, 0)
            uni.append(np.abs(grpo_advantage(stratified_normalise(r0, m0, EASY), True)).mean())
            r1, m1 = make_group(rng, 8, 1)
            mixed.append(np.abs(grpo_advantage(stratified_normalise(r1, m1, EASY), True)).mean())
        # An all-incorrect group ranked purely by the RM speaks at least as
        # loudly as a group carrying a genuine verifier signal.
        assert float(np.mean(uni)) > 0.8 * float(np.mean(mixed))

    def test_band_width_restores_the_hierarchy_without_std_norm(self, rng):
        uni, mixed = [], []
        for _ in range(2000):
            r0, m0 = make_group(rng, 8, 0)
            uni.append(np.abs(grpo_advantage(stratified_normalise(r0, m0, EASY), False)).mean())
            r1, m1 = make_group(rng, 8, 4)
            mixed.append(np.abs(grpo_advantage(stratified_normalise(r1, m1, EASY), False)).mean())
        # alpha=0.05 keeps uniform groups roughly an order of magnitude quieter.
        assert float(np.mean(uni)) < 0.2 * float(np.mean(mixed))


# --------------------------------------------------------------------------- #
# End-to-end group shaping
# --------------------------------------------------------------------------- #
class TestShapeGroup:
    def test_end_to_end_fields(self, rng):
        r_rule, r_rm = make_group(rng, 8, 3)
        out = shape_group(r_rule, r_rm, EASY, sigma_bar=2.0)
        assert isinstance(out, GroupOutcome)
        assert out.n_correct == 3
        assert not out.is_uniform
        assert EASY.w_min <= out.weight <= EASY.w_max
        np.testing.assert_allclose(out.r_final, out.weight * out.r_hat)

    def test_weighting_disabled_is_the_ablation_arm(self, rng):
        r_rule, r_rm = make_group(rng, 8, 3)
        out = shape_group(r_rule, r_rm, EASY, sigma_bar=None)
        assert out.weight == 1.0
        np.testing.assert_allclose(out.r_final, out.r_hat)

    def test_uniform_flag(self, rng):
        for n_correct, expected in ((0, True), (8, True), (4, False)):
            r_rule, r_rm = make_group(rng, 8, n_correct)
            assert shape_group(r_rule, r_rm, EASY, 2.0).is_uniform is expected

    def test_singleton_band_counter(self, rng):
        r_rule = np.array([1, 0, 0, 0])
        r_rm = np.array([9.0, 1.0, 2.0, 3.0])
        assert shape_group(r_rule, r_rm, EASY, 2.0).singleton_bands == 1
        # Both bands degenerate: one correct, and all incorrect scores tied.
        assert shape_group(
            np.array([1, 0, 0]), np.array([9.0, 2.0, 2.0]), EASY, 2.0
        ).singleton_bands == 2

    def test_full_pipeline_preserves_ordering_after_weighting(self, rng):
        """Weighting is a positive scalar, so P1 survives Eq. 5."""
        for _ in range(300):
            r_rule, r_rm = make_group(rng, 8, int(rng.integers(1, 8)))
            out = shape_group(r_rule, r_rm, EASY, sigma_bar=2.0)
            assert out.r_final[out.r_rule == 1].min() > out.r_final[out.r_rule == 0].max()
