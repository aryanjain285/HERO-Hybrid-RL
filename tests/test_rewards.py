"""Tests for the reward-arm dispatcher.

Each arm is pinned by the behaviour that makes it a meaningful experimental
control, not merely by its output shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from hero.core import HeroConfig, grpo_advantage
from hero.rewards import RewardArm, RewardArmConfig, compute_group_reward

MIXED = (np.array([1, 1, 0, 0, 0, 0, 0, 0]), np.array([7.0, 5.5, 3.0, 2.0, 1.0, 4.0, 0.5, 2.5]))
ALL_WRONG = (np.zeros(8, dtype=int), np.array([3.0, 1.0, 2.0, 5.0, 4.0, 0.5, 2.5, 1.5]))


def arm(name: RewardArm, **kw) -> RewardArmConfig:
    return RewardArmConfig(arm=name, **kw)


class TestVerifierOnly:
    def test_reward_is_the_binary_label(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.VERIFIER_ONLY), None)
        np.testing.assert_array_equal(out.r_final, MIXED[0].astype(float))

    def test_uniform_group_has_zero_gradient(self):
        """The pathology HERO exists to fix, asserted rather than assumed."""
        out = compute_group_reward(*ALL_WRONG, arm(RewardArm.VERIFIER_ONLY), None)
        assert np.allclose(grpo_advantage(out.r_final), 0.0)

    def test_does_not_require_a_reward_model(self):
        assert not arm(RewardArm.VERIFIER_ONLY).needs_reward_model


class TestRmOnly:
    def test_reward_is_the_raw_score(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.RM_ONLY), None)
        np.testing.assert_array_equal(out.r_final, MIXED[1])

    def test_can_rank_a_wrong_answer_above_a_right_one(self):
        """Why RM-only drifts: correctness is not enforced at all."""
        r_rule = np.array([1, 0])
        r_rm = np.array([1.0, 9.0])
        out = compute_group_reward(r_rule, r_rm, arm(RewardArm.RM_ONLY), None)
        assert out.r_final[1] > out.r_final[0]

    def test_does_not_require_a_verifier(self):
        assert not arm(RewardArm.RM_ONLY).needs_verifier


class TestHeroArms:
    def test_uniform_group_gains_gradient(self):
        """Gradient revival: the mechanism's central claim."""
        out = compute_group_reward(*ALL_WRONG, arm(RewardArm.HERO_NO_WEIGHT), None)
        assert np.abs(grpo_advantage(out.r_final)).max() > 0.5

    def test_weighting_requires_sigma_bar(self):
        unweighted = compute_group_reward(*MIXED, arm(RewardArm.HERO), None)
        weighted = compute_group_reward(*MIXED, arm(RewardArm.HERO), 1.0)
        assert unweighted.weight == 1.0
        assert weighted.weight != 1.0

    def test_no_weight_arm_ignores_sigma_bar(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.HERO_NO_WEIGHT), 1.0)
        assert out.weight == 1.0

    def test_ordering_preserved(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.HERO), 1.0)
        assert out.r_final[out.r_rule == 1].min() > out.r_final[out.r_rule == 0].max()


class TestNaiveBlend:
    @pytest.mark.parametrize("mix", [0.1, 0.5, 0.9])
    def test_stays_in_unit_interval(self, mix):
        out = compute_group_reward(*MIXED, arm(RewardArm.NAIVE_BLEND, blend_mix=mix), None)
        assert (out.r_final >= 0.0).all() and (out.r_final <= 1.0).all()

    def test_can_invert_correctness_which_is_the_point(self):
        """Table 9's failure mode: a high-RM wrong answer outranks a low-RM right one.

        This is exactly what stratification prevents, so the arm earns its place
        in the comparison table as a control.
        """
        r_rule = np.array([1, 0])
        r_rm = np.array([0.0, 10.0])
        out = compute_group_reward(r_rule, r_rm, arm(RewardArm.NAIVE_BLEND, blend_mix=0.1), None)
        assert out.r_final[1] > out.r_final[0]

    def test_pure_rule_limit(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.NAIVE_BLEND, blend_mix=1.0), None)
        np.testing.assert_allclose(out.r_final, MIXED[0].astype(float))

    def test_mix_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="blend_mix"):
            RewardArmConfig(blend_mix=1.5)


class TestGatedFallback:
    def test_mixed_group_is_untouched_binary(self):
        out = compute_group_reward(*MIXED, arm(RewardArm.GATED_FALLBACK), None)
        np.testing.assert_array_equal(out.r_final, MIXED[0].astype(float))

    def test_uniform_group_falls_back_to_rm_ranking(self):
        out = compute_group_reward(*ALL_WRONG, arm(RewardArm.GATED_FALLBACK), None)
        assert np.abs(grpo_advantage(out.r_final)).max() > 0.5

    def test_differs_from_hero_only_on_mixed_groups(self):
        """The A-4 comparison is well posed: the arms agree in uniform groups."""
        gated_u = compute_group_reward(*ALL_WRONG, arm(RewardArm.GATED_FALLBACK), None)
        hero_u = compute_group_reward(*ALL_WRONG, arm(RewardArm.HERO_NO_WEIGHT), None)
        np.testing.assert_allclose(gated_u.r_final, hero_u.r_final)

        gated_m = compute_group_reward(*MIXED, arm(RewardArm.GATED_FALLBACK), None)
        hero_m = compute_group_reward(*MIXED, arm(RewardArm.HERO_NO_WEIGHT), None)
        assert not np.allclose(gated_m.r_final, hero_m.r_final)


class TestSharedContract:
    @pytest.mark.parametrize("name", list(RewardArm))
    def test_every_arm_returns_well_formed_telemetry(self, name):
        out = compute_group_reward(*MIXED, arm(name), 1.0)
        assert out.r_final.shape == MIXED[0].shape
        assert np.isfinite(out.r_final).all()
        assert out.n_correct == 2
        assert out.is_uniform is False
        assert np.isfinite(out.sigma_u)

    @pytest.mark.parametrize("name", list(RewardArm))
    def test_uniform_flag_is_set_for_uniform_groups(self, name):
        assert compute_group_reward(*ALL_WRONG, arm(name), 1.0).is_uniform

    @pytest.mark.parametrize(
        "r_rule, r_rm, match",
        [
            (np.array([0, 1]), np.array([1.0]), "shape mismatch"),
            (np.array([], dtype=int), np.array([]), "non-empty"),
            (np.array([0, 3]), np.array([1.0, 2.0]), "only 0/1"),
            (np.array([0, 1]), np.array([1.0, np.inf]), "non-finite"),
        ],
    )
    def test_malformed_input_rejected_by_every_arm(self, r_rule, r_rm, match):
        for name in RewardArm:
            with pytest.raises(ValueError, match=match):
                compute_group_reward(r_rule, r_rm, arm(name), 1.0)

    def test_band_config_flows_through(self):
        wide = RewardArmConfig(
            arm=RewardArm.HERO_NO_WEIGHT, hero=HeroConfig(alpha=0.2, beta=0.2)
        )
        out = compute_group_reward(*ALL_WRONG, wide, None)
        assert out.r_final.min() == pytest.approx(-0.2)
        assert out.r_final.max() == pytest.approx(0.2, rel=1e-4)
