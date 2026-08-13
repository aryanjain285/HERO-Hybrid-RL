"""Tests for the end-to-end GRPO trainer.

The trainer is the evidence base for audit A-1's training-level claim, so its
correctness matters as much as the reward core's. These tests check that it is a
real optimiser, that it is deterministic, and that the A-1 result holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from hero.rewards import RewardArm, RewardArmConfig
from hero.toy import ToyTask, ToyTaskConfig, TrainConfig, train

TASK = ToyTask(ToyTaskConfig())
FAST = TrainConfig(steps=40)


def arm(name: RewardArm) -> RewardArmConfig:
    return RewardArmConfig(arm=name)


class TestToyTask:
    def test_task_is_hard_enough_to_have_uniform_groups(self):
        """If most responses were correct there would be no regime to study."""
        assert 0.05 < TASK.base_correct_fraction < 0.25

    def test_task_is_deterministic(self):
        a, b = ToyTask(ToyTaskConfig()), ToyTask(ToyTaskConfig())
        np.testing.assert_array_equal(a.rm_scores, b.rm_scores)
        np.testing.assert_array_equal(a.correct, b.correct)

    def test_policy_is_a_distribution(self):
        probs = TASK.policy(np.array([1.0, -0.5, 0.25, 2.0]))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert (probs > 0).all()

    def test_uniform_policy_accuracy_equals_base_rate(self):
        """theta = 0 gives a uniform policy, so accuracy is the correct fraction."""
        assert TASK.expected_accuracy(np.zeros(4)) == pytest.approx(
            TASK.base_correct_fraction, rel=1e-9
        )

    def test_rm_quality_matches_the_paper_diagnostic(self):
        """Calibrated to the paper's measured mean group AUROC of 0.79 (App. B.1).

        A near-perfect RM would rig the task in favour of dense rewards; a useless
        one would make the hybrid unlearnable. Neither would tell us anything.
        """
        auroc = TASK.rm_group_auroc(np.zeros(4), np.random.default_rng(3), 8)
        assert 0.70 < auroc < 0.90, auroc

    def test_lowering_noise_makes_the_rm_near_perfect(self):
        """Sanity check on the calibration knob's direction."""
        clean = ToyTask(ToyTaskConfig(rm_noise=0.1))
        assert clean.rm_group_auroc(np.zeros(4), np.random.default_rng(3), 8) > 0.97

    def test_skill_direction_is_learnable(self):
        """Aligning theta with the skill coordinate must beat a uniform policy."""
        aligned = np.array([4.0, 0.0, 0.0, 0.0])
        assert TASK.expected_accuracy(aligned) > 4 * TASK.expected_accuracy(np.zeros(4))

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError, match="at least 2 responses"):
            ToyTaskConfig(n_responses=1)
        with pytest.raises(ValueError, match="rm_noise"):
            ToyTaskConfig(rm_noise=-1.0)


class TestTrainConfig:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rollouts_per_prompt": 1}, "at least 2 rollouts"),
            ({"mini_batches": 0}, "at least one gradient step"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"clip_low": 0.5}, "clip_low < clip_high"),
            ({"adv_epsilon": -1e-6}, "adv_epsilon"),
        ],
    )
    def test_invalid_settings_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TrainConfig(**kwargs)


class TestTrainerIsReal:
    """A no-op trainer would make every downstream claim vacuous."""

    @pytest.mark.parametrize("name", list(RewardArm))
    def test_every_arm_improves_accuracy(self, name):
        result = train(TASK, arm(name), FAST)
        assert result.final_accuracy > result.initial_accuracy + 0.1

    def test_theta_moves_toward_the_skill_coordinate(self):
        """Learning the task means loading the skill feature, not a distractor."""
        theta = train(TASK, arm(RewardArm.HERO), FAST).theta
        assert theta[0] > 0
        assert theta[0] > np.abs(theta[1:]).max()

    def test_training_is_deterministic(self):
        a = train(TASK, arm(RewardArm.HERO), FAST).theta
        b = train(TASK, arm(RewardArm.HERO), FAST).theta
        np.testing.assert_array_equal(a, b)

    def test_seeds_produce_different_trajectories(self):
        a = train(TASK, arm(RewardArm.HERO), TrainConfig(steps=40, seed=0)).theta
        b = train(TASK, arm(RewardArm.HERO), TrainConfig(steps=40, seed=1)).theta
        assert not np.array_equal(a, b)

    def test_telemetry_lengths_are_consistent(self):
        r = train(TASK, arm(RewardArm.HERO), FAST)
        # Accuracy is recorded before each step and once after the last.
        assert len(r.accuracy) == FAST.steps + 1
        assert len(r.uniform_group_fraction) == FAST.steps
        assert len(r.mean_abs_advantage) == FAST.steps

    def test_all_incorrect_groups_are_common_early(self):
        """Confirms the run actually exercises HERO's target regime."""
        r = train(TASK, arm(RewardArm.HERO), FAST)
        assert r.all_incorrect_fraction[0] > 0.25

    def test_uniformity_shifts_from_failure_to_success(self):
        """Late-run uniformity is all-correct, so a single metric would mislead."""
        r = train(TASK, arm(RewardArm.HERO), FAST)
        assert r.all_incorrect_fraction[0] > r.all_incorrect_fraction[-1]
        assert r.all_correct_fraction[-1] > r.all_correct_fraction[0]

    def test_uniform_fraction_is_the_sum_of_its_parts(self):
        r = train(TASK, arm(RewardArm.HERO), FAST)
        assert r.uniform_group_fraction[0] == pytest.approx(
            r.all_incorrect_fraction[0] + r.all_correct_fraction[0]
        )

    def test_verifier_only_arm_needs_no_rm_scores(self):
        """End-to-end check that the baseline runs without an RM (see rewards.py)."""
        assert train(TASK, arm(RewardArm.VERIFIER_ONLY), FAST).final_accuracy > 0.2


class TestAuditA1AtTrainingLevel:
    """The load-bearing result: does the weight change what is learned?"""

    def test_weight_is_inert_up_to_float_rounding_when_epsilon_is_zero(self):
        """With the epsilon removed, cancellation is exact and training matches."""
        cfg = TrainConfig(steps=40, norm_adv_by_std=True, adv_epsilon=0.0)
        weighted = train(TASK, arm(RewardArm.HERO), cfg).theta
        plain = train(TASK, arm(RewardArm.HERO_NO_WEIGHT), cfg).theta
        np.testing.assert_allclose(weighted, plain, atol=1e-12)

    def test_epsilon_seeds_a_divergence_that_does_not_change_outcomes(self):
        """verl's 1e-6 breaks exactness, but the effect is noise, not mechanism."""
        cfg = TrainConfig(steps=40, norm_adv_by_std=True)
        weighted = train(TASK, arm(RewardArm.HERO), cfg)
        plain = train(TASK, arm(RewardArm.HERO_NO_WEIGHT), cfg)
        assert not np.array_equal(weighted.theta, plain.theta)
        assert abs(weighted.final_accuracy - plain.final_accuracy) < 0.005

    def test_weight_has_a_consistent_effect_without_std_normalisation(self):
        """Same sign on every seed, two orders of magnitude above the epsilon noise."""
        deltas = []
        for seed in (0, 1, 2):
            cfg = TrainConfig(steps=60, norm_adv_by_std=False, seed=seed)
            weighted = train(TASK, arm(RewardArm.HERO), cfg)
            plain = train(TASK, arm(RewardArm.HERO_NO_WEIGHT), cfg)
            deltas.append(weighted.final_accuracy - plain.final_accuracy)
        assert all(d > 0.002 for d in deltas), deltas

    def test_epsilon_is_irrelevant_without_std_normalisation(self):
        """Nothing divides by it, so zeroing it must change nothing at all."""
        base = TrainConfig(steps=40, norm_adv_by_std=False)
        zeroed = TrainConfig(steps=40, norm_adv_by_std=False, adv_epsilon=0.0)
        np.testing.assert_array_equal(
            train(TASK, arm(RewardArm.HERO), base).theta,
            train(TASK, arm(RewardArm.HERO), zeroed).theta,
        )

    def test_weighting_actually_fires_during_training(self):
        """Guards against a vacuous pass: the weight must vary, not sit at 1.0."""
        r = train(TASK, arm(RewardArm.HERO), FAST)
        assert len(set(r.weights_seen)) > 1
        assert not np.allclose(r.weights_seen, 1.0)


class TestAuditA12AurocScope:
    def test_auroc_is_undefined_without_mixed_groups(self):
        """A policy that is always wrong admits no reliability measurement."""
        hopeless = ToyTask(ToyTaskConfig(skill_threshold=100.0))
        assert not hopeless.correct.any()
        auroc = hopeless.rm_group_auroc(np.zeros(4), np.random.default_rng(0), 8)
        assert np.isnan(auroc)

    def test_mixed_groups_become_scarce_as_the_policy_sharpens(self):
        """The diagnostic's domain shrinks exactly where the mechanism matters."""
        weak = TASK.rm_group_auroc(np.zeros(4), np.random.default_rng(5), 8)
        assert not np.isnan(weak)
        strong_theta = np.array([12.0, 0.0, 0.0, 0.0])
        probs = TASK.policy(strong_theta)
        mixed = sum(
            1
            for x in range(TASK.cfg.n_prompts)
            if 0 < (TASK.correct[x] * (probs[x] > 1e-3)).sum() < (probs[x] > 1e-3).sum()
        )
        assert mixed < TASK.cfg.n_prompts // 2
