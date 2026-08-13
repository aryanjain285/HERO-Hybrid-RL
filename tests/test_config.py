"""Tests for the registry and experiment configuration.

The step-accounting tests exist because getting them wrong misprices the compute
budget by a factor of four (audit A-13).
"""

from __future__ import annotations

import pytest

from hero.config import (
    DataConfig,
    ExperimentConfig,
    GrpoConfig,
    TrainingRegime,
    a1_grid,
)
from hero.registry import ModelRole, ServingMode, all_specs, by_role, resolve
from hero.rewards import RewardArm, RewardArmConfig


class TestRegistry:
    def test_keys_are_unique(self):
        keys = [spec.key for spec in all_specs()]
        assert len(keys) == len(set(keys))

    def test_resolve_checks_role(self):
        assert resolve("qwen3-4b", ModelRole.POLICY).params_b == 4.0
        with pytest.raises(ValueError, match="role"):
            resolve("gpt-4o", ModelRole.POLICY)

    def test_unknown_key_lists_alternatives(self):
        with pytest.raises(KeyError, match="registered:"):
            resolve("qwen4-9000")

    def test_every_role_is_populated(self):
        for role in ModelRole:
            assert by_role(role), f"no models registered for {role}"

    def test_paper_defaults_are_marked(self):
        marked = {spec.key for spec in all_specs() if spec.paper_default}
        assert {"qwen3-4b", "octothinker-8b", "acemath-7b-rm", "gpt-4o"} <= marked

    def test_remote_models_are_flagged_for_budgeting(self):
        assert resolve("gpt-4o").serving is ServingMode.REMOTE_API


class TestGrpoConfig:
    def test_paper_defaults(self):
        cfg = GrpoConfig()
        assert (cfg.rollouts_per_prompt, cfg.train_batch_prompts) == (8, 512)
        assert (cfg.clip_ratio_low, cfg.clip_ratio_high) == (0.2, 0.28)
        assert cfg.kl_loss_coef == 0.0

    def test_four_step_off_policy(self):
        """512 / 128 = 4, matching the paper's own description."""
        assert GrpoConfig().gradient_steps_per_rollout_batch == 4

    def test_verl_default_normalisation_is_the_starting_point(self):
        assert GrpoConfig().norm_adv_by_std_in_grpo is True

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rollouts_per_prompt": 1}, "at least 2 rollouts"),
            ({"train_batch_prompts": 500}, "divisible"),
            ({"clip_ratio_low": 0.3}, "clip_low < clip_high"),
            ({"loss_agg_mode": "mean"}, "loss_agg_mode"),
        ],
    )
    def test_invalid_settings_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GrpoConfig(**kwargs)


class TestDataConfig:
    def test_paper_sizes(self):
        cfg = DataConfig()
        assert (cfg.n_prompts, cfg.pool_size) == (2000, 40000)
        assert cfg.filter_overlong_prompts is True

    def test_prompt_count_cannot_exceed_pool(self):
        with pytest.raises(ValueError, match="exceeds pool_size"):
            DataConfig(n_prompts=50_000)


class TestExperimentConfig:
    def test_minimal_construction(self):
        cfg = ExperimentConfig(name="dev-hero")
        assert cfg.policy == "qwen3-1.7b"
        assert cfg.reward.arm is RewardArm.HERO

    def test_verifier_only_needs_no_reward_model(self):
        cfg = ExperimentConfig(
            name="verifier-only",
            reward_model=None,
            reward=RewardArmConfig(arm=RewardArm.VERIFIER_ONLY),
        )
        assert cfg.reward_model is None

    def test_missing_reward_model_for_dense_arm_is_rejected(self):
        with pytest.raises(ValueError, match="needs a reward model"):
            ExperimentConfig(name="broken", reward_model=None)

    def test_bad_model_key_is_rejected_at_construction(self):
        with pytest.raises(KeyError):
            ExperimentConfig(name="typo", policy="qwen3-1.7")

    def test_role_confusion_is_rejected(self):
        with pytest.raises(ValueError, match="role"):
            ExperimentConfig(name="swapped", policy="gpt-4o")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(name="   ")


class TestStepAccounting:
    """Paper Table 5 arithmetic, which the training-curve axes contradict."""

    def test_paper_scale_run(self):
        cfg = ExperimentConfig(name="paper", policy="qwen3-4b")
        assert cfg.rollouts_per_batch == 4096
        assert cfg.rollout_batches_per_epoch == 3
        assert cfg.total_rollout_batches == 60
        assert cfg.total_gradient_steps == 240

    def test_gradient_steps_land_near_the_published_curve_length(self):
        """The paper's 300-step curves match gradient steps, not rollout batches.

        2,000 prompts, batch 512, 20 epochs gives 60-78 rollout batches depending
        on remainder handling, hence 240-312 gradient steps. Reading the x-axis as
        rollout batches would overstate the run roughly fourfold.
        """
        cfg = ExperimentConfig(name="paper", policy="qwen3-4b")
        assert 200 <= cfg.total_gradient_steps <= 320
        assert cfg.total_rollout_batches < 100

    def test_generation_volume(self):
        cfg = ExperimentConfig(name="paper", policy="qwen3-4b")
        assert cfg.total_generations == 60 * 4096

    def test_small_prompt_set_still_yields_one_batch_per_epoch(self):
        cfg = ExperimentConfig(name="tiny", data=DataConfig(n_prompts=100))
        assert cfg.rollout_batches_per_epoch == 1


class TestDigest:
    def test_stable_across_calls(self):
        cfg = ExperimentConfig(name="a")
        assert cfg.digest() == cfg.digest()

    def test_ignores_name_and_notes(self):
        a = ExperimentConfig(name="a", notes="first")
        b = ExperimentConfig(name="b", notes="second")
        assert a.digest() == b.digest()

    def test_sensitive_to_scientific_content(self):
        base = ExperimentConfig(name="a")
        assert base.digest() != base.with_(seed=1).digest()
        assert (
            base.digest()
            != base.with_(grpo=GrpoConfig(norm_adv_by_std_in_grpo=False)).digest()
        )
        assert (
            base.digest()
            != base.with_(data=DataConfig(regime=TrainingRegime.MIXED)).digest()
        )

    def test_with_returns_a_copy(self):
        base = ExperimentConfig(name="a")
        assert base.with_(seed=7).seed == 7
        assert base.seed == 0


class TestA1Grid:
    def test_shape_and_distinctness(self):
        grid = a1_grid(ExperimentConfig(name="base"))
        assert len(grid) == 4
        assert len({cfg.digest() for cfg in grid}) == 4

    def test_covers_both_switches(self):
        grid = a1_grid(ExperimentConfig(name="base"))
        combos = {
            (cfg.grpo.norm_adv_by_std_in_grpo, cfg.reward.arm is RewardArm.HERO)
            for cfg in grid
        }
        assert combos == {(True, True), (True, False), (False, True), (False, False)}

    def test_inherits_base_settings(self):
        base = ExperimentConfig(name="base", policy="qwen3-4b", seed=3)
        assert all(c.policy == "qwen3-4b" and c.seed == 3 for c in a1_grid(base))

    def test_names_are_unique_and_descriptive(self):
        names = [cfg.name for cfg in a1_grid(ExperimentConfig(name="base"))]
        assert len(set(names)) == 4
        assert all(name.startswith("a1-std") for name in names)
