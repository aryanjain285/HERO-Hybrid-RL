"""Tests for the shell-facing CLI.

The pipeline scripts parse this output, so its shape is a contract.
"""

from __future__ import annotations

import json

import pytest

from hero.cli import main
from hero.registry import TIERS, ModelRole, format_tier, resolve, tier


class TestTiers:
    @pytest.mark.parametrize("name", sorted(TIERS))
    def test_every_tier_resolves(self, name):
        specs = tier(name)
        assert specs
        assert all(spec.hf_id for spec in specs)

    @pytest.mark.parametrize("name", sorted(TIERS))
    def test_every_tier_has_exactly_one_policy(self, name):
        policies = [s for s in tier(name) if s.role is ModelRole.POLICY]
        assert len(policies) == 1, f"{name} has {len(policies)} policies"

    def test_unknown_tier_lists_alternatives(self):
        with pytest.raises(KeyError, match="available:"):
            tier("enormous")

    def test_tiers_exclude_api_judges(self):
        """Nothing downloadable exists for a closed API model."""
        for name in TIERS:
            assert all(s.role is not ModelRole.JUDGE for s in tier(name))

    def test_dense_tiers_include_a_reward_model(self):
        for name in TIERS:
            assert any(s.role is ModelRole.REWARD_MODEL for s in tier(name))


class TestFormatTier:
    def test_hf_id_is_one_per_line(self):
        lines = format_tier("dev", "hf_id").splitlines()
        assert lines == [s.hf_id for s in tier("dev")]

    def test_keys_round_trip_through_the_registry(self):
        for key in format_tier("headline", "key").splitlines():
            assert resolve(key).key == key

    def test_table_is_aligned_and_labelled(self):
        lines = format_tier("dev", "table").splitlines()
        assert len(lines) == len(tier("dev"))
        assert all("/" in line for line in lines)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="unknown field"):
            format_tier("dev", "size")


class TestCli:
    def test_models_prints_ids(self, capsys):
        assert main(["models", "--tier", "smoke", "--field", "hf_id"]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert out == [s.hf_id for s in tier("smoke")]

    def test_config_emits_valid_json_with_costs(self, capsys):
        assert main(["config", "--arm", "hero", "--name", "x"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["arm"] == "hero"
        assert payload["digest"]
        assert payload["gradient_steps"] > 0
        assert payload["total_generations"] > 0

    def test_config_drops_the_reward_model_for_verifier_only(self, capsys):
        assert main(["config", "--arm", "verifier_only"]) == 0
        assert json.loads(capsys.readouterr().out)["reward_model"] is None

    def test_config_surfaces_the_a1_switch(self, capsys):
        """The most consequential setting must be visible in every manifest."""
        main(["config"])
        assert "norm_adv_by_std_in_grpo" in json.loads(capsys.readouterr().out)

    def test_digest_is_stable_across_invocations(self, capsys):
        main(["config", "--name", "a"])
        first = json.loads(capsys.readouterr().out)["digest"]
        main(["config", "--name", "b"])
        assert json.loads(capsys.readouterr().out)["digest"] == first

    def test_missing_subcommand_is_an_error(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_arm_is_rejected(self):
        with pytest.raises(SystemExit):
            main(["config", "--arm", "magic"])
