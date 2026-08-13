"""HERO reproduction: reward core, experiment configuration, and audit tooling."""

from hero.config import (
    DataConfig,
    ExperimentConfig,
    GrpoConfig,
    TrainingRegime,
    a1_grid,
)
from hero.core import (
    VERL_ADV_EPSILON,
    VERL_NORM_ADV_BY_STD_DEFAULT,
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
from hero.registry import ModelRole, ModelSpec, ServingMode, all_specs, by_role, resolve
from hero.rewards import RewardArm, RewardArmConfig, compute_group_reward

__all__ = [
    "VERL_ADV_EPSILON",
    "VERL_NORM_ADV_BY_STD_DEFAULT",
    "DataConfig",
    "ExperimentConfig",
    "GroupOutcome",
    "GrpoConfig",
    "HeroConfig",
    "HeroConfigError",
    "ModelRole",
    "ModelSpec",
    "RewardArm",
    "RewardArmConfig",
    "RunningMeanDispersion",
    "ServingMode",
    "TrainingRegime",
    "a1_grid",
    "all_specs",
    "by_role",
    "compute_group_reward",
    "grpo_advantage",
    "group_dispersion",
    "resolve",
    "shape_group",
    "stratified_normalise",
    "variance_weight",
]
