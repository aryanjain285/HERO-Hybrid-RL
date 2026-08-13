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
    build_outcome,
    grpo_advantage,
    group_dispersion,
    minmax_z,
    shape_group,
    stratified_normalise,
    variance_weight,
)
from hero.registry import ModelRole, ModelSpec, ServingMode, all_specs, by_role, resolve
from hero.rewards import RewardArm, RewardArmConfig, compute_group_reward

__all__ = [
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
    "VERL_ADV_EPSILON",
    "VERL_NORM_ADV_BY_STD_DEFAULT",
    "a1_grid",
    "all_specs",
    "build_outcome",
    "by_role",
    "compute_group_reward",
    "group_dispersion",
    "grpo_advantage",
    "minmax_z",
    "resolve",
    "shape_group",
    "stratified_normalise",
    "variance_weight",
]
