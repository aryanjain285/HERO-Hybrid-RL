"""Reward arms: every baseline and ablation behind one dispatcher.

Selecting a training arm is a config change, not a code change. The arms are the
five the project needs: the paper's two baselines, HERO, the naive blend the
paper reports as failing (Table 9), and the minimal gated fallback the paper
omits (audit A-4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from hero.core import (
    GroupOutcome,
    HeroConfig,
    group_dispersion,
    stratified_normalise,
    variance_weight,
)


class RewardArm(StrEnum):
    """Reward shaping strategies, one per experimental arm."""

    VERIFIER_ONLY = "verifier_only"
    """Paper baseline: binary rule reward. Zero gradient in uniform groups."""

    RM_ONLY = "rm_only"
    """Paper baseline: raw RM score. Drifts and hacks (paper Fig. 6)."""

    HERO = "hero"
    """Stratified normalisation (Eq. 3) plus variance weighting (Eq. 4-5)."""

    HERO_NO_WEIGHT = "hero_no_weight"
    """HERO with w = 1: the paper's "w/o reweighting" arm (Table 4)."""

    NAIVE_BLEND = "naive_blend"
    """``mix * r_rule + (1 - mix) * r_rm_normalised``. Paper Table 9."""

    GATED_FALLBACK = "gated_fallback"
    """Audit A-4: verifier reward in mixed groups, RM ranking only in uniform
    ones. Tests whether always-on band machinery is necessary."""


@dataclass(frozen=True)
class RewardArmConfig:
    """Arm selection plus the parameters only some arms use.

    Args:
        arm: Which reward strategy to apply.
        hero: Band and weighting parameters, used by the HERO and gated arms.
        blend_mix: The ``alpha`` of paper Table 9, the rule reward's weight in
            :attr:`RewardArm.NAIVE_BLEND`. Paper sweeps 0.1, 0.5, 0.9.
    """

    arm: RewardArm = RewardArm.HERO
    hero: HeroConfig = HeroConfig()
    blend_mix: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend_mix <= 1.0:
            raise ValueError(f"blend_mix must lie in [0, 1]; got {self.blend_mix}")

    @property
    def needs_reward_model(self) -> bool:
        """Whether this arm requires RM scoring, i.e. an RM server."""
        return self.arm is not RewardArm.VERIFIER_ONLY

    @property
    def needs_verifier(self) -> bool:
        """Whether this arm requires verifier labels."""
        return self.arm is not RewardArm.RM_ONLY


def _minmax(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Min-max scale to [0, 1] within a group; all-tied maps to zeros."""
    lo, hi = values.min(), values.max()
    return np.zeros_like(values) if hi == lo else (values - lo) / (hi - lo + epsilon)


def compute_group_reward(
    r_rule: np.ndarray,
    r_rm: np.ndarray,
    cfg: RewardArmConfig,
    sigma_bar: float | None,
) -> GroupOutcome:
    """Sequence-level reward for one prompt group under the selected arm.

    Args:
        r_rule: Verifier labels in {0, 1}. Ignored by :attr:`RewardArm.RM_ONLY`.
        r_rm: Raw RM scores. Ignored by :attr:`RewardArm.VERIFIER_ONLY`.
        cfg: Arm selection and parameters.
        sigma_bar: Frozen running dispersion mean, or None to disable weighting.
            Only :attr:`RewardArm.HERO` consumes it.

    Returns:
        A :class:`GroupOutcome`; ``r_hat`` is the pre-weight reward and
        ``r_final`` the reward handed to GRPO.
    """
    labels = np.asarray(r_rule)
    scores = np.asarray(r_rm, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"shape mismatch: {labels.shape} vs {scores.shape}")
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"expected a non-empty 1-D group; got {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"r_rule must contain only 0/1; got {np.unique(labels)}")
    if not np.isfinite(scores).all():
        raise ValueError("r_rm contains non-finite values; RM scoring failed")

    hero_cfg = cfg.hero
    n_correct = int((labels == 1).sum())
    is_uniform = n_correct in (0, labels.size)
    weight = 1.0

    match cfg.arm:
        case RewardArm.VERIFIER_ONLY:
            r_hat = labels.astype(float)
        case RewardArm.RM_ONLY:
            r_hat = scores.copy()
        case RewardArm.NAIVE_BLEND:
            r_hat = cfg.blend_mix * labels.astype(float) + (1.0 - cfg.blend_mix) * _minmax(
                scores, hero_cfg.minmax_epsilon
            )
        case RewardArm.GATED_FALLBACK:
            # Uniform groups have no verifier gradient, so rank them by RM inside
            # the band; mixed groups keep the untouched binary reward.
            r_hat = (
                stratified_normalise(labels, scores, hero_cfg)
                if is_uniform
                else labels.astype(float)
            )
        case RewardArm.HERO | RewardArm.HERO_NO_WEIGHT:
            r_hat = stratified_normalise(labels, scores, hero_cfg)
            if cfg.arm is RewardArm.HERO and sigma_bar is not None:
                weight = variance_weight(
                    group_dispersion(scores, r_hat, hero_cfg), sigma_bar, hero_cfg
                )
        case _:  # pragma: no cover - StrEnum is exhaustive
            raise ValueError(f"unhandled reward arm: {cfg.arm}")

    singletons = sum(
        1
        for label in (0, 1)
        if (m := labels == label).any() and scores[m].max() == scores[m].min()
    )
    return GroupOutcome(
        r_rule=labels,
        r_rm=scores,
        r_hat=r_hat,
        r_final=weight * r_hat,
        sigma_u=group_dispersion(scores, r_hat, hero_cfg),
        weight=weight,
        is_uniform=is_uniform,
        n_correct=n_correct,
        singleton_bands=singletons,
    )
