"""Reward arms: every baseline and ablation behind one dispatcher.

Selecting a training arm is a config change, not a code change. The arms are the
five the project needs: the paper's two baselines, HERO, the naive blend the
paper reports as failing (Table 9), and the minimal gated fallback the paper
omits (audit A-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from hero.core import (
    GroupOutcome,
    HeroConfig,
    build_outcome,
    minmax_z,
    shape_group,
    stratified_normalise,
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
    hero: HeroConfig = field(default_factory=HeroConfig)
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


def compute_group_reward(
    r_rule: np.ndarray,
    r_rm: np.ndarray | None,
    cfg: RewardArmConfig,
    sigma_bar: float | None,
) -> GroupOutcome:
    """Sequence-level reward for one prompt group under the selected arm.

    Args:
        r_rule: Verifier labels in {0, 1}. Required by every arm: verifiers are
            cheap, and band-occupancy telemetry is wanted even for
            :attr:`RewardArm.RM_ONLY`, where the labels do not enter the reward.
        r_rm: Raw RM scores, or None for :attr:`RewardArm.VERIFIER_ONLY`, which
            must be runnable without standing up an RM server.
        cfg: Arm selection and parameters.
        sigma_bar: Frozen running dispersion mean, or None to disable weighting.
            Consumed only by :attr:`RewardArm.HERO`; the gated arm deliberately
            ignores it, since its point is to be the minimal hybrid.

    Returns:
        A :class:`GroupOutcome`; ``r_hat`` is the pre-weight reward and
        ``r_final`` the reward handed to GRPO.

    Raises:
        ValueError: Malformed group, or ``r_rm`` missing for an arm that needs it.
    """
    labels = np.asarray(r_rule)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"expected a non-empty 1-D group; got {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"r_rule must contain only 0/1; got {np.unique(labels)}")

    if r_rm is None:
        if cfg.needs_reward_model:
            raise ValueError(f"arm {cfg.arm} requires r_rm, but None was given")
        scores = None
    else:
        scores = np.asarray(r_rm, dtype=float)
        if labels.shape != scores.shape:
            raise ValueError(f"shape mismatch: {labels.shape} vs {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("r_rm contains non-finite values; RM scoring failed")

    hero_cfg = cfg.hero
    is_uniform = int((labels == 1).sum()) in (0, labels.size)

    # HERO's shaping lives in hero.core.shape_group; dispatch rather than repeat it.
    if cfg.arm in (RewardArm.HERO, RewardArm.HERO_NO_WEIGHT):
        return shape_group(
            labels,
            scores,
            hero_cfg,
            sigma_bar if cfg.arm is RewardArm.HERO else None,
        )

    match cfg.arm:
        case RewardArm.VERIFIER_ONLY:
            r_hat = labels.astype(float)
        case RewardArm.RM_ONLY:
            r_hat = scores.copy()
        case RewardArm.NAIVE_BLEND:
            r_hat = cfg.blend_mix * labels.astype(float) + (
                1.0 - cfg.blend_mix
            ) * minmax_z(scores, hero_cfg)
        case RewardArm.GATED_FALLBACK:
            # Uniform groups have no verifier gradient, so rank them by RM inside
            # the band; mixed groups keep the untouched binary reward.
            r_hat = (
                stratified_normalise(labels, scores, hero_cfg)
                if is_uniform
                else labels.astype(float)
            )
        case _:  # pragma: no cover - every arm is handled above
            raise ValueError(f"unhandled reward arm: {cfg.arm}")

    return build_outcome(labels, scores, r_hat, 1.0, hero_cfg)
