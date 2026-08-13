"""HERO reward core (Tao et al., ICLR 2026), framework-free.

Pure numpy: no torch, no verl, no network. Single source of truth for HERO's
mathematics, shared by the audit in ``analysis/`` and (later) by the verl reward
manager, and pinned by ``tests/``.

Paper anchors: Eq. 3 stratified normalisation, Eq. 4 variance weighting,
Eq. 5 final reward. Ambiguities the paper leaves open are named fields on
:class:`HeroConfig` carrying their decision-log ID; see ``docs/decisions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Transcribed from verl/trainer/ppo/core_algos.py, not chosen here.
VERL_ADV_EPSILON: float = 1e-6
VERL_NORM_ADV_BY_STD_DEFAULT: bool = True


class HeroConfigError(ValueError):
    """A HERO configuration violates a documented precondition."""


@dataclass(frozen=True)
class HeroConfig:
    """HERO hyperparameters. Defaults follow Appendix A.1 (authoritative, D-07).

    Args:
        alpha: Incorrect-band half-width (Eq. 3). Paper: 0.05 easy, 0.1 mixed/hard.
        beta: Correct-band half-width (Eq. 3).
        minmax_epsilon: Eq. 3 epsilon. Guards division by zero only; it does not
            govern the rule-vs-RM balance the paper attributes to it (A-8).
        w_min: Eq. 4 lower bound. Appendix 0.4, main text 0.5 (A-7i, D-07).
        w_max: Eq. 4 upper bound. Appendix 3.0, main text 2.0.
        w_slope: Eq. 4 ``k``. Appendix 6, main text 5.
        sigma_on_raw_rm: Compute ``sigma_u`` on raw RM scores (D-02) rather than
            band-normalised ones, whose dispersion is capped by the band width.
        singleton_z: ``z`` when a band has one member or all-tied scores. Eq. 3
            literally gives 0 (band floor); 0.5 is the band midpoint (D-05).
    """

    alpha: float = 0.05
    beta: float = 0.05
    minmax_epsilon: float = 1e-6
    w_min: float = 0.4
    w_max: float = 3.0
    w_slope: float = 6.0
    sigma_on_raw_rm: bool = True
    singleton_z: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise HeroConfigError(f"alpha must lie in (0, 1]; got {self.alpha}")
        if not 0.0 < self.beta <= 1.0:
            raise HeroConfigError(f"beta must lie in (0, 1]; got {self.beta}")
        # Ordering preservation (P1) needs 1 - beta > alpha. The paper asserts P1
        # but permits alpha, beta in (0, 1], which admits band overlap (A-15).
        if self.alpha + self.beta >= 1.0:
            raise HeroConfigError(
                f"alpha + beta must be < 1 to preserve correctness ordering (P1); "
                f"got {self.alpha} + {self.beta}. Incorrect band reaches "
                f"{self.alpha}, correct band starts at {1.0 - self.beta}."
            )
        if self.w_min <= 0.0:
            raise HeroConfigError(f"w_min must be > 0; got {self.w_min}")
        if self.w_max < self.w_min:
            raise HeroConfigError(f"w_max {self.w_max} < w_min {self.w_min}")
        if self.minmax_epsilon <= 0.0:
            raise HeroConfigError(f"minmax_epsilon must be > 0; got {self.minmax_epsilon}")
        if not 0.0 <= self.singleton_z <= 1.0:
            raise HeroConfigError(f"singleton_z must lie in [0, 1]; got {self.singleton_z}")

    @property
    def incorrect_band(self) -> tuple[float, float]:
        """Interval reachable by verifier-incorrect rollouts."""
        return (-self.alpha, self.alpha)

    @property
    def correct_band(self) -> tuple[float, float]:
        """Interval reachable by verifier-correct rollouts."""
        return (1.0 - self.beta, 1.0 + self.beta)


def minmax_z(values: np.ndarray, cfg: HeroConfig) -> np.ndarray:
    """Min-max scale to roughly [0, 1] within a group (the inner term of Eq. 3).

    A degenerate range (single member, or all values tied) yields
    ``cfg.singleton_z`` rather than dividing 0 by epsilon, which is the same
    number Eq. 3 produces but without relying on the epsilon to rescue it.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full(values.shape, cfg.singleton_z)
    return (values - lo) / (hi - lo + cfg.minmax_epsilon)


def stratified_normalise(
    r_rule: np.ndarray, r_rm: np.ndarray, cfg: HeroConfig
) -> np.ndarray:
    """Map RM scores into disjoint verifier-defined bands (Eq. 3).

    Min-max normalisation is per band, so only the RM's within-band ranking and
    relative spacing reach the policy; location and scale are discarded up to
    O(epsilon/range). RM calibration therefore cannot influence ``r_hat``.

    Args:
        r_rule: Verifier labels in {0, 1}, shape (n,).
        r_rm: Raw reward-model scores, shape (n,).
        cfg: HERO configuration.

    Returns:
        Shaped rewards, shape (n,), each inside its label's band.

    Raises:
        ValueError: Shape mismatch, empty group, labels outside {0, 1}, or
            non-finite scores. Verifier errors map to 0 upstream (D-04).
    """
    r_rule = np.asarray(r_rule)
    r_rm = np.asarray(r_rm, dtype=float)
    if r_rule.shape != r_rm.shape:
        raise ValueError(f"shape mismatch: r_rule {r_rule.shape} vs r_rm {r_rm.shape}")
    if r_rule.ndim != 1 or r_rule.size == 0:
        raise ValueError(f"expected a non-empty 1-D group; got shape {r_rule.shape}")
    if not np.isin(r_rule, (0, 1)).all():
        raise ValueError(
            f"r_rule must contain only 0/1; got {np.unique(r_rule)}. Map verifier "
            "errors and timeouts to 0 upstream (D-04)."
        )
    if not np.isfinite(r_rm).all():
        raise ValueError("r_rm contains non-finite values; RM scoring failed")

    r_hat = np.zeros(r_rm.shape, dtype=float)
    for label, floor, span in ((0, -cfg.alpha, 2.0 * cfg.alpha),
                               (1, 1.0 - cfg.beta, 2.0 * cfg.beta)):
        m = r_rule == label
        if m.any():
            r_hat[m] = floor + span * minmax_z(r_rm[m], cfg)
    return r_hat


def variance_weight(sigma_u: float, sigma_bar: float, cfg: HeroConfig) -> float:
    """Bounded logistic prompt weight (Eq. 4), in ``[w_min, w_max]``.

    The logistic argument is in raw RM units. With ``k = 6`` and AceMath-scale
    scores, dispersion gaps of O(1) saturate it, making the weight an effective
    two-level gate at ``sigma_u = sigma_bar`` (A-16).
    """
    if not np.isfinite(sigma_u) or not np.isfinite(sigma_bar):
        raise ValueError(f"non-finite dispersion: sigma_u={sigma_u}, sigma_bar={sigma_bar}")
    # tanh form: numerically stable at the large arguments saturation makes routine.
    logistic = 0.5 * (1.0 + np.tanh(0.5 * cfg.w_slope * (sigma_u - sigma_bar)))
    return float(cfg.w_min + (cfg.w_max - cfg.w_min) * logistic)


def group_dispersion(r_rm: np.ndarray, r_hat: np.ndarray, cfg: HeroConfig) -> float:
    """``sigma_u`` for one group (D-02). Unbiased std, matching ``torch.std``."""
    source = np.asarray(r_rm if cfg.sigma_on_raw_rm else r_hat, dtype=float)
    return 0.0 if source.size < 2 else float(source.std(ddof=1))


class RunningMeanDispersion:
    """The running mean ``sigma_bar`` of Eq. 4; EMA, warm-started (D-03).

    Reads are frozen within a batch: :attr:`value` changes only at
    :meth:`end_batch`. A per-group update would make each reward depend on the
    order groups were visited, which is nondeterministic under data-parallel
    sharding and would make runs unreproducible.
    """

    def __init__(self, momentum: float = 0.9) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1); got {momentum}")
        self.momentum = momentum
        self._value: float | None = None
        self._pending: list[float] = []

    @property
    def is_warm(self) -> bool:
        """True once at least one batch has been folded in."""
        return self._value is not None

    @property
    def has_pending(self) -> bool:
        """True when groups have been observed but not yet folded in."""
        return bool(self._pending)

    @property
    def value(self) -> float:
        """Current ``sigma_bar``, frozen for the duration of the batch."""
        if self._value is None:
            raise RuntimeError(
                "sigma_bar is undefined before the first end_batch(); weight the "
                "first batch neutrally (w = 1) instead (D-03)."
            )
        return self._value

    def observe(self, sigma_u: float) -> None:
        """Stage one group's dispersion for the end-of-batch update."""
        if not np.isfinite(sigma_u):
            raise ValueError(f"non-finite sigma_u: {sigma_u}")
        self._pending.append(float(sigma_u))

    def end_batch(self) -> float:
        """Fold the staged batch in and return the new ``sigma_bar``."""
        if not self._pending:
            raise RuntimeError("end_batch() called with no observed groups")
        batch_mean = float(np.mean(self._pending))
        self._pending.clear()
        self._value = (
            batch_mean
            if self._value is None
            else self.momentum * self._value + (1.0 - self.momentum) * batch_mean
        )
        return self._value


@dataclass(frozen=True)
class GroupOutcome:
    """Per-group rewards and the telemetry the audit plan requires.

    Band degeneracy is reported three ways because they answer different
    questions and conflating them would misinform decision D-05, whose revisit
    trigger is a rate:

    * ``singleton_bands`` -- bands holding exactly one rollout. Unavoidable.
    * ``tied_bands`` -- bands holding several rollouts whose RM scores are all
      equal. Means the RM failed to discriminate, which is a different problem.
    * ``pinned_rollouts`` -- rollouts actually forced to ``singleton_z`` by
      either case. This is the quantity D-05's threshold should be read against.

    Fields requiring RM scores are None when the arm does not use them.

    ``sigma_u`` is the dispersion of the RM signal per decision D-02: raw RM
    scores under the default ``sigma_on_raw_rm=True``, which is well defined for
    every arm. Under ``sigma_on_raw_rm=False`` it is the dispersion of ``r_hat``,
    whose meaning is arm-dependent and only comparable within one arm.
    """

    r_rule: np.ndarray
    r_hat: np.ndarray
    r_final: np.ndarray
    weight: float
    is_uniform: bool
    n_correct: int
    r_rm: np.ndarray | None = None
    sigma_u: float | None = None
    singleton_bands: int | None = None
    tied_bands: int | None = None
    pinned_rollouts: int | None = None


def build_outcome(
    r_rule: np.ndarray,
    r_rm: np.ndarray | None,
    r_hat: np.ndarray,
    weight: float,
    cfg: HeroConfig,
) -> GroupOutcome:
    """Assemble a :class:`GroupOutcome`, the single place telemetry is derived.

    Arrays are copied, so a caller mutating its inputs afterwards cannot
    retroactively alter a recorded outcome -- the frozen dataclass otherwise
    promises an immutability it does not have.
    """
    labels = np.array(r_rule, copy=True)
    n_correct = int((labels == 1).sum())
    scores = None if r_rm is None else np.array(r_rm, dtype=float, copy=True)

    singleton_bands = tied_bands = pinned = None
    sigma_u = None
    if scores is not None:
        singleton_bands = tied_bands = pinned = 0
        for label in (0, 1):
            m = labels == label
            size = int(m.sum())
            if size == 0:
                continue
            if size == 1:
                singleton_bands += 1
                pinned += 1
            elif scores[m].max() == scores[m].min():
                tied_bands += 1
                pinned += size
        sigma_u = group_dispersion(scores, r_hat, cfg)

    return GroupOutcome(
        r_rule=labels,
        r_hat=np.array(r_hat, copy=True),
        r_final=weight * np.asarray(r_hat, dtype=float),
        weight=weight,
        is_uniform=n_correct in (0, labels.size),
        n_correct=n_correct,
        r_rm=scores,
        sigma_u=sigma_u,
        singleton_bands=singleton_bands,
        tied_bands=tied_bands,
        pinned_rollouts=pinned,
    )


def shape_group(
    r_rule: np.ndarray,
    r_rm: np.ndarray,
    cfg: HeroConfig,
    sigma_bar: float | None,
) -> GroupOutcome:
    """Full HERO reward for one prompt group (Eq. 3 -> 4 -> 5).

    The single implementation of HERO's shaping; ``hero.rewards`` dispatches its
    HERO arms here rather than reimplementing Eq. 3-5.

    Args:
        r_rule: Verifier labels in {0, 1}.
        r_rm: Raw RM scores.
        cfg: HERO configuration.
        sigma_bar: Frozen running dispersion mean, or None to disable weighting
            (``w = 1``) -- both the "w/o reweighting" ablation arm and the
            correct behaviour for the first batch, before warm-up.
    """
    r_hat = stratified_normalise(r_rule, r_rm, cfg)
    weight = (
        1.0
        if sigma_bar is None
        else variance_weight(group_dispersion(r_rm, r_hat, cfg), sigma_bar, cfg)
    )
    return build_outcome(r_rule, r_rm, r_hat, weight, cfg)


def grpo_advantage(
    rewards: np.ndarray,
    norm_by_std: bool = VERL_NORM_ADV_BY_STD_DEFAULT,
    epsilon: float = VERL_ADV_EPSILON,
) -> np.ndarray:
    """Group-relative advantage for one prompt group, mirroring verl.

    Follows ``compute_grpo_outcome_advantage``: unbiased std (``torch.std``),
    and verl's single-member special case of mean=0/std=1, which makes the
    advantage equal the raw reward. That case cannot fire at n >= 2 but would
    silently distort any experiment that filtered a group down to one member.

    Fidelity assumption: verl works on token-level reward tensors and reduces
    them with ``sum(dim=-1)`` over the response mask. This function takes the
    already-reduced sequence-level reward, which is exactly equivalent for
    outcome rewards placed on the final token -- the regime HERO operates in.
    Anything that spreads reward across tokens invalidates the comparison.

    Args:
        rewards: Sequence-level rewards for one group, shape (n,).
        norm_by_std: verl's ``algorithm.norm_adv_by_std_in_grpo``. True is
            canonical GRPO and verl's default; False is Dr. GRPO / mean-only.
        epsilon: Denominator epsilon; verl's default is 1e-6.
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError(f"expected a non-empty 1-D group; got shape {rewards.shape}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative; got {epsilon}")
    if rewards.size == 1:
        return rewards.copy()
    centred = rewards - rewards.mean()
    if not norm_by_std:
        return centred
    denominator = rewards.std(ddof=1) + epsilon
    if denominator == 0.0:
        # Only reachable with epsilon=0 on a constant group, which verl's 1e-6
        # default rules out. Zero is the correct limit: no within-group signal.
        return np.zeros_like(centred)
    return centred / denominator
