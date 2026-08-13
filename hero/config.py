"""Experiment configuration: one object defines a run, hashes to an id.

Defaults reproduce the paper's Qwen3-4B-Base setup (Table 5). Fields the paper
leaves ambiguous carry their decision-log ID. :meth:`ExperimentConfig.digest`
gives runs a stable identity so a results table can be regenerated from configs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

from hero.core import VERL_NORM_ADV_BY_STD_DEFAULT
from hero.registry import ModelRole, resolve
from hero.rewards import RewardArm, RewardArmConfig


LOSS_AGG_MODES: frozenset[str] = frozenset(
    {"token-mean", "seq-mean-token-mean", "seq-mean-token-sum"}
)
"""verl loss aggregation modes, verified against its docs at read time.
Re-check against the pinned commit before M1; verl has added modes over time."""


class TrainingRegime(StrEnum):
    """Which of the paper's three training sets a run uses."""

    EASY_TO_VERIFY = "easy_to_verify"
    HARD_TO_VERIFY = "hard_to_verify"
    MIXED = "mixed"


@dataclass(frozen=True)
class DataConfig:
    """Training data. Sizes follow paper Sec. 4.1 / Appendix A.2.2.

    Args:
        regime: Which training set to use.
        n_prompts: Prompts in the set. The paper uses 2,000 for the single-regime
            sets and 1,000 + 1,000 for mixed.
        pool_size: Size of the CoT working pool the set is drawn from.
        max_prompt_tokens: Paper Table 5.
        max_response_tokens: Paper Table 5 (Qwen line; the OctoThinker line
            uses 4096).
        filter_overlong_prompts: Paper Table 5, ``True``. Unlisted in PRD v1.1.
        decontaminate: n-gram and exact dedup against every eval set. The paper
            does not mention decontamination (D-06); do it and report overlaps.
    """

    regime: TrainingRegime = TrainingRegime.EASY_TO_VERIFY
    n_prompts: int = 2000
    pool_size: int = 40000
    max_prompt_tokens: int = 1024
    max_response_tokens: int = 8192
    filter_overlong_prompts: bool = True
    decontaminate: bool = True

    def __post_init__(self) -> None:
        if self.n_prompts <= 0:
            raise ValueError(f"n_prompts must be positive; got {self.n_prompts}")
        if self.n_prompts > self.pool_size:
            raise ValueError(
                f"n_prompts {self.n_prompts} exceeds pool_size {self.pool_size}"
            )


@dataclass(frozen=True)
class GrpoConfig:
    """GRPO settings. Defaults are paper Table 5 for Qwen3-4B-Base.

    Args:
        rollouts_per_prompt: Paper's ``n``: 8 for Qwen, 16 for OctoThinker.
        train_batch_prompts: Prompts per rollout batch. Paper's "full batch 512".
        mini_batch_prompts: Prompts per gradient step. Paper's 128, which with
            512 gives the "4 step off-policy" the paper names.
        learning_rate: Paper Table 5.
        kl_loss_coef: Paper Table 5: 0 for the Qwen line, 0.001 for OctoThinker.
        entropy_coef: Paper Table 5, 0.
        clip_ratio_low: DAPO-style asymmetric clip lower bound.
        clip_ratio_high: DAPO-style clip-higher upper bound.
        temperature: Training-time sampling temperature, paper Table 5.
        epochs: Passes over the prompt set, paper Table 5.
        norm_adv_by_std_in_grpo: verl's advantage normalisation switch and the
            crux of audit A-1. True is canonical GRPO and verl's default; under
            it the variance weight and the uniform-group band width are both
            provably inert. False is Dr. GRPO / mean-only.
        loss_agg_mode: verl's aggregation; ``token-mean`` is its default.
        use_dynamic_bsz: Paper Table 5, ``True``. Unlisted in PRD v1.1.
    """

    rollouts_per_prompt: int = 8
    train_batch_prompts: int = 512
    mini_batch_prompts: int = 128
    learning_rate: float = 1e-6
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    temperature: float = 1.0
    epochs: int = 20
    norm_adv_by_std_in_grpo: bool = VERL_NORM_ADV_BY_STD_DEFAULT
    loss_agg_mode: str = "token-mean"
    use_dynamic_bsz: bool = True

    def __post_init__(self) -> None:
        if self.rollouts_per_prompt < 2:
            raise ValueError(
                f"GRPO needs at least 2 rollouts per prompt for a group-relative "
                f"advantage; got {self.rollouts_per_prompt}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive; got {self.epochs}")
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive; GRPO needs stochastic rollouts to "
                f"produce within-group variation, got {self.temperature}"
            )
        if self.kl_loss_coef < 0.0:
            raise ValueError(f"kl_loss_coef must be >= 0; got {self.kl_loss_coef}")
        if self.entropy_coef < 0.0:
            raise ValueError(f"entropy_coef must be >= 0; got {self.entropy_coef}")
        if self.train_batch_prompts % self.mini_batch_prompts != 0:
            raise ValueError(
                f"train_batch_prompts {self.train_batch_prompts} must be divisible "
                f"by mini_batch_prompts {self.mini_batch_prompts}"
            )
        if not 0.0 < self.clip_ratio_low < self.clip_ratio_high:
            raise ValueError(
                f"need 0 < clip_low < clip_high; got "
                f"({self.clip_ratio_low}, {self.clip_ratio_high})"
            )
        if self.loss_agg_mode not in LOSS_AGG_MODES:
            raise ValueError(
                f"unknown loss_agg_mode {self.loss_agg_mode!r}; "
                f"expected one of {sorted(LOSS_AGG_MODES)}"
            )

    @property
    def gradient_steps_per_rollout_batch(self) -> int:
        """Mini-batch updates per rollout batch: the paper's "4 step off-policy"."""
        return self.train_batch_prompts // self.mini_batch_prompts


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete, hashable run definition.

    Args:
        name: Human-readable run name.
        policy: Registry key for the backbone.
        reward_model: Registry key for the RM, or None for verifier-only arms.
        judge: Registry key for the hard-to-verify judge.
        reward: Arm selection and reward parameters.
        data: Training data settings.
        grpo: Optimisation settings.
        seed: Run seed. The protocol requires at least 3 per headline number.
        sigma_bar_momentum: EMA momentum for ``sigma_bar`` (D-03).
        notes: Free text recorded in the manifest.
    """

    name: str
    policy: str = "qwen3-1.7b"
    reward_model: str | None = "acemath-7b-rm"
    judge: str = "qwen2.5-72b-judge"
    reward: RewardArmConfig = field(default_factory=RewardArmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    seed: int = 0
    sigma_bar_momentum: float = 0.9
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        resolve(self.policy, ModelRole.POLICY)
        resolve(self.judge, ModelRole.JUDGE)
        if self.reward.needs_reward_model:
            if self.reward_model is None:
                raise ValueError(
                    f"arm {self.reward.arm} needs a reward model, but reward_model "
                    "is None"
                )
            resolve(self.reward_model, ModelRole.REWARD_MODEL)
        if not 0.0 <= self.sigma_bar_momentum < 1.0:
            raise ValueError(
                f"sigma_bar_momentum must lie in [0, 1); got {self.sigma_bar_momentum}"
            )
        if self.data.n_prompts < self.grpo.train_batch_prompts:
            raise ValueError(
                f"prompt set ({self.data.n_prompts}) is smaller than one rollout "
                f"batch ({self.grpo.train_batch_prompts}); under drop_last this "
                "yields no training steps. Lower grpo.train_batch_prompts."
            )

    @property
    def rollouts_per_batch(self) -> int:
        """Sequences generated per rollout batch, and RM calls per step."""
        return self.grpo.train_batch_prompts * self.grpo.rollouts_per_prompt

    @property
    def rollout_batches_per_epoch(self) -> int:
        """Rollout batches per pass over the prompt set, dropping the remainder.

        Assumes ``drop_last`` semantics: 2,000 prompts at batch 512 gives 3
        batches per epoch, not 3.9. A config whose prompt set is smaller than one
        batch is rejected at construction rather than silently reported as 1,
        because under ``drop_last`` it would yield no batches at all.
        """
        return self.data.n_prompts // self.grpo.train_batch_prompts

    @property
    def total_rollout_batches(self) -> int:
        """Rollout batches for the whole run.

        Worth computing before budgeting: 2,000 prompts at batch 512 for 20
        epochs is ~78 rollout batches, not the ~300 the paper's training-curve
        x-axis suggests. The figures are consistent with gradient steps
        (78 x 4 = 312), so treating them as rollout batches overstates the run
        roughly fourfold (audit A-13).
        """
        return self.rollout_batches_per_epoch * self.grpo.epochs

    @property
    def total_gradient_steps(self) -> int:
        """Optimizer steps for the whole run."""
        return self.total_rollout_batches * self.grpo.gradient_steps_per_rollout_batch

    @property
    def total_generations(self) -> int:
        """Total rollouts generated, the dominant cost driver."""
        return self.total_rollout_batches * self.rollouts_per_batch

    def to_dict(self) -> dict:
        """Plain-data view, suitable for JSON or a run manifest."""
        return asdict(self)

    def digest(self, length: int = 12) -> str:
        """Stable content hash of the config, for run ids and result tables.

        Excludes :attr:`name` and :attr:`notes` so that renaming a run does not
        change its scientific identity, and two runs that differ only in label
        collide visibly.
        """
        payload = self.to_dict()
        payload.pop("name", None)
        payload.pop("notes", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:length]

    def with_(self, **changes) -> ExperimentConfig:
        """Copy with top-level fields replaced. Keeps sweeps to one line."""
        return replace(self, **changes)


def a1_grid(base: ExperimentConfig) -> tuple[ExperimentConfig, ...]:
    """The audit A-1 2x2: {std normalisation on, off} x {weighting on, off}.

    The algebra and ``analysis/invariance_check.py`` already settle that the
    weight cannot act when std normalisation is on. This grid measures the
    training-level consequence, and its std-on half is a falsifiable prediction:
    those two arms should be statistically indistinguishable.
    """
    arms = {
        True: RewardArm.HERO,
        False: RewardArm.HERO_NO_WEIGHT,
    }
    return tuple(
        base.with_(
            name=f"a1-std{'on' if norm else 'off'}-w{'on' if weighted else 'off'}",
            reward=replace(base.reward, arm=arms[weighted]),
            grpo=replace(base.grpo, norm_adv_by_std_in_grpo=norm),
            notes="Audit A-1 2x2 cell.",
        )
        for norm in (True, False)
        for weighted in (True, False)
    )
