"""A complete, runnable GRPO trainer on a synthetic reasoning task.

Purpose: close the training-level half of audit A-1 without a GPU. The claim that
HERO's variance weight cancels is a property of the *optimiser*, not of language
models, so it can be demonstrated with any policy class. This module supplies a
real one -- sampling, PPO-clipped ratios, off-policy mini-batch updates, analytic
gradients -- and drives it with the actual reward code from ``hero.rewards``.

What this does and does not establish
-------------------------------------
Establishes, end to end: whether an arm changes the learned policy; whether
verifier-only training stalls when groups are uniformly labelled; whether HERO
recovers learning signal there; and whether the variance weight alters the
optimisation trajectory under each normalisation setting.

Does NOT establish: anything language-specific -- entropy collapse, response-length
bias, tokeniser effects, or reward hacking driven by textual artefacts. Those need
the real stack and are what M1-M3 are for.

The task
--------
Each prompt exposes ``n_responses`` candidate answers described by feature vectors.
One feature coordinate is a latent "skill" level. A response is verifier-correct
when its skill exceeds a threshold, and the reward model reports skill plus fixed
per-response noise -- a deterministic function of the response, as a real RM is.
The policy is a softmax over ``theta . features`` with ``theta`` shared across
prompts, so credit assigned on one prompt transfers to others. That shared
parameterisation is what makes partial credit useful: rewarding a high-skill wrong
answer moves ``theta`` toward the skill direction, which also raises the
probability of correct answers. Without it, dense reward inside an all-wrong group
could not help, and the experiment would be rigged against HERO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hero.core import VERL_ADV_EPSILON, RunningMeanDispersion, grpo_advantage
from hero.rewards import RewardArm, RewardArmConfig, compute_group_reward


@dataclass(frozen=True)
class ToyTaskConfig:
    """Synthetic task geometry.

    Args:
        n_prompts: Distinct prompts in the training set.
        n_responses: Candidate answers per prompt (the policy's action space).
        n_features: Feature dimension; coordinate 0 carries latent skill.
        skill_threshold: Verifier accepts a response when skill >= this. Set high
            so that most groups start uniformly incorrect, which is the regime
            HERO targets.
        rm_noise: Standard deviation of the reward model's per-response error.
            Calibrated so mixed-group AUROC lands near the 0.79 mean the paper
            measures in Appendix B.1. Lower values make the RM near-perfect,
            which would rig the task in favour of dense rewards.
        seed: Task construction seed, separate from the training seed.
    """

    n_prompts: int = 64
    n_responses: int = 12
    n_features: int = 4
    skill_threshold: float = 1.15
    rm_noise: float = 1.6
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_responses < 2:
            raise ValueError("need at least 2 responses to form a group")
        if self.n_features < 1:
            raise ValueError("need at least one feature")
        if self.rm_noise < 0.0:
            raise ValueError("rm_noise must be non-negative")


class ToyTask:
    """Fixed task instance: features, verifier labels, and RM scores.

    All three are deterministic once constructed, so every arm and seed sees an
    identical problem and differences can only come from the reward design.
    """

    def __init__(self, cfg: ToyTaskConfig) -> None:
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        shape = (cfg.n_prompts, cfg.n_responses)

        self.features = rng.normal(0.0, 1.0, (*shape, cfg.n_features))
        # Coordinate 0 is latent skill; the remaining coordinates are distractors
        # the policy must learn to ignore.
        self.skill = self.features[:, :, 0]
        self.correct = self.skill >= cfg.skill_threshold
        self.rm_scores = self.skill + rng.normal(0.0, cfg.rm_noise, shape)

    @property
    def base_correct_fraction(self) -> float:
        """Fraction of all responses the verifier accepts."""
        return float(self.correct.mean())

    def policy(self, theta: np.ndarray) -> np.ndarray:
        """Softmax action probabilities, shape (n_prompts, n_responses)."""
        logits = self.features @ theta
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def expected_accuracy(self, theta: np.ndarray) -> float:
        """Exact verifier accuracy under the policy, averaged over prompts.

        Computed in closed form rather than sampled, so training curves carry no
        evaluation noise.
        """
        return float((self.policy(theta) * self.correct).sum(axis=1).mean())

    def rm_group_auroc(self, theta: np.ndarray, rng: np.random.Generator, n: int) -> float:
        """Mean RM-vs-verifier AUROC over mixed groups, as in paper Appendix B.1.

        Returns NaN when no sampled group contains both labels -- which is itself
        the point of audit A-12: in uniform groups this diagnostic does not exist.
        """
        probs = self.policy(theta)
        scores = []
        for x in range(self.cfg.n_prompts):
            actions = rng.choice(self.cfg.n_responses, size=n, p=probs[x])
            labels = self.correct[x, actions]
            if labels.all() or not labels.any():
                continue
            pos = self.rm_scores[x, actions][labels]
            neg = self.rm_scores[x, actions][~labels]
            wins = (pos[:, None] > neg[None, :]).sum()
            ties = (pos[:, None] == neg[None, :]).sum()
            scores.append((wins + 0.5 * ties) / (pos.size * neg.size))
        return float(np.mean(scores)) if scores else float("nan")


@dataclass(frozen=True)
class TrainConfig:
    """GRPO optimisation settings for the toy trainer.

    Mirrors the structure of the paper's setup -- group sampling, asymmetric
    clipping, several off-policy mini-batch updates per rollout batch -- at a
    scale that runs in a second. The learning rate is not the paper's 1e-6:
    that value belongs to an 4B-parameter transformer, and using it here would
    simply not move a 4-parameter policy.

    Args:
        rollouts_per_prompt: The paper's ``n``.
        steps: Rollout batches.
        mini_batches: Off-policy gradient steps per rollout batch (paper: 4).
        learning_rate: Step size for plain gradient ascent.
        clip_low: PPO lower clip bound.
        clip_high: DAPO-style clip-higher upper bound.
        norm_adv_by_std: verl's ``norm_adv_by_std_in_grpo``.
        adv_epsilon: Denominator epsilon in the advantage. Defaults to verl's
            1e-6. Settable to 0.0 to isolate its effect: it is the only term that
            breaks exact cancellation of a per-group reward scale, so a run with
            it zeroed distinguishes "the weight matters" from "float residue
            amplified by 600 chaotic updates".
        sigma_bar_momentum: EMA momentum for HERO's running dispersion.
        seed: Training seed, controlling rollout sampling only.
    """

    rollouts_per_prompt: int = 8
    steps: int = 150
    mini_batches: int = 4
    learning_rate: float = 0.5
    clip_low: float = 0.2
    clip_high: float = 0.28
    norm_adv_by_std: bool = True
    adv_epsilon: float = VERL_ADV_EPSILON
    sigma_bar_momentum: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rollouts_per_prompt < 2:
            raise ValueError("GRPO needs at least 2 rollouts per prompt")
        if self.mini_batches < 1:
            raise ValueError("need at least one gradient step per rollout batch")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.clip_low < self.clip_high:
            raise ValueError("need 0 < clip_low < clip_high")
        if self.adv_epsilon < 0.0:
            raise ValueError("adv_epsilon must be non-negative")


@dataclass
class TrainResult:
    """Outcome of one training run."""

    theta: np.ndarray
    accuracy: list[float] = field(default_factory=list)
    all_incorrect_fraction: list[float] = field(default_factory=list)
    """Fraction of groups labelled all-0. HERO's primary target regime."""
    all_correct_fraction: list[float] = field(default_factory=list)
    """Fraction labelled all-1. Also gradient-free, but arises from success, so
    reporting a single 'uniform' number conflates starvation with mastery."""
    mean_abs_advantage: list[float] = field(default_factory=list)
    weights_seen: list[float] = field(default_factory=list)

    @property
    def uniform_group_fraction(self) -> list[float]:
        """Total gradient-free fraction, all-0 plus all-1."""
        return [a + b for a, b in zip(self.all_incorrect_fraction, self.all_correct_fraction)]

    @property
    def final_accuracy(self) -> float:
        return self.accuracy[-1]

    @property
    def initial_accuracy(self) -> float:
        return self.accuracy[0]


def _clipped_policy_gradient(
    task: ToyTask,
    theta: np.ndarray,
    prompts: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    log_probs_old: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
    """Analytic gradient of the PPO-clipped surrogate w.r.t. theta.

    For a softmax policy, d(log pi(a|x))/d(theta) = phi[x, a] - E_pi[phi[x, .]],
    so d(ratio)/d(theta) = ratio * that. A sample contributes nothing when its
    ratio is clipped in the direction the objective would push it -- the standard
    PPO condition, applied per sample rather than approximated.
    """
    probs = task.policy(theta)
    log_probs = np.log(probs[prompts, actions])
    ratios = np.exp(log_probs - log_probs_old)

    baseline = np.einsum("ik,ikd->id", probs[prompts], task.features[prompts])
    score = task.features[prompts, actions] - baseline

    active = np.where(
        advantages >= 0.0,
        ratios <= 1.0 + cfg.clip_high,
        ratios >= 1.0 - cfg.clip_low,
    )
    coeff = np.where(active, advantages * ratios, 0.0)
    return (coeff[:, None] * score).mean(axis=0)


def train(
    task: ToyTask,
    arm: RewardArmConfig,
    cfg: TrainConfig,
) -> TrainResult:
    """Run GRPO on the toy task under one reward arm.

    Rollout sampling depends only on the policy and the seeded RNG, so two arms
    given the same seed draw identical rollouts until their updates diverge. That
    makes an exact trajectory comparison meaningful.
    """
    rng = np.random.default_rng(cfg.seed)
    theta = np.zeros(task.cfg.n_features)
    sigma_bar = RunningMeanDispersion(momentum=cfg.sigma_bar_momentum)
    result = TrainResult(theta=theta)
    n = cfg.rollouts_per_prompt
    n_prompts = task.cfg.n_prompts

    for _ in range(cfg.steps):
        result.accuracy.append(task.expected_accuracy(theta))

        probs = task.policy(theta)
        actions = np.stack(
            [rng.choice(task.cfg.n_responses, size=n, p=probs[x]) for x in range(n_prompts)]
        )

        # Reward each group through the production reward path.
        advantages = np.zeros((n_prompts, n))
        all_incorrect = all_correct = 0
        weights = []
        frozen_sigma = sigma_bar.value if sigma_bar.is_warm else None
        for x in range(n_prompts):
            labels = task.correct[x, actions[x]].astype(int)
            scores = task.rm_scores[x, actions[x]]
            outcome = compute_group_reward(
                labels,
                None if arm.arm is RewardArm.VERIFIER_ONLY else scores,
                arm,
                frozen_sigma,
            )
            advantages[x] = grpo_advantage(
                outcome.r_final, cfg.norm_adv_by_std, cfg.adv_epsilon
            )
            all_incorrect += int(outcome.n_correct == 0)
            all_correct += int(outcome.n_correct == n)
            weights.append(outcome.weight)
            # sigma_u is None only for verifier-only, which never consults it.
            if outcome.sigma_u is not None:
                sigma_bar.observe(outcome.sigma_u)
        if sigma_bar.has_pending:
            sigma_bar.end_batch()

        result.all_incorrect_fraction.append(all_incorrect / n_prompts)
        result.all_correct_fraction.append(all_correct / n_prompts)
        result.mean_abs_advantage.append(float(np.abs(advantages).mean()))
        result.weights_seen.append(float(np.mean(weights)))

        flat_prompts = np.repeat(np.arange(n_prompts), n)
        flat_actions = actions.reshape(-1)
        flat_adv = advantages.reshape(-1)
        log_probs_old = np.log(probs[flat_prompts, flat_actions])

        # Off-policy mini-batch updates over a fixed partition of the batch, so
        # the split is deterministic and identical across arms.
        splits = np.array_split(np.arange(flat_prompts.size), cfg.mini_batches)
        for idx in splits:
            grad = _clipped_policy_gradient(
                task,
                theta,
                flat_prompts[idx],
                flat_actions[idx],
                flat_adv[idx],
                log_probs_old[idx],
                cfg,
            )
            theta = theta + cfg.learning_rate * grad

    result.theta = theta
    result.accuracy.append(task.expected_accuracy(theta))
    return result
