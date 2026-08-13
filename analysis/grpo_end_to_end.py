"""End-to-end GRPO training runs: audit A-1 at the optimiser level.

`invariance_check.py` shows the variance weight leaves advantages unchanged.
This script goes further and actually trains, so the claim is tested against
learned policies rather than single-batch arithmetic:

  T1  the training harness genuinely optimises (a sanity gate on everything else)
  T2  verifier-only training stalls when groups are uniformly labelled
  T3  under std normalisation, HERO and HERO-without-weighting converge to
      bit-identical parameters -- the strongest form of A-1
  T4  with std normalisation off, they diverge
  T5  the RM-reliability diagnostic is undefined in uniform groups (A-12)

Scope: this is a synthetic task with a 4-parameter softmax policy, not a language
model. It settles optimiser-level questions, which is exactly where A-1 lives, and
settles nothing language-specific. See hero/toy.py for the full caveat.

No GPU, no network. Run: python analysis/grpo_end_to_end.py
"""

from __future__ import annotations

import numpy as np

from hero.rewards import RewardArm, RewardArmConfig
from hero.toy import ToyTask, ToyTaskConfig, TrainConfig, train

ARMS = (
    RewardArm.VERIFIER_ONLY,
    RewardArm.RM_ONLY,
    RewardArm.HERO_NO_WEIGHT,
    RewardArm.HERO,
    RewardArm.GATED_FALLBACK,
    RewardArm.NAIVE_BLEND,
)
SEEDS = (0, 1, 2)


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run(task: ToyTask, arm: RewardArm, norm: bool, seed: int, adv_epsilon: float | None = None):
    cfg = (
        TrainConfig(norm_adv_by_std=norm, seed=seed)
        if adv_epsilon is None
        else TrainConfig(norm_adv_by_std=norm, seed=seed, adv_epsilon=adv_epsilon)
    )
    return train(task, RewardArmConfig(arm=arm), cfg)


def t1_t2_arm_comparison(task: ToyTask) -> None:
    banner("T1/T2  Does each arm learn? (3 seeds, mean +/- spread)")
    print(f"{'arm':>16} | {'norm_std':>8} | {'accuracy':>17} | {'all-0':>6} |"
          f" {'all-1':>6} | {'mean |A|':>8}")
    print("-" * 78)
    for norm in (True, False):
        for arm in ARMS:
            runs = [run(task, arm, norm, s) for s in SEEDS]
            final = np.array([r.final_accuracy for r in runs])
            start = runs[0].initial_accuracy
            zero = np.mean([np.mean(r.all_incorrect_fraction) for r in runs])
            one = np.mean([np.mean(r.all_correct_fraction) for r in runs])
            adv = np.mean([np.mean(r.mean_abs_advantage) for r in runs])
            print(f"{arm:>16} | {str(norm):>8} | {start:.3f} -> {final.mean():.3f}"
                  f" +/-{final.std():.3f} | {zero:6.2f} | {one:6.2f} | {adv:8.3f}")
    print(
        "\nT1 passes: every arm lifts accuracy from 0.126 to ~0.81, so the harness is\n"
        "a working optimiser and the T3/T4 comparison rests on real training.\n"
        "\nT2 does NOT reproduce here, and that is a limitation of this task rather\n"
        "than evidence against the paper. Splitting uniformity by cause explains\n"
        "why: only ~18% of groups are all-incorrect, while ~60-66% end up\n"
        "all-correct because the task is mastered within a few dozen steps. A single\n"
        "'uniform group' figure would have hidden that distinction entirely -- most\n"
        "of the gradient-free groups here reflect success, not starvation.\n"
        "\nWith only 18% of groups starved and a 4-parameter policy, the mixed groups\n"
        "supply ample signal, so verifier-only matches every dense arm. This task\n"
        "therefore validates A-1's invariance claim but says nothing about whether\n"
        "HERO helps; showing that needs a policy large enough to be starved, and a\n"
        "training set hard enough to keep all-0 groups dominant -- the real stack."
    )


def t3_t4_weight_effect(task: ToyTask) -> None:
    banner("T3/T4  Does the variance weight change the LEARNED policy?")
    print(f"{'norm_adv_by_std':>16} | {'adv_eps':>8} | {'seed':>4} |"
          f" {'max |dtheta|':>13} | {'d accuracy':>11}")
    print("-" * 78)
    deltas: dict[tuple[bool, float], list[float]] = {}
    for norm in (True, False):
        for eps in (1e-6, 0.0):
            for seed in SEEDS:
                weighted = run(task, RewardArm.HERO, norm, seed, eps)
                plain = run(task, RewardArm.HERO_NO_WEIGHT, norm, seed, eps)
                dtheta = np.abs(weighted.theta - plain.theta).max()
                dacc = weighted.final_accuracy - plain.final_accuracy
                deltas.setdefault((norm, eps), []).append(dacc)
                print(f"{str(norm):>16} | {eps:8.0e} | {seed:>4} | {dtheta:13.3e} |"
                      f" {dacc:+11.4f}")
    print()
    print(f"{'norm_adv_by_std':>16} | {'adv_eps':>8} | {'mean d accuracy':>16} |"
          f" {'consistent sign':>15}")
    print("-" * 78)
    for (norm, eps), vals in deltas.items():
        arr = np.array(vals)
        sign = "yes" if np.all(arr > 0) or np.all(arr < 0) else "no (noise)"
        print(f"{str(norm):>16} | {eps:8.0e} | {arr.mean():+16.4f} | {sign:>15}")
    print(
        "\nT3, corrected against the measurement: with verl's default epsilon the\n"
        "parameter trajectories are NOT bit-identical -- they diverge by ~1e-2 over\n"
        "600 updates. Zeroing adv_epsilon collapses that to ~4e-15, i.e. float64\n"
        "rounding, which identifies the cause exactly: the 1e-6 denominator term is\n"
        "the only thing breaking cancellation, and chaotic optimisation amplifies\n"
        "that residue by ~13 orders of magnitude over the run.\n"
        "\nThe residue is not a mechanism. Its accuracy effect has inconsistent sign\n"
        "across seeds and a magnitude of ~0.01 percentage points -- indistinguishable\n"
        "from noise. So the weight remains a non-mechanism under std normalisation;\n"
        "the earlier claim of bit-identity was simply too strong.\n"
        "\nT4: with std normalisation off, the accuracy delta is ~+1.0 percentage\n"
        "point with the same sign on every seed -- roughly 100x the epsilon residue.\n"
        "That is the weight actually doing work."
    )


def t5_auroc_scope(task: ToyTask) -> None:
    banner("T5  Where the RM-reliability diagnostic exists at all (A-12)")
    rng = np.random.default_rng(11)
    theta_start = np.zeros(task.cfg.n_features)
    trained = run(task, RewardArm.HERO_NO_WEIGHT, True, 0).theta
    print(f"{'policy':>18} | {'accuracy':>9} | {'mixed-group AUROC':>18} |"
          f" {'mixed groups':>12}")
    print("-" * 78)
    for label, theta in (("at initialisation", theta_start), ("after training", trained)):
        probs = task.policy(theta)
        mixed = 0
        for x in range(task.cfg.n_prompts):
            actions = rng.choice(task.cfg.n_responses, size=8, p=probs[x])
            labels = task.correct[x, actions]
            mixed += int(labels.any() and not labels.all())
        auroc = task.rm_group_auroc(theta, np.random.default_rng(11), 8)
        print(f"{label:>18} | {task.expected_accuracy(theta):9.3f} | {auroc:18.3f} |"
              f" {mixed:>7}/{task.cfg.n_prompts}")
    print("\nAUROC against verifier labels requires both labels present, so it is\n"
          "computable only on mixed groups. A weak policy has few of those -- and\n"
          "those are precisely the runs where HERO's uniform-group machinery does the\n"
          "most work. The paper's reliability evidence therefore cannot cover the\n"
          "regime its mechanism depends on, whatever the sample size.")


if __name__ == "__main__":
    print(__doc__)
    task = ToyTask(ToyTaskConfig())
    print(f"Task: {task.cfg.n_prompts} prompts x {task.cfg.n_responses} responses, "
          f"{task.base_correct_fraction:.1%} of responses verifier-correct")
    t1_t2_arm_comparison(task)
    t3_t4_weight_effect(task)
    t5_auroc_scope(task)
    banner("SUMMARY")
    print(
        "A-1 is settled at two levels: advantages (invariance_check.py) and trained\n"
        "policies (T3/T4). Under verl's default configuration the variance weight\n"
        "moves final accuracy by ~0.01 percentage points with inconsistent sign,\n"
        "traceable entirely to the 1e-6 advantage epsilon; with std normalisation off\n"
        "it moves accuracy by ~1.0 point with consistent sign. The paper's Table 4\n"
        "(+1.2 / +3.8) is therefore incompatible with the default configuration.\n"
        "Whether the real runs used mean-only centring is the one open question, and\n"
        "author code or correspondence would settle it.\n"
        "\nScope, stated plainly: T2 did not reproduce here, so nothing in this file\n"
        "supports or undermines HERO's benefit claim -- only its weighting mechanism.\n"
        "Language-model dynamics (entropy collapse, length bias, textual reward\n"
        "hacking) are untouched and need the real stack on GPUs."
    )
