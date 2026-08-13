"""Audit A-1 and A-1b: which HERO knobs actually reach the policy gradient.

Canonical GRPO standardises advantages within each prompt group,
A_i = (r_i - mean_g) / (std_g + eps), and that is verl's default
(``algorithm.norm_adv_by_std_in_grpo: True``, ``epsilon = 1e-6``). Standardisation
removes any positive affine rescaling of a group's rewards, which makes two of
HERO's mechanisms inert:

  A-1   the variance-aware prompt weight w(sigma_u) is one scalar per group;
  A-1b  the band half-width alpha is the slope of an affine map, because min-max
        normalisation forces z to span [0, 1] whatever alpha is -- and this bites
        precisely in uniform groups, where HERO claims to create gradient.

Both contradict the paper's own ablations (Table 4: +1.2/+3.8 from reweighting;
Figure 2b: 2-3 point swings from alpha), so the published runs cannot have used
canonical std-normalised GRPO. This script measures the residuals and quantifies
what changes with std normalisation off.

No GPU, no network. Run: python analysis/invariance_check.py
"""

from __future__ import annotations

import numpy as np

from hero.core import (
    VERL_ADV_EPSILON,
    HeroConfig,
    grpo_advantage,
    stratified_normalise,
    variance_weight,
)

RNG = np.random.default_rng(20260813)
GROUP_SIZE = 8  # paper Table 5: rollouts per prompt for the Qwen line
EASY = HeroConfig(alpha=0.05, beta=0.05)
MIXED_REGIME = HeroConfig(alpha=0.10, beta=0.10)
WIDE = HeroConfig(alpha=0.20, beta=0.20)


def make_group(n_correct: int, n: int = GROUP_SIZE, rm_spread: float = 2.0):
    """Synthetic rollout group; correct answers score higher on average."""
    r_rule = np.zeros(n, dtype=int)
    r_rule[:n_correct] = 1
    r_rm = np.where(
        r_rule == 1, RNG.normal(6.0, rm_spread, n), RNG.normal(2.0, rm_spread, n)
    )
    return r_rule, r_rm


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def e1_weight_invariance(trials: int = 4000) -> None:
    banner("E1  Does the variance-aware weight change any advantage?")
    print(f"{'norm_adv_by_std':>16} | {'max rel. shift':>14} | verdict")
    print("-" * 78)
    for norm in (True, False):
        worst = 0.0
        for _ in range(trials):
            r_rule, r_rm = make_group(int(RNG.integers(0, GROUP_SIZE + 1)))
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            w = variance_weight(r_rm.std(ddof=1), 2.0, EASY)
            a = grpo_advantage(r_hat, norm)
            b = grpo_advantage(w * r_hat, norm)
            worst = max(worst, np.abs(b - a).max() / max(np.abs(a).max(), 1e-12))
        print(f"{str(norm):>16} | {worst:14.3e} | "
              f"{'cancels (no-op)' if worst < 1e-3 else 'live (scales gradient)'}")
    print("\nWith verl's default the weight is float noise. With std normalisation\n"
          "off it rescales the group's advantages linearly, by w in [0.4, 3.0].")


def e2_epsilon_residual() -> None:
    banner("E2  Could the denominator epsilon explain the +3.8 point ablation?")
    print("A_i(w)/A_i(1) = w(sigma + eps) / (w*sigma + eps)")
    print(f"{'sigma_group':>12} | {'w':>5} | {'deviation from 1':>18}")
    print("-" * 78)
    for sigma in (0.03, 0.02, 0.005):
        for w in (0.4, 3.0):
            ratio = w * (sigma + VERL_ADV_EPSILON) / (w * sigma + VERL_ADV_EPSILON)
            print(f"{sigma:12.4f} | {w:5.1f} | {abs(ratio - 1):18.3e}")
    print("\nDeviations are O(1e-5) relative: five orders of magnitude too small.\n"
          "This candidate explanation is dead, not merely improbable.")


def e3_alpha_inert_in_uniform_groups(trials: int = 2000) -> None:
    banner("E3  Band width alpha inside uniform groups (the gradient-revival regime)")
    print(f"{'norm_adv_by_std':>16} | {'alpha pair':>13} | {'max rel. shift':>14} |"
          f" {'mean |A| ratio':>14}")
    print("-" * 78)
    for norm in (True, False):
        for wide in (MIXED_REGIME, WIDE):
            worst, ratios = 0.0, []
            for _ in range(trials):
                r_rule, r_rm = make_group(0)
                a = grpo_advantage(stratified_normalise(r_rule, r_rm, EASY), norm)
                b = grpo_advantage(stratified_normalise(r_rule, r_rm, wide), norm)
                worst = max(worst, np.abs(b - a).max() / max(np.abs(a).max(), 1e-12))
                ratios.append(np.abs(b).mean() / max(np.abs(a).mean(), 1e-12))
            print(f"{str(norm):>16} | {EASY.alpha:.2f} vs {wide.alpha:.2f} |"
                  f" {worst:14.3e} | {np.mean(ratios):14.4f}")
    print("\nStd normalisation on: alpha has no effect whatsoever (ratio 1.0000).\n"
          "Off: mean |A| scales exactly as the alpha ratio (2x, 4x). So under the\n"
          "default config the range ablation cannot act through uniform groups --\n"
          "which is the very channel the paper credits for the mixed-regime gain.")


def e4_alpha_live_in_mixed_groups(trials: int = 2000) -> None:
    banner("E4  Where alpha does survive std normalisation: mixed groups")
    print(f"{'n_correct/8':>12} | {'mean rel. shift':>16} | {'max rel. shift':>15}")
    print("-" * 78)
    for n_correct in (1, 4, 7):
        shifts = []
        for _ in range(trials):
            r_rule, r_rm = make_group(n_correct)
            a = grpo_advantage(stratified_normalise(r_rule, r_rm, EASY), True)
            b = grpo_advantage(stratified_normalise(r_rule, r_rm, WIDE), True)
            shifts.append(np.abs(b - a).max() / np.abs(a).max())
        print(f"{n_correct:>10}/8   | {np.mean(shifts):16.4f} | {np.max(shifts):15.4f}")
    print("\nalpha is NOT globally inert: in mixed groups, widening it from 0.05 to\n"
          "0.20 moves advantages by 15-33% on average, because it trades\n"
          "between-band separation against within-band ranking. So the range\n"
          "ablation is reproducible under std normalisation -- but only via mixed\n"
          "groups, contradicting the paper's stated explanation, which attributes\n"
          "the mixed-regime preference for larger alpha to all-0 groups (see E3).")


def e5_gradient_budget(trials: int = 4000) -> None:
    banner("E5  Gradient budget: how loudly does each group type speak?")
    print(f"{'group type':>22} | {'std-norm ON':>12} | {'std-norm OFF':>13}")
    print("-" * 78)
    for name, n_correct in (
        ("all-0 (uniform)", 0),
        ("1 of 8 correct", 1),
        ("4 of 8 correct", 4),
        ("all-1 (uniform)", GROUP_SIZE),
    ):
        on, off = [], []
        for _ in range(trials):
            r_rule, r_rm = make_group(n_correct)
            r_hat = stratified_normalise(r_rule, r_rm, EASY)
            on.append(np.abs(grpo_advantage(r_hat, True)).mean())
            off.append(np.abs(grpo_advantage(r_hat, False)).mean())
        print(f"{name:>22} | {np.mean(on):12.4f} | {np.mean(off):13.5f}")
    print("\nStd-norm ON: an all-incorrect group ranked purely by the RM produces\n"
          "advantages of the same magnitude as a group carrying a real verifier\n"
          "signal -- louder, in fact, than a 1-of-8 group. Standardisation divides\n"
          "out exactly the band structure meant to keep the RM subordinate, so the\n"
          "bounded-hackability property holds on rewards but not on advantages.\n"
          "\nStd-norm OFF: uniform groups stay ~20x quieter, and alpha is the dial\n"
          "setting that ratio. Under mean-only centring alpha's real semantics is\n"
          "the share of gradient budget granted to RM-ranked uniform groups, which\n"
          "explains the paper's ablation directions: data with more all-0 groups\n"
          "prefers larger alpha.")


def e6_ordering_condition() -> None:
    banner("E6  Property P1: the admissible region the paper does not state")
    print("P1 needs min(correct band) > max(incorrect band), i.e. alpha + beta < 1.\n")
    print(f"{'alpha':>7} | {'beta':>6} | {'incorrect':>16} | {'correct':>16} | P1")
    print("-" * 78)
    for a, b in ((0.05, 0.05), (0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (0.6, 0.6)):
        ok = a + b < 1.0
        print(f"{a:7.2f} | {b:6.2f} | [{-a:6.2f},{a:6.2f}] | [{1 - b:6.2f},{1 + b:6.2f}]"
              f" | {'holds' if ok else 'VIOLATED'}")
    print("\nThe paper permits alpha, beta in (0, 1], which admits overlap: at 0.6 a\n"
          "high-RM wrong answer outranks a low-RM right one. Published settings are\n"
          "safe; HeroConfig now rejects the rest at construction.")


def e7_weight_saturation() -> None:
    banner("E7  Is the logistic weight smooth in practice?")
    print("Eq. 4 takes k*(sigma_u - sigma_bar) in RAW RM units. AceMath scores run\n"
          "to ~40 during training (paper Fig. 6), so gaps of O(1) are ordinary.\n")
    span = EASY.w_max - EASY.w_min
    print(f"{'sigma_u - sigma_bar':>20} | {'w (k=6, appendix)':>18} | {'w (k=5, main text)':>19}")
    print("-" * 78)
    main_text = HeroConfig(w_min=0.5, w_max=2.0, w_slope=5.0)
    for gap in (-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0):
        print(f"{gap:20.2f} | {variance_weight(gap, 0.0, EASY):18.4f} |"
              f" {variance_weight(gap, 0.0, main_text):19.4f}")
    at_one = (variance_weight(1.0, 0.0, EASY) - EASY.w_min) / span
    print(f"\nAt a one-unit gap the weight has already travelled {at_one:.2%} of its\n"
          "range. With k=6 and raw-score dispersion, Eq. 4 is not a smooth weighting\n"
          "but a hard two-level gate at sigma_u = sigma_bar, taking w_min or w_max\n"
          "and little in between. Two consequences: the k sweep is nearly a no-op\n"
          "away from k~0, and because sigma_u is computed on raw scores while\n"
          "sigma_bar is an EMA over training, RM score inflation drives the gate\n"
          "toward w_max for everything -- so even where the weight is live, it\n"
          "decays into a constant, which cancels again.")


if __name__ == "__main__":
    print(__doc__)
    e1_weight_invariance()
    e2_epsilon_residual()
    e3_alpha_inert_in_uniform_groups()
    e4_alpha_live_in_mixed_groups()
    e5_gradient_budget()
    e6_ordering_condition()
    e7_weight_saturation()
    banner("SUMMARY")
    print(
        "Under verl's default GRPO config (norm_adv_by_std_in_grpo: True):\n"
        "  E1/E2  the variance-aware weight is a numerical no-op\n"
        "  E3     alpha is a no-op inside uniform groups\n"
        "  E4     alpha survives only through mixed groups, 15-33% of |A|\n"
        "  E5     uniform groups are amplified to full mixed-group gradient scale\n"
        "  E7     with k=6 the weight is a two-level gate, and RM drift flattens it\n"
        "\nTable 4 and Figure 2b both report effects this config cannot produce.\n"
        "Leading hypothesis: the runs used mean-only centring\n"
        "(norm_adv_by_std_in_grpo: False, Dr. GRPO style), which is consistent with\n"
        "the paper adopting DAPO's asymmetric clip (0.2, 0.28) and makes every\n"
        "published ablation direction mechanically explicable (E3, E5).\n"
        "\nThe algebraic half of A-1 is now settled at zero GPU cost. What remains is\n"
        "the training-level consequence: hero.config.a1_grid() defines the 2x2, and\n"
        "its std-on half is a falsifiable prediction of no difference."
    )
