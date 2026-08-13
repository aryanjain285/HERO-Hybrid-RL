# HERO: deep read and audit

Source: Tao, Kulikov, Saha, Wang, Xu, Li, Weston, Yu. *Hybrid Reinforcement: When
Reward Is Sparse, Better To Be Dense.* ICLR 2026 (arXiv 2510.07242 v3, 22 pp.
incl. appendices A–D). Read in full on 13 Aug 2026.

This document supersedes §4–5 of PRD v1.1. It records (1) a verification pass over
every number the PRD cites, (2) the standing audit findings and their status, and
(3) eleven new findings, one of which changes what the project should lead with.

**No code has been released.** arXiv v3 lists no repository, and the ICLR page
none either. Every implementation detail below is inferred from the text or
resolved by project decision; nothing can be checked against author code.

---

## 1. The mechanism, precisely

For each prompt, `N` rollouts are scored twice: `r_rule ∈ {0,1}` by a rule
verifier and `r_RM ∈ ℝ` by a reward model.

**Stratified normalisation (Eq. 3).** Partition rollouts by verifier label,
min–max normalise RM scores *within* each partition, map into disjoint bands:

```
z    = (r_RM − min r_RM) / (max r_RM − min r_RM + ε)      # within partition
r̂    = −α + 2α·z            if r_rule = 0                 # band [−α, +α]
r̂    = (1−β) + 2β·z         if r_rule = 1                 # band [1−β, 1+β]
```

**Variance-aware weighting (Eq. 4–5).** With `σ_u` the standard deviation of RM
scores across the group and `σ̄` a running mean:

```
w(σ_u)  = w_min + (w_max − w_min)·sigmoid(k·(σ_u − σ̄))
r_final = w(σ_u)·r̂
```

Three properties are claimed or implied. **P1** ordering preservation: a correct
rollout always outscores an incorrect one. **P2** gradient revival: in all-0 or
all-1 groups, where binary GRPO gives exactly zero advantage, intra-band RM
variation restores a nonzero gradient. **P3** bounded hackability: the RM can move
a reward by at most `2α` (or `2β`). P3 is only implicit in the paper.

Two structural facts follow immediately, neither stated by the authors, and both
load-bearing for §4:

- **Min–max forces full band occupancy.** By construction the group minimum maps
  to `z = 0` and the maximum to `z ≈ 1`, so every non-degenerate group spans the
  *entire* band, whatever `α` is and however weakly the RM discriminates. The
  realised spread carries no information about RM confidence.
- **`r̂` is invariant to positive affine transforms of `r_RM`** (up to `O(ε/range)`;
  bounded exactly in `tests/test_core.py`). RM *calibration* therefore cannot
  reach the policy through `r̂` — only within-band ranking and relative spacing
  can. This is a clean mechanistic explanation for the paper's own null result
  when swapping the 7B RM for the 72B one (Table 8: 62.0 → 62.8, and 73.2 → 71.4).
  `σ_u` is the sole scale-sensitive term in the whole method.

---

## 2. Verification pass

Every paper-sourced number in PRD v1.1 was re-checked against the PDF. **All
verified correct**, including Table 1 in full, Table 2 in full, the Table 5/6
hyperparameters, the four ablation groups, the AUROC statistics, and the
reward-hacking trajectory. Two items the PRD reports as inconsistencies are
confirmed as genuine, and one more is added:

| PRD claim | Status |
|---|---|
| `w_min/w_max/k` = 0.5/2.0/5 (main text) vs 0.4/3.0/6 (App. A.1) | Confirmed, both verbatim |
| Judge is GPT-4o (protocol, Table 2) but Table 3 caption says "GPT4.1 judges results" | Confirmed |
| Verifier named three ways | Confirmed, and it is **three** distinct namings: §4.1 "math verify (library) in the VERL repo", Table 2 caption "math verify (verl)", App. A.1 "the HuggingFace math verify library", App. A.2.2 "math verifier (verl)" |

**Two Table 5 settings PRD v1.1 omits**, both material:

- `use_dynamic_bsz: True` — token-based micro-batching. Interacts with
  `loss_agg_mode`, so it must be pinned before any baseline comparison.
- `filter_overlong_prompts: True` — silently drops prompts over 1024 tokens,
  changing the effective training set size relative to the nominal 2,000.

**A derived fact worth having for M0.** Table 1 is internally consistent with a
labelled set of ~387 positives out of 750 responses (base rate 51.6%). Solving
the `math_verify (verl)` row (recall 68.4, precision 100.0, FPR 0.0, acc. 83.7)
for the positive count gives 387; substituting that into the `math_reward` row
predicts precision 97.3 and accuracy 53.5 against the published 97.5 and 53.6,
and into the `AceMath @ thr 1` row predicts 67.8 and 73.3 against 67.7 and 73.2.
Three independent rows agree, so Table 1 is arithmetically sound and the M0
harness has a concrete target: recover a ~52% positive base rate on the 750
responses. If M0's base rate diverges, the labelling protocol is wrong, not the
verifiers.

---

## 3. Standing findings A-1 … A-9

A-2, A-3, A-4, A-5, A-6, A-7, A-8, A-9 all stand as written in the PRD. A-8's ε
critique is confirmed verbatim: the paper says ε is "set to relatively small value
so that the training dynamics are primarily led by rule-based rewards", but ε only
guards division by zero; the rule-vs-RM balance is governed entirely by `α`, `β`.
Two refinements:

- **A-2 is stronger than stated** — see A-12 below, which shows the paper's
  reliability evidence does not cover the regime the mechanism depends on.
- **A-8 singleton case, sharpened.** A lone correct rollout gets `z = 0/ε = 0`,
  pinning it to `1−β`, the *floor* of the correct band, while a group with several
  correct rollouts awards its best `1+β`. Rare-correct prompts — the hardest and
  most informative ones — are therefore systematically under-credited relative to
  easy ones. The band midpoint (`z = 0.5`) is the defensible default;
  `HeroConfig(singleton_z=...)` makes it a one-line switch, defaulting to 0.0 for
  faithfulness.

---

## 4. A-1, settled algebraically and numerically

This is the project's most valuable finding and it is now resolved to the point
where only its training-level consequence remains open.

### 4.1 The setup

verl's GRPO advantage is `A_i = (r_i − mean_g) / (std_g + ε)` with
`algorithm.norm_adv_by_std_in_grpo` **defaulting to `True`** and `epsilon = 1e-6`
(confirmed against `verl/trainer/ppo/core_algos.py` and
`verl/trainer/config/ppo_trainer.yaml`). Group standardisation removes any positive
affine rescaling of a group's rewards. Two HERO knobs are exactly that.

### 4.2 A-1: the variance weight cancels

`w` is one scalar per group, applied to all `N` rollouts, so it scales both
`mean_g` and `std_g` and divides out. Measured worst-case relative shift over
4,000 random groups × 4 weight values: **5.1e-5** — pure float noise from the
`1e-6` denominator. With `norm_adv_by_std_in_grpo=False`, the same weight scales
advantages linearly by up to 3×.

The PRD listed a numerical-ε residual as candidate explanation (3). It is now
**dead, not merely improbable**: in closed form
`A_i(w)/A_i(1) = w(σ+ε)/(wσ+ε)`, which at realistic group dispersion
(`σ ≈ 0.02–0.03` for `α = 0.05`) deviates from 1 by `O(1e-5)` — five orders of
magnitude too small to move a benchmark average by 3.8 points.

### 4.3 A-1b (new): the band width is inert in uniform groups

Because min–max forces `z` to span `[0,1]`, in a uniform group
`r̂ = −α + 2α·z` is affine in `z` with slope `2α`, and standardisation removes it:

```
A_i = 2α(z_i − z̄) / (2α·std(z) + ε)  ≈  (z_i − z̄)/std(z)     — α cancels
```

Measured: changing `α` from 0.05 to 0.10 or 0.20 on identical rollouts leaves
uniform-group advantages **bit-identical** (mean |A| ratio 1.0000, worst shift
2.8e-5). With std normalisation off, mean |A| scales as exactly 2× and 4×.

This is sharper than A-1 because uniform groups are the whole point of the
method. Under verl's default config, the band width — HERO's primary
hyperparameter — does nothing in the regime HERO exists to serve.

`α` is *not* globally inert: in mixed groups it trades between-band separation
against within-band ranking, shifting advantages by a measured **15–33%** on
average when widened 0.05 → 0.20. So the range ablation is reproducible under the
default config, but only through mixed groups — which contradicts the paper's
stated explanation, that mixed-regime data prefers larger `α` because "many
samples fail the rule-based verifier", i.e. via all-0 groups.

### 4.4 A-17 (new): standardisation erases bounded hackability

Mean `|A|` by group composition at `α = β = 0.05`, 4,000 groups each:

| group | std-norm ON (verl default) | std-norm OFF |
|---|---|---|
| all-0 (uniform) | 0.773 | 0.027 |
| 1 of 8 correct | 0.616 | 0.208 |
| 4 of 8 correct | 0.933 | 0.500 |
| all-1 (uniform) | 0.773 | 0.027 |

Under the default, an all-incorrect group ranked *purely by the reward model*
speaks as loudly as a group carrying a genuine verifier signal — louder than a
1-of-8 group. Standardisation divides out precisely the band structure that was
meant to keep the RM subordinate to the verifier. **P3 bounded hackability holds
at the reward level and evaporates at the advantage level.** Formalising this
distinction is a real, small, publishable result, and it replaces the PRD's §8.2c
plan with something more pointed than "prove P3".

With std normalisation off, the hierarchy is restored: uniform groups stay ~20×
quieter, and `α` is the dial that sets the ratio. This reframes the band width:
under mean-only centring, **`α` is not a safety margin on correctness, it is the
share of the gradient budget granted to RM-ranked uniform groups.** That reading
explains the paper's ablation directions — data with more all-0 groups wants
larger `α` — which the std-normalised reading cannot.

### 4.5 A-16 (new): the logistic weight is a two-level gate

Eq. 4 evaluates `k·(σ_u − σ̄)` in *raw RM units*. AceMath raw scores reach ~40
during training (paper Fig. 6), so per-group dispersion gaps of `O(1)` are
ordinary. At `k = 6` a one-unit gap has already traversed **99.75%** of the weight
range:

| `σ_u − σ̄` | `w` (k=6, appendix) | `w` (k=5, main text) |
|---|---|---|
| −1.0 | 0.406 | 0.510 |
| −0.1 | 1.321 | 1.066 |
| 0.0 | 1.700 | 1.250 |
| +0.1 | 2.079 | 1.434 |
| +1.0 | 2.994 | 1.990 |

So Eq. 4 is not a smooth bounded weighting but a hard gate at `σ_u = σ̄` taking
`w_min` or `w_max` and little between. Two consequences: sweeping `k` is nearly a
no-op away from `k ≈ 0`, and because `σ_u` is raw-scale while `σ̄` is an EMA over
training, the RM score inflation the paper documents drives the gate toward
`w_max` for everything — so even where the weight is live it decays toward a
constant, which cancels again.

### 4.6 Conclusion and the one experiment left

Table 4 (+1.2/+3.8 from reweighting) and Figure 2b (2–3 point swings from `α`)
both report effects that verl's default config **cannot** produce through the
channels the paper describes. The leading hypothesis is that the runs used
mean-only centring (`norm_adv_by_std_in_grpo: False`, Dr. GRPO style), which is
consistent with the paper adopting DAPO's asymmetric clip `(0.2, 0.28)` and which
makes every published ablation direction mechanically explicable.

Reproduce with `analysis/invariance_check.py`; the claims are pinned as tests in
`tests/test_core.py::TestAuditA1WeightInvariance` and `::TestAuditA1bAlphaInvariance`.

What remains is the training-level consequence, not the algebra:
`hero.config.a1_grid()` defines the 2×2 `{std on, off} × {weight on, off}`. Its
std-on half is a **falsifiable prediction: those two arms should be statistically
indistinguishable.** If they differ, something in the pipeline is not what this
analysis assumes, and finding it is itself the result.

**The design rule this yields** — and the transferable lesson for the FYP —
is: *difficulty or reliability scaling must be applied to the advantage, or to the
prompt sampling distribution, never as a reward multiplier, because group-relative
objectives cancel reward multipliers exactly.*

---

## 5. New findings A-10 … A-20

| ID | Finding | Consequence |
|---|---|---|
| **A-10** | Every ablation table reports "Hard-to-verify" as **HVM alone** (73.2), not the HVM/TBR average that Table 2 calls hard-eval (66.3). Confirmed across Tables 4, 8, 9 and Fig. 2. | Ablation deltas must be compared against HVM only. Mixing the two axes silently invents a 7-point discrepancy. |
| **A-11** | Fig. 2(a) "None" scores 59.4/62.2, but Table 2's verifier-only arm on the same data scores 58.3/61.0. A no-dense-reward arm should *be* the verifier-only baseline. | Either "None" retains variance weighting (making it "HERO minus stratification", and implying weighting alone is worth ≈+1.1/+1.2), or the arms come from different runs. Ablation baseline is ambiguous; do not treat 59.4 as the verifier-only number. |
| **A-12** | The AUROC reliability study (App. B.1, Fig. 7) is computed **on mixed groups only** — necessarily, since AUROC against verifier labels is undefined when all labels agree. Its scope is one training step of the verifiable run (250 groups). | The paper's *only* evidence that RM ranking is trustworthy cannot, by construction, cover uniform groups — the exact regime HERO's gradient revival operates in. This is the strongest form of A-2 and the core justification for the calibration-gated extension. |
| **A-13** | 2,000 prompts at batch 512 for 20 epochs is 60–78 rollout batches, but Fig. 5 plots 300 steps and Fig. 6 plots 350. The figures match *gradient* steps (78 × 4 = 312). | Reading the x-axis as rollout batches overstates a run ~4×. Encoded and tested in `hero.config` step-accounting properties. |
| **A-14** | Table 1 needs ground-truth correctness for 750 responses, but the paper never states how those labels were obtained. Table 10 shows `o3` used as a judge elsewhere. | M0 is not reproducible without a labelling protocol. New decision D-10; the 51.6% base rate derived in §2 is the acceptance check. |
| **A-15** | P1 requires `α + β < 1`, but Eq. 3 permits `α, β ∈ (0,1]`. At `α = β = 0.6` bands overlap and a high-RM wrong answer outranks a low-RM right one. | Published settings are safe. Now rejected at construction by `HeroConfig`. |
| **A-16** | Weight saturation and drift (§4.5). | `k` sweeps are near-vacuous; `σ̄` must be tracked against RM drift. |
| **A-17** | Standardisation erases bounded hackability (§4.4). | Replaces the "prove P3" plan with a sharper reward-vs-advantage distinction. |
| **A-18** | The Table 7 discussion states "HERO attains an average score of 62.0, outperforming General Reasoner (58.4) and Qwen2.5-7B-Instruct (62.5)" — comparing HERO's *easy*-eval average against the other two systems' *hard*-eval averages, and claiming to outperform 62.5 with 62.0. The same passage cites "Table 2" for Table 7's results. Table 7's caption also claims pass@1 over 8 seeds for hard-to-verify columns, contradicting the `N=1` judge protocol. | On the correct columns HERO does win (62.0 vs 58.6 and 58.1). The claim survives; the prose does not. Quote Table 7 directly, never this passage. |
| **A-19** | Fig. 6's caption is a verbatim duplicate of Fig. 7's, describing AUROC panels while the figure shows reward-mean and MATH500 accuracy. | Cite the reward-hacking evidence by figure content, not caption. |
| **A-20** | Easy-eval protocol reads "generate `N=8` candidates per problem, and evaluate the first decoded output (pass@1)", averaged over 8 seeds. Generating 8 candidates and scoring only the first is either redundant or shorthand for averaging over the 8. | Ambiguity of a factor of 8 in effective eval samples, which directly affects the variance A-6 is concerned with. New decision D-11: report mean over all 8 candidates and state it. |

---

## 6. What this changes about the plan

### 6.1 Lead with mechanism, not domain transfer

PRD v1.1 recommends FinHERO (Track 1) as the extension, with the A-1 experiment
executed regardless. **Recommendation: swap the ranking.** Track 2 is now
substantially de-risked — three of its findings are already established (§4), with
executable evidence and tests — while FinHERO's principal risk (dataset
construction) is untouched. Leading with mechanism means the project has a real
result banked before Semester 2 begins, and finance becomes the validation domain
if time allows rather than the load-bearing contribution.

The concrete proposal the audit points to, replacing both inert knobs with one
mechanism that cannot cancel:

> **Calibration-gated advantage scaling.** Maintain an online estimate of RM
> reliability from mixed groups (where AUROC against verifier labels is defined),
> transfer it to uniform groups via a prompt-difficulty or embedding-similarity
> proxy, and scale uniform-group *advantages* — post-standardisation — in
> proportion to estimated reliability. Verifier-signalled groups keep their full
> weight.

This attacks A-2, A-12, A-16 and A-17 with a single change; is provably live under
both normalisation settings, because it acts after standardisation; and reduces to
HERO-with-mean-only-centring when the reliability estimate is constant, which
makes for a clean ablation ladder. The pre-registered hypothesis is that it
dominates fixed-`α` HERO on hard-to-verify training data, where uniform groups are
most frequent and RM reliability most variable.

### 6.2 Do first, in order

1. **M0 unchanged, plus a labelling protocol (D-10).** Reproduce Table 1 on the
   750 HardVerify-Math responses. Acceptance gains a check: base rate ≈ 51.6%.
2. **Pin `norm_adv_by_std_in_grpo` explicitly in every config**, and record it in
   the run manifest. It is the single most consequential unstated setting in the
   paper. Already surfaced as a first-class field in `hero.config.GrpoConfig`.
3. **Log the telemetry that makes the audit free**: per group, `(uid, r_rule,
   r_rm_raw, r_hat, σ_u, w, band occupancy, singleton count)` plus realised
   advantage magnitudes split by uniform vs mixed. Without the advantage split,
   A-17 cannot be measured after the fact.
4. **Track RM score drift against `σ̄`** from the first run, per A-16.
5. **Then** the A-1 2×2 at dev tier, and the A-4 gated-fallback arm — which is
   already implemented and tested as `RewardArm.GATED_FALLBACK`, so it costs a
   config line rather than an implementation.

### 6.3 Additions to the anti-goals

The PRD's eight anti-goals stand. Add:

- **Do not tune ε.** It guards division by zero. Raising it shrinks the realised
  band by `range/(range+ε)` and nothing else useful.
- **Do not sweep `k` expecting a smooth response.** A-16: it is a gate. If a
  smooth difficulty response is wanted, normalise `σ_u` first.
- **Do not compare ablation numbers against the hard-eval average.** A-10: the
  ablation axis is HVM alone.
- **Do not quote the Table 7 discussion passage.** A-18: it compares mismatched
  columns. Quote the table.
- **Do not budget from a 300-step reading of Fig. 5.** A-13: those are gradient
  steps; the run is ~78 rollout batches.
- **Do not rely on P3 as a training-time safety property.** A-17: it does not
  survive standardisation.

### 6.4 Questions for the supervisor, revised

The PRD's six questions stand. Three are now sharper:

1. **A-1 is largely settled on paper. Is that the FYP's headline?** It is a
   negative-plus-reframing result about a just-published FAIR method. Worth
   confirming this is viewed as a contribution rather than a criticism, and
   whether the authors should be contacted for the config — the group's proximity
   to them cuts both ways.
2. **Given no code release, is a faithful reproduction still the right anchor?**
   Every ambiguity is now a project decision. The decision log becomes a
   first-class deliverable rather than an appendix.
3. **Does the mechanism-first reordering (§6.1) have supervisor support**, with
   finance demoted to validation?

---

## 7. Reproducibility of this document

- `analysis/invariance_check.py` regenerates every measured number in §4.
- `python -m pytest tests/ -q` asserts each claim (121 tests).
- Paper numbers in §2 were transcribed from a full-text extraction of the ICLR
  PDF; the Table 1 consistency derivation is arithmetic, reproduced inline.
- verl behaviour was checked against `main` at read time. Pin the commit before
  M1 and re-verify `norm_adv_by_std_in_grpo`'s default, since it is the hinge of §4.
