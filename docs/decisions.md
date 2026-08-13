# Decision log

Ambiguities in the HERO paper resolved by project decision. No code was released,
so every entry here is a choice this project owns and must report. Cite the ID in
commit messages and run manifests; each has a revisit trigger.

| ID | Ambiguity | Decision | Revisit trigger |
|---|---|---|---|
| D-01 | GRPO advantage normalisation variant unstated | Pin explicitly per run via `GrpoConfig.norm_adv_by_std_in_grpo`; run both arms in the A-1 2×2 and adopt whichever reproduces the paper's weighting ablation | Author code release |
| D-02 | `σ_u` on raw or normalised RM scores | Raw (`HeroConfig.sigma_on_raw_rm=True`); normalised dispersion is capped by band width and near-meaningless | If raw-score scale drifts materially across training, switch to z-scored-within-batch (see A-16) |
| D-03 | `σ̄` running-mean rule | EMA, momentum 0.9, warm-started from the first batch mean. Reads frozen within a batch, so rewards cannot depend on group visit order under DP sharding | Sensitivity check at M3 |
| D-04 | Reward for unparseable, truncated, or timed-out responses | `r_rule = 0`; RM still scores the raw text; event counted separately in telemetry | If >5% of rollouts, investigate generation config |
| D-05 | Singleton band (`min = max`) | `z = 0` → pinned to band floor, faithful to Eq. 3 (`HeroConfig.singleton_z=0.0`). Note this under-credits rare-correct rollouts | Test `singleton_z=0.5` (band midpoint) if singleton rate >15% |
| D-06 | Decontamination unmentioned | n-gram + exact dedup against all six eval sets; overlaps reported | — |
| D-07 | `w_min`/`w_max`/`k` conflict (0.5/2.0/5 main text vs 0.4/3.0/6 appendix) | Appendix values; it reads as the actual run config | Sweep only if A-1 shows the weight is live |
| D-08 | Judge model for hard-to-verify evals | Primary GPT-4o with the paper's exact template; secondary open 70B-class judge for agreement stats | API budget review |
| D-09 | Training-time verifier named four ways across §4.1, Table 2 caption, App. A.1, App. A.2.2 | verl `math_verify` for training reward and data filtering (matches Table 2 caption and A.2.2); HF Math-Verify measured separately in M0 | Author code release |
| D-10 | Provenance of Table 1's ground-truth correctness labels is unstated (A-14) | Label all 750 HardVerify-Math responses with a strong judge, then hand-audit ≥100. Acceptance: positive base rate ≈ 51.6%, derived from Table 1's internal arithmetic | If base rate diverges >5 points, the protocol is wrong before the verifiers are |
| D-11 | Easy-eval protocol ambiguous: `N=8` candidates but "evaluate the first decoded output", over 8 seeds (A-20) | Report mean accuracy over all 8 candidates per problem, and state it explicitly alongside the seed count | If numbers sit systematically below the paper's, test first-candidate-only |

## Fixed by construction

Decisions now enforced in code rather than by discipline:

- **`α + β < 1`** — `HeroConfig` rejects overlapping bands at construction (A-15).
- **Role-checked model keys** — `ExperimentConfig` resolves every model against
  the registry with its expected role, so a judge cannot land in the policy slot.
- **Batch-frozen `σ̄`** — `RunningMeanDispersion.value` raises before warm-up and
  changes only at `end_batch()`.
- **Verifier labels strictly `{0,1}`** — verifier errors must be mapped upstream
  per D-04; a third label raises rather than being silently banded.
- **Step accounting** — `ExperimentConfig` derives rollout batches, gradient steps
  and total generations, so compute is never budgeted off a figure axis (A-13).
