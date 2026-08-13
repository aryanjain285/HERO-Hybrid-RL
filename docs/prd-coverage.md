# PRD coverage

Every requirement in PRD v1.1, with its implementation status. Statuses are
deliberately blunt: **done** means implemented and tested; **partial** means the
testable part exists and what remains is named; **blocked** means it requires
hardware or data this project does not yet have, with the blocker stated.

Nothing is marked done on the strength of a plan.

## §6.1 Stack and environment

| Requirement | Status | Where |
|---|---|---|
| Pinned RL framework (verl) | **blocked** | Needs Linux + CUDA. `scripts/setup.sh --with-gpu` installs it; the commit pin is the first M1 task, and audit A-1 turns on its `norm_adv_by_std_in_grpo` default |
| Rollout engine (vllm) | **blocked** | Same; no Windows/macOS wheels |
| Policy backbones (0.6B/1.7B/4B/8B) | **done** | `hero/registry.py`, all HF ids verified live against the Hub |
| Rule verifier | **done** | `hero/verifiers.py`, four implementations |
| Reward model (AceMath-7B-RM) | **partial** | Registered and size-verified (14.1 GB); serving needs a GPU |
| SFT trainer | **blocked** | GPU |
| Tracking + in-repo decision log | **done** | `docs/decisions.md`, D-01…D-11; every run writes a git-stamped manifest |

## §6.2 Data preparation

| Requirement | Status | Where |
|---|---|---|
| Benchmark loading | **done** | `hero/data.py`, MATH-500 |
| OpenMathReasoning 40k pool → 2k/2k/1k+1k splits | **not started** | Needs the training corpus; the split logic is small and follows the loader pattern |
| Decontamination against eval sets (D-06) | **not started** | Pure computation, no blocker — the honest gap |
| Content hashing of splits | **partial** | `ExperimentConfig.digest()` hashes configs; dataset hashing not yet wired |

## §6.3–6.5 SFT, reward manager, RM serving

| Requirement | Status | Where |
|---|---|---|
| SFT cold start | **blocked** | GPU |
| HERO reward mathematics | **done** | `hero/core.py`; 100+ property tests |
| All baseline/ablation arms | **done** | `hero/rewards.py`: verifier-only, RM-only, HERO, HERO-no-weight, naive blend, gated fallback (A-4) |
| Group-aware verl reward manager | **not started** | Deliberate: needs the pinned verl commit to bind against. Wrapping `compute_group_reward` is the whole job |
| Per-group telemetry | **partial** | `GroupOutcome` carries every required field including band degeneracy; the parquet writer is not built |

## §6.6–6.8 Decisions, compute, fidelity

| Requirement | Status | Where |
|---|---|---|
| Decision log D-01…D-09 | **done** | Extended to D-11 |
| Compute tiers and budgets | **done** | `hero/registry.py` tiers; real disk figures measured via `hero.cli check-models` |
| Step accounting | **done** | `ExperimentConfig` derives rollout batches, gradient steps, generations (audit A-13) |
| Fidelity checklist | **partial** | Config validation enforces clip ratios, KL, batch divisibility, band constraints; curve-shape acceptance needs training runs |

## §7 Milestones

| Milestone | Status |
|---|---|
| **M0 verifier study** | **done** — executed end to end on real MATH-500 problems with a GPT-4o judge; see `docs/m0-results.md` |
| M0 qualitative failure taxonomy (Table 10) | **done** — the six published cases are regression tests in `tests/test_verifiers.py` |
| M1 cold start + infra | **blocked** (GPU) |
| M2 baselines | **blocked** (GPU) |
| M3 HERO + audit experiments | **partial** — A-1 settled analytically and at optimiser level without a GPU; A-4 arm implemented; the LLM-scale 2×2 is blocked |
| M4 extension | **not started** |
| M5 headline runs | **blocked** (GPU) |

## §9 Evaluation protocol

| Requirement | Status | Where |
|---|---|---|
| Verifier-scored easy benchmarks | **partial** | MATH-500 loaded and scored; AMC/Minerva/Olympiad loaders not written |
| Judge-scored hard benchmarks | **done** | Paper's Figure 4 template verbatim, GPT-4o backend |
| ≥3 seeds on headline numbers | **partial** | Enforced in the toy trainer (3 seeds); LLM runs blocked |
| Bootstrap / binomial CIs (fixes A-6) | **done** | `hero/stats.py`; Wilson and paired bootstrap, verified against published z-values. Wired into the M0 table |
| Dual judges + agreement rate (fixes A-5) | **done** | `hero/judges.py`, `analysis/judge_agreement.py`; measured 82.8% [71.1, 90.4] |
| Manual audit of ≥50 judged items | **done** (sheet generated) | `runs/judge_audit.csv`, 60 rows with a `human_verdict` column. Filling it in is a human task |
| Verifier-scored numbers alongside judge-scored | **done** | M0 reports both per response |

## §10–§12

| Requirement | Status |
|---|---|
| Risk: verifier hangs/crashes | **done** — SymPy runs in a subprocess under timeout; infrastructure failure surfaces as `ERROR`, never silently as "incorrect" |
| Risk: judge cost | **done** — `OpenAIJudge` tracks tokens and reports cost ($0.13 for 61 calls) |
| Risk: reproduction diverges | **done** — qualitative acceptance criteria; every deviation in the decision log |
| Anti-goals | **done** — §6.3 of `paper-audit.md` adds six more derived from the audit |
| Open questions for supervisor | **done** — `paper-audit.md` §6.4, three sharpened |

## Honest summary

Closable without a GPU and **not yet done**: the OpenMathReasoning split builder,
decontamination (D-06), dataset content hashing, the telemetry parquet writer, and
the three remaining easy-eval benchmark loaders. These are the real gaps.

Genuinely **blocked on hardware**: everything involving policy-weight updates —
SFT, the verl reward manager binding, and every RL training run. An inference API
cannot substitute, because GRPO needs gradients.

What has been carried further than the PRD anticipated: audit A-1 is resolved at
two levels without a GPU, and findings A-10…A-20 did not exist when the PRD was
written.
