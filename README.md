# HERO-Hybrid-RL

Reproduction, audit, and extension of **HERO** (Hybrid Ensemble Reward
Optimization), ICLR 2026 — hybrid sparse-verifier / dense-reward-model RL for
reasoning. NTU B.Eng. Computer Science final year project.

Student: Aryan Jain · Supervisor: Asst Prof Sean (Xuefeng) Du · CCDS, NTU
Base paper: Tao et al., *Hybrid Reinforcement: When Reward Is Sparse, Better To Be
Dense* (arXiv 2510.07242). No official code release.

## Status

Pre-kickoff. The reward mathematics, experiment configuration, a working GRPO
trainer, and the audit of finding A-1 are implemented and tested. No LLM training
has been run: that needs a Linux CUDA VM, and the verl reward manager is the next
build step (see [`docs/pipeline.md`](docs/pipeline.md)).

**Headline audit result.** Under verl's default GRPO config
(`norm_adv_by_std_in_grpo: True`), both of HERO's advertised knobs are inert: the
variance-aware prompt weight cancels, and the band width `α` cancels inside uniform
groups — the very regime the method exists to serve. Confirmed twice, on advantages
and on trained policies: over 600 gradient updates the weight shifts final accuracy
by ~0.01 points with inconsistent sign (traceable entirely to the `1e-6` advantage
epsilon), versus a consistent **+0.99 points** with std normalisation off. The
paper's own ablations report effects the default configuration cannot produce.
See [`docs/paper-audit.md`](docs/paper-audit.md) §4.

## Quick start

```bash
scripts/setup.sh        # venv, install, tests   (~1 min, CPU only)
scripts/run_audit.sh    # full audit + artefacts (~1 min, CPU only)
scripts/run_m0.sh       # verifier study on real problems (~15 min, needs Ollama)
```

All CPU-only, on Linux or macOS with Python ≥3.11.
[`docs/pipeline.md`](docs/pipeline.md) covers the GPU stages.

## Layout

```
hero/                    Reward core and configuration (numpy only, no GPU)
  core.py                Stratified normalisation, variance weighting, GRPO advantage
  rewards.py             All six reward arms behind one dispatcher
  registry.py            Named model presets and compute tiers
  config.py              Hashable run definitions, step accounting, A-1 grid
  toy.py                 Complete GRPO trainer on a synthetic task
  verifiers.py           Rule-based math answer checkers (four, spanning the trade-off)
  llm.py                 Ollama client and the paper's judge template
  judges.py              Pluggable judges (Ollama / GPT-4o) plus agreement stats
  stats.py               Wilson and bootstrap confidence intervals
  data.py                MATH-500 loader
  study.py               Verifier-study metrics (Table 1 methodology)
  env.py                 Credential loading; never logs values
  cli.py                 Shell-facing entry points
analysis/
  invariance_check.py    A-1 / A-1b at the advantage level
  grpo_end_to_end.py     A-1 at the training level, plus A-12
  verifier_study.py      Milestone 0, end to end on real problems
  rescore_study.py       Re-score verifier changes against fixed labels, free
  judge_agreement.py     Dual-judge agreement and the manual audit sheet
scripts/                 setup / audit / M0 / model-fetch, strict mode throughout
data/                    GPT-4o-labelled M0 set and the audit sheet
tests/                   348 property tests pinning every audit claim
docs/
  paper-audit.md         Deep read, verification pass, findings A-1 … A-20
  m0-results.md          Milestone 0 results, with intervals and limitations
  prd-coverage.md        Every PRD requirement and its honest status
  decisions.md           Decision log D-01 … D-11
  pipeline.md            How to run everything, and what is not automated
  FYP_PRD_HERO_Hybrid_Reward_RL.docx   Project requirements document (v1.1)
```

The reward core imports only numpy. Heavier dependencies are confined to their own
modules and imported lazily, so `import hero` stays cheap and the audit stages run
without sympy, huggingface_hub, or a model server.

## Milestone 0 result

Executed end to end on real problems with a GPT-4o judge. Wilson 95% intervals,
because a point estimate at n=23 positives is not a claim:

| verifier | recall | precision |
|---|---|---|
| `raw_match` | 60.9 [40.8, 77.8] | 100.0 [78.5, 100.0] |
| `exact_match` | 95.7 [79.0, 99.2] | 100.0 [85.1, 100.0] |
| `symbolic` | 100.0 [85.7, 100.0] | 100.0 [85.7, 100.0] |

Literal matching loses 39% of genuinely correct answers at perfect precision — the
false-negative problem HERO exists to address. Dual-judge agreement between a local
1.5B judge and GPT-4o is 82.8% [71.1, 90.4], and every hand-inspected disagreement
favours GPT-4o. Full write-up and limitations in
[`docs/m0-results.md`](docs/m0-results.md).

For what is and is not implemented against the PRD, see
[`docs/prd-coverage.md`](docs/prd-coverage.md).

## Switching models and arms

Models are registry keys, and reward strategies are enum values, so an experiment
is a config change rather than a code change:

```python
from hero import ExperimentConfig, GrpoConfig, RewardArm, RewardArmConfig, a1_grid

dev = ExperimentConfig(name="dev-hero", policy="qwen3-1.7b", reward_model="acemath-7b-rm")

baseline = dev.with_(
    name="verifier-only",
    reward_model=None,
    reward=RewardArmConfig(arm=RewardArm.VERIFIER_ONLY),
)

mean_only = dev.with_(grpo=GrpoConfig(norm_adv_by_std_in_grpo=False))

for cfg in a1_grid(dev):          # the audit A-1 2x2
    print(cfg.name, cfg.digest(), cfg.total_gradient_steps)
```

Available arms: `VERIFIER_ONLY`, `RM_ONLY`, `HERO`, `HERO_NO_WEIGHT`,
`NAIVE_BLEND`, `GATED_FALLBACK` (the minimal hybrid the paper omits, audit A-4).
`hero.registry.by_role()` lists the models registered for each role.

## Conventions

- Every run is identified by `ExperimentConfig.digest()`, a content hash that
  ignores name and notes, so results tables regenerate from configs.
- Paper ambiguities are named fields carrying their decision-log ID, never
  literals buried in code.
- Claims in `docs/` are asserted in `tests/`. If a claim is not testable, it is
  labelled as a hypothesis.
