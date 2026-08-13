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
```

Both work on Linux and macOS with Python ≥3.11 and nothing else.
[`docs/pipeline.md`](docs/pipeline.md) covers the GPU stages.

## Layout

```
hero/                    Reward core and configuration (numpy only, no GPU)
  core.py                Stratified normalisation, variance weighting, GRPO advantage
  rewards.py             All six reward arms behind one dispatcher
  registry.py            Named model presets and compute tiers
  config.py              Hashable run definitions, step accounting, A-1 grid
  toy.py                 Complete GRPO trainer on a synthetic task
  cli.py                 Shell-facing entry points
analysis/
  invariance_check.py    A-1 / A-1b at the advantage level
  grpo_end_to_end.py     A-1 at the training level, plus A-12
scripts/                 setup / audit / model-fetch pipeline, strict mode
tests/                   208 property tests pinning every audit claim
docs/
  paper-audit.md         Deep read, verification pass, findings A-1 … A-20
  decisions.md           Decision log D-01 … D-11
  pipeline.md            How to run everything, and what is not automated
  FYP_PRD_HERO_Hybrid_Reward_RL.docx   Project requirements document (v1.1)
```

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
