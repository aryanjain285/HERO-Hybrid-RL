# Pipeline

Clone, run one command, get the audit. Two stages need no GPU; the training stage
needs a Linux CUDA VM.

## Quick start

```bash
git clone https://github.com/aryanjain285/HERO-Hybrid-RL.git
cd HERO-Hybrid-RL
scripts/setup.sh            # venv, install, test suite     (~1 min, CPU)
scripts/run_audit.sh        # full audit with artefacts     (~1 min, CPU)
```

On a GPU VM, add the training stack and stage weights:

```bash
scripts/setup.sh --with-gpu           # + torch, vllm, verl   (Linux + CUDA only)
scripts/fetch_models.sh --tier dev    # Qwen3-1.7B, AceMath RM, general-verifier
```

## Stages

| Script | Needs | Time | Produces |
|---|---|---|---|
| `setup.sh` | Python ≥3.11 | ~1 min | `.venv`, editable install, passing tests |
| `setup.sh --with-gpu` | Linux, CUDA, NVIDIA driver | ~10 min | torch, vllm, verl |
| `run_audit.sh` | setup | ~1 min | `runs/audit-<stamp>/` logs, junit xml, manifest |
| `run_m0.sh` | setup, Ollama server | ~15 min | `runs/m0-<stamp>/` Table 1 metrics + labelled responses |
| `fetch_models.sh --tier T` | setup, disk | varies | weights under `$HF_HOME` |

### Milestone 0: the verifier study

```bash
scripts/run_m0.sh                                  # 30 problems x 2 samples
scripts/run_m0.sh --problems 100 --samples 3       # tighter confidence intervals
scripts/run_m0.sh --model qwen2.5:7b-instruct      # stronger generator
```

Needs a running Ollama server (`ollama serve`); the script pulls the model if
absent, because an implicit mid-run pull would stall the batch. Everything runs on
CPU — measured throughput on a machine with no discrete GPU was ~48 tok/s for
`qwen2.5:1.5b-instruct` and ~106 tok/s for the 0.5B.

Disk needed per tier, measured against the Hub with
`python -m hero.cli check-models --tier T`:

| tier | weights | recommended free |
|---|---|---|
| `smoke` | 15.3 GB | 19.9 GB |
| `dev` | 20.7 GB | 26.9 GB |
| `headline` | 25.3 GB | 32.9 GB |
| `octothinker` | 30.2 GB | 39.3 GB |
| `extension` | 18.5 GB | 24.0 GB |

Tiers are defined in `hero/registry.py` and listed with
`python -m hero.cli models --tier dev --field table`, so no script carries its own
copy of a model list. Available: `smoke`, `dev`, `headline`, `octothinker`,
`extension`.

## Conventions

- **Strict mode everywhere.** Each script sources `scripts/lib.sh`, which sets
  `set -euo pipefail`, so a failure aborts rather than continuing with half-built
  state.
- **Idempotent.** Re-running `setup.sh` reuses the venv; `fetch_models.sh` resumes
  partial downloads and skips complete ones.
- **Preflight before cost.** GPU presence, credentials, and package importability
  are checked before any download or paid API call.
- **Every result carries a manifest** with UTC timestamp, git SHA, dirty flag,
  host, and Python version. A result whose manifest says `git_dirty: yes` is not
  reproducible; commit first for anything that goes in the report.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HERO_VENV` | `./.venv` | Virtualenv location |
| `HERO_RUNS` | `./runs` | Artefact root |
| `HF_HOME` | `./.cache/huggingface` | Weight cache |
| `HF_TOKEN` | unset | Required for gated repositories |

## What is not automated yet

The GRPO training stage has no script, deliberately. It needs a group-aware verl
reward manager, which does not exist yet: verl's stock managers score samples
independently, whereas HERO's band normalisation and dispersion are group
statistics keyed on the prompt UID. Writing a launcher that calls a trainer that
is not there would produce a script that fails on a VM rather than one that works.

The order that unblocks it:

1. Pin a verl commit and re-verify `algorithm.norm_adv_by_std_in_grpo`'s default.
   Audit A-1 turns on it, so the pin is a scientific decision, not hygiene.
2. Implement the reward manager against that commit, wrapping
   `hero.rewards.compute_group_reward` rather than reimplementing it.
3. Add `scripts/run_smoke.sh` for a 50-step Qwen3-0.6B run with full telemetry.
4. Only then the dev-tier baselines.

Anti-goal from the PRD that applies here: do not launch long runs before smoke-tier
telemetry is verified.
