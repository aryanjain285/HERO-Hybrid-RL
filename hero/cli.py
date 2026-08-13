"""Command-line entry points for the shell pipeline.

Kept out of ``hero/registry.py`` so that ``python -m hero.cli`` does not trigger
runpy's double-import warning, which would otherwise pollute every pipeline log.

    python -m hero.cli models --tier dev --field hf_id
    python -m hero.cli config --policy qwen3-1.7b --arm hero
"""

from __future__ import annotations

import argparse
import json
import sys

from hero.config import ExperimentConfig
from hero.registry import TIERS, format_tier, resolve, tier
from hero.rewards import RewardArm, RewardArmConfig


def _use_lf_newlines() -> None:
    """Emit LF regardless of platform.

    This is a machine interface: shell stages capture the output with
    ``mapfile``, and Python's default newline translation appends CR on Windows,
    which then travels into a HuggingFace repo id and fails validation with an
    error that shows nothing wrong. Fixing it here covers every consumer rather
    than asking each caller to strip it.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")


def _models(args: argparse.Namespace) -> int:
    if args.key:
        print(getattr(resolve(args.key), args.field if args.field != "table" else "hf_id"))
        return 0
    print(format_tier(args.tier, args.field))
    return 0


def _check_models(args: argparse.Namespace) -> int:
    """Verify each model in a tier exists and is reachable, and size it.

    Queries repository metadata only, so it costs kilobytes rather than the tens
    of gigabytes a real fetch does. Run before provisioning a VM: it catches a
    renamed repo, a missing token for a gated one, and an undersized disk, which
    are the three failures that otherwise surface an hour into a download.

    Exits non-zero if any repository is unreachable.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        GatedRepoError,
        RepositoryNotFoundError,
    )

    api = HfApi()
    specs = tier(args.tier)
    width = max(len(s.hf_id) for s in specs)
    total_bytes = 0
    failures: list[str] = []

    for spec in specs:
        try:
            info = api.model_info(spec.hf_id, files_metadata=True)
        except GatedRepoError:
            failures.append(f"{spec.hf_id}: gated, needs an accepted licence and HF_TOKEN")
            print(f"{spec.hf_id:<{width}}  GATED")
            continue
        except RepositoryNotFoundError:
            failures.append(f"{spec.hf_id}: not found (renamed or private?)")
            print(f"{spec.hf_id:<{width}}  NOT FOUND")
            continue
        except Exception as exc:  # network, auth, rate limit
            failures.append(f"{spec.hf_id}: {type(exc).__name__}: {exc}")
            print(f"{spec.hf_id:<{width}}  ERROR {type(exc).__name__}")
            continue

        # Weight files only: tokenisers and configs are negligible, and counting
        # every sibling would include duplicate GGUF or ONNX exports.
        size = sum(
            f.size or 0
            for f in (info.siblings or [])
            if f.rfilename.endswith((".safetensors", ".bin"))
        )
        total_bytes += size
        print(f"{spec.hf_id:<{width}}  {size / 1e9:8.2f} GB  {len(info.siblings or [])} files")

    print(f"\n{'total':<{width}}  {total_bytes / 1e9:8.2f} GB")
    print(f"{'recommended free disk':<{width}}  {total_bytes / 1e9 * 1.3:8.2f} GB  (30% headroom)")

    if failures:
        print("\nunreachable:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


def _config(args: argparse.Namespace) -> int:
    """Emit a run's resolved config and derived costs as JSON.

    Lets a shell stage report what it is about to launch, and record the digest,
    without duplicating any of the accounting logic.
    """
    cfg = ExperimentConfig(
        name=args.name,
        policy=args.policy,
        reward_model=None if args.arm == RewardArm.VERIFIER_ONLY else args.reward_model,
        reward=RewardArmConfig(arm=RewardArm(args.arm)),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "name": cfg.name,
                "digest": cfg.digest(),
                "policy": cfg.policy,
                "reward_model": cfg.reward_model,
                "arm": str(cfg.reward.arm),
                "norm_adv_by_std_in_grpo": cfg.grpo.norm_adv_by_std_in_grpo,
                "rollout_batches": cfg.total_rollout_batches,
                "gradient_steps": cfg.total_gradient_steps,
                "total_generations": cfg.total_generations,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hero", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="list a compute tier's models")
    models.add_argument("--tier", default="dev", choices=sorted(TIERS))
    models.add_argument("--key", help="resolve a single registry key instead of a tier")
    models.add_argument("--field", default="hf_id", choices=("hf_id", "key", "table"))
    models.set_defaults(func=_models)

    check = sub.add_parser(
        "check-models", help="verify a tier's repos are reachable and size them"
    )
    check.add_argument("--tier", default="dev", choices=sorted(TIERS))
    check.set_defaults(func=_check_models)

    config = sub.add_parser("config", help="resolve a run config and its costs")
    config.add_argument("--name", default="dev-run")
    config.add_argument("--policy", default="qwen3-1.7b")
    config.add_argument("--reward-model", default="acemath-7b-rm")
    config.add_argument("--arm", default=str(RewardArm.HERO),
                        choices=[str(a) for a in RewardArm])
    config.add_argument("--seed", type=int, default=0)
    config.set_defaults(func=_config)

    args = parser.parse_args(argv)
    _use_lf_newlines()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
