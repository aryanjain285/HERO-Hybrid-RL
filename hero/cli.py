"""Command-line entry points for the shell pipeline.

Kept out of ``hero/registry.py`` so that ``python -m hero.cli`` does not trigger
runpy's double-import warning, which would otherwise pollute every pipeline log.

    python -m hero.cli models --tier dev --field hf_id
    python -m hero.cli config --policy qwen3-1.7b --arm hero
"""

from __future__ import annotations

import argparse
import json

from hero.config import ExperimentConfig
from hero.registry import TIERS, format_tier
from hero.rewards import RewardArm, RewardArmConfig


def _models(args: argparse.Namespace) -> int:
    print(format_tier(args.tier, args.field))
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
    models.add_argument("--field", default="hf_id", choices=("hf_id", "key", "table"))
    models.set_defaults(func=_models)

    config = sub.add_parser("config", help="resolve a run config and its costs")
    config.add_argument("--name", default="dev-run")
    config.add_argument("--policy", default="qwen3-1.7b")
    config.add_argument("--reward-model", default="acemath-7b-rm")
    config.add_argument("--arm", default=str(RewardArm.HERO),
                        choices=[str(a) for a in RewardArm])
    config.add_argument("--seed", type=int, default=0)
    config.set_defaults(func=_config)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
