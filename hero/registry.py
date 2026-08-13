"""Named model presets, so switching a model is a string change.

Every entry records what the project needs to know before launching a run:
the HuggingFace id, how it is served, and its role. Sizes and context lengths
are the vendor-published values, kept here so compute planning does not depend
on recalling them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelRole(StrEnum):
    """What a model does in the pipeline."""

    POLICY = "policy"
    REWARD_MODEL = "reward_model"
    JUDGE = "judge"
    VERIFIER_LM = "verifier_lm"


class ServingMode(StrEnum):
    """How a model is executed."""

    TRAINED_IN_PROCESS = "trained_in_process"
    """Loaded by the trainer (verl/FSDP); not served over HTTP."""

    LOCAL_HTTP = "local_http"
    """Served locally by sglang or vllm behind an HTTP endpoint."""

    REMOTE_API = "remote_api"
    """Third-party API; costs money per call and needs a budget."""


@dataclass(frozen=True)
class ModelSpec:
    """A model the project can select by name.

    Args:
        key: Short registry name used in configs.
        hf_id: HuggingFace repo id, or the API model string for remote models.
        role: Pipeline role.
        serving: How it is executed.
        params_b: Parameter count in billions, for compute planning, or None
            when undisclosed (as for closed API models).
        notes: Why this model is in the registry.
        paper_default: True if the HERO paper used this model in this role.
    """

    key: str
    hf_id: str
    role: ModelRole
    serving: ServingMode
    params_b: float | None
    notes: str
    paper_default: bool = False

    def __post_init__(self) -> None:
        if self.params_b is not None and self.params_b <= 0.0:
            raise ValueError(f"{self.key}: params_b must be positive or None")
        if self.serving is ServingMode.REMOTE_API and self.params_b is not None:
            raise ValueError(
                f"{self.key}: remote API models have undisclosed parameter counts; "
                "use None rather than a placeholder"
            )


_SPECS: tuple[ModelSpec, ...] = (
    # ---- Policy backbones ------------------------------------------------- #
    ModelSpec(
        key="qwen3-0.6b",
        hf_id="Qwen/Qwen3-0.6B-Base",
        role=ModelRole.POLICY,
        serving=ServingMode.TRAINED_IN_PROCESS,
        params_b=0.6,
        notes="Smoke tier: pipeline shakeout on 2 GPUs. Not for science claims.",
    ),
    ModelSpec(
        key="qwen3-1.7b",
        hf_id="Qwen/Qwen3-1.7B-Base",
        role=ModelRole.POLICY,
        serving=ServingMode.TRAINED_IN_PROCESS,
        params_b=1.7,
        notes="Dev tier: primary science platform for all mechanism experiments.",
    ),
    ModelSpec(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B-Base",
        role=ModelRole.POLICY,
        serving=ServingMode.TRAINED_IN_PROCESS,
        params_b=4.0,
        notes="Headline tier: directly comparable to paper Table 2.",
        paper_default=True,
    ),
    ModelSpec(
        key="octothinker-8b",
        hf_id="OctoThinker/OctoThinker-8B-Hybrid-Base",
        role=ModelRole.POLICY,
        serving=ServingMode.TRAINED_IN_PROCESS,
        params_b=8.0,
        notes="Paper's second backbone (Table 3). Uses KL 0.001 and n=16, not 0/8.",
        paper_default=True,
    ),
    # ---- Reward models ---------------------------------------------------- #
    ModelSpec(
        key="acemath-7b-rm",
        hf_id="nvidia/AceMath-7B-RM",
        role=ModelRole.REWARD_MODEL,
        serving=ServingMode.LOCAL_HTTP,
        params_b=7.0,
        notes="Paper's RM. Table 8 shows the 72B variant adds nothing.",
        paper_default=True,
    ),
    ModelSpec(
        key="acemath-72b-rm",
        hf_id="nvidia/AceMath-72B-RM",
        role=ModelRole.REWARD_MODEL,
        serving=ServingMode.LOCAL_HTTP,
        params_b=72.0,
        notes="Paper's RM-scale ablation. Out of budget; listed for completeness.",
    ),
    ModelSpec(
        key="skywork-v2-8b",
        hf_id="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        role=ModelRole.REWARD_MODEL,
        serving=ServingMode.LOCAL_HTTP,
        params_b=8.0,
        notes="General-domain RM for the finance extension, where a math RM is "
        "out of distribution.",
    ),
    # ---- Judges and model-based verifiers --------------------------------- #
    ModelSpec(
        key="gpt-4o",
        hf_id="gpt-4o",
        role=ModelRole.JUDGE,
        serving=ServingMode.REMOTE_API,
        params_b=None,
        notes="Paper's hard-to-verify judge. Note Table 3 reports GPT-4.1 "
        "instead (audit A-7ii); pin one version across all runs.",
        paper_default=True,
    ),
    ModelSpec(
        key="qwen2.5-72b-judge",
        hf_id="Qwen/Qwen2.5-72B-Instruct",
        role=ModelRole.JUDGE,
        serving=ServingMode.LOCAL_HTTP,
        params_b=72.0,
        notes="Architecturally distinct second judge for the A-5 agreement study; "
        "becomes primary if API budget is denied.",
    ),
    ModelSpec(
        key="general-verifier",
        hf_id="TIGER-Lab/general-verifier",
        role=ModelRole.VERIFIER_LM,
        serving=ServingMode.LOCAL_HTTP,
        params_b=1.5,
        notes="Generative verifier measured in paper Table 1 and used as a "
        "baseline in Table 7.",
        paper_default=True,
    ),
)

_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}


def resolve(key: str, expected_role: ModelRole | None = None) -> ModelSpec:
    """Look up a model by registry key.

    Args:
        key: Registry key, e.g. ``"qwen3-1.7b"``.
        expected_role: If given, assert the model plays this role, so a config
            typo cannot silently put a judge in the policy slot.

    Raises:
        KeyError: Unknown key; the message lists the valid keys for that role.
        ValueError: Role mismatch.
    """
    if key not in _BY_KEY:
        available = ", ".join(sorted(_BY_KEY))
        raise KeyError(f"unknown model {key!r}; registered: {available}")
    spec = _BY_KEY[key]
    if expected_role is not None and spec.role is not expected_role:
        raise ValueError(
            f"model {key!r} has role {spec.role}, but {expected_role} was required"
        )
    return spec


def by_role(role: ModelRole) -> tuple[ModelSpec, ...]:
    """All registered models in a given role, registry order preserved."""
    return tuple(spec for spec in _SPECS if spec.role is role)


def all_specs() -> tuple[ModelSpec, ...]:
    """Every registered model."""
    return _SPECS


TIERS: dict[str, tuple[str, ...]] = {
    "smoke": ("qwen3-0.6b", "acemath-7b-rm"),
    "dev": ("qwen3-1.7b", "acemath-7b-rm", "general-verifier"),
    "headline": ("qwen3-4b", "acemath-7b-rm", "general-verifier"),
    "octothinker": ("octothinker-8b", "acemath-7b-rm"),
    "extension": ("qwen3-1.7b", "skywork-v2-8b"),
}
"""Model sets per compute tier, so a fetch stage names a tier, not a list.

Judges are excluded: ``gpt-4o`` is an API model with nothing to download, and the
open judge is only needed at evaluation time.
"""


def tier(name: str) -> tuple[ModelSpec, ...]:
    """Resolve a compute tier to its downloadable model specs."""
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; available: {', '.join(sorted(TIERS))}")
    return tuple(resolve(key) for key in TIERS[name])


def format_tier(name: str, field: str) -> str:
    """Render a tier for shell consumption or human reading.

    Args:
        name: Tier name.
        field: ``hf_id`` or ``key`` for one value per line; ``table`` for an
            aligned summary.
    """
    specs = tier(name)
    if field == "table":
        width = max(len(s.key) for s in specs)
        return "\n".join(
            f"{s.key:<{width}}  "
            f"{'unknown' if s.params_b is None else f'{s.params_b:g}B':>8}  "
            f"{s.role:<13} {s.hf_id}"
            for s in specs
        )
    if field not in ("hf_id", "key"):
        raise ValueError(f"unknown field {field!r}")
    return "\n".join(getattr(s, field) for s in specs)
