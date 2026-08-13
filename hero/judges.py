"""Equivalence judges for hard-to-verify scoring.

Two backends behind one protocol, so the judge is a configuration choice:

* :class:`OllamaJudge` -- local, free, and weak. Adequate for pipeline work.
* :class:`OpenAIJudge` -- the paper's protocol (GPT-4o), and the reference for
  agreement statistics.

Audit A-5 requires reporting agreement between two architecturally distinct
judges rather than trusting either, which :func:`agreement` computes.

Credentials are read from the environment only. No key is ever accepted as an
argument, logged, or written to a manifest.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from hero.llm import EQUIVALENCE_TEMPLATE, JudgeVerdict, OllamaClient, _parse_decision

__all__ = [
    "Judge",
    "JudgeAgreement",
    "OllamaJudge",
    "OpenAIJudge",
    "agreement",
    "resolve_judge",
]

# Published USD per million tokens, for budget estimates only (PRD 9.3).
_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


class Judge(Protocol):
    """Structural type for equivalence judges."""

    name: str

    def preflight(self) -> None:
        """Raise if the judge cannot serve requests. Called before a batch."""

    def judge(self, question: str, ground_truth: str, student_answer: str) -> JudgeVerdict: ...


class OllamaJudge:
    """Local judge backed by an Ollama model."""

    def __init__(self, client: OllamaClient) -> None:
        self._client = client
        self.name = f"ollama:{client.config.model}"

    def preflight(self) -> None:
        """Verify the server is up and the model is already pulled."""
        self._client.require_model()

    def judge(self, question: str, ground_truth: str, student_answer: str) -> JudgeVerdict:
        return self._client.judge_equivalence(question, ground_truth, student_answer)


class OpenAIJudge:
    """GPT-4o-class judge over the Chat Completions API.

    Matches the paper's protocol: the template of Figure 4 at temperature 0. Token
    usage is accumulated so a run can report its own cost.

    Args:
        model: Model id; must be a chat model.
        api_key_env: Environment variable holding the key.
        max_tokens: Response cap. Generous enough that a model which reasons
            before answering still reaches its verdict.
        timeout_s: Per-request timeout.
        retries: Attempts per request.

    Raises:
        RuntimeError: If the key variable is unset.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 256,
        timeout_s: float = 120.0,
        retries: int = 3,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{api_key_env} is not set. Export it in the environment; do not "
                "pass keys as arguments or commit them."
            )
        self._key = key
        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Cost so far from published per-token rates, or 0.0 if unknown."""
        rates = _OPENAI_PRICING.get(self.model)
        if rates is None:
            return 0.0
        prompt_rate, completion_rate = rates
        return (
            self.prompt_tokens * prompt_rate + self.completion_tokens * completion_rate
        ) / 1_000_000

    def preflight(self) -> None:
        """Confirm the key is accepted, with one minimal request."""
        self.judge("1+1?", "2", "2")

    def judge(self, question: str, ground_truth: str, student_answer: str) -> JudgeVerdict:
        prompt = EQUIVALENCE_TEMPLATE.format(
            question=question, ground_truth=ground_truth, student_answer=student_answer
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode()
        last: Exception | None = None

        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                # 4xx other than rate limiting will not succeed on retry.
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise RuntimeError(
                        f"OpenAI rejected the request ({exc.code}): {exc.reason}"
                    ) from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(2.0 * attempt)
        else:
            raise RuntimeError(f"OpenAI request failed after {self.retries} attempts: {last}")

        usage = data.get("usage", {})
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.calls += 1

        choice = (data.get("choices") or [{}])[0]
        raw = (choice.get("message") or {}).get("content") or ""
        return JudgeVerdict(
            _parse_decision(raw), raw, truncated=choice.get("finish_reason") == "length"
        )


def resolve_judge(spec: str, *, host: str = "http://localhost:11434") -> Judge:
    """Build a judge from a spec string.

    Args:
        spec: ``openai:<model>`` or ``ollama:<model>``.
        host: Ollama host, ignored for OpenAI.

    Raises:
        ValueError: On an unrecognised backend.
    """
    backend, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"judge spec must be '<backend>:<model>'; got {spec!r}")
    if backend == "openai":
        return OpenAIJudge(model)
    if backend == "ollama":
        from hero.llm import OllamaConfig

        return OllamaJudge(OllamaClient(OllamaConfig(model=model, host=host, max_tokens=256)))
    raise ValueError(f"unknown judge backend {backend!r}; expected 'openai' or 'ollama'")


@dataclass(frozen=True)
class JudgeAgreement:
    """Pairwise agreement between two judges (audit A-5)."""

    both_labelled: int
    agreements: int
    primary_only_yes: int
    secondary_only_yes: int
    either_abstained: int

    @property
    def rate(self) -> float:
        """Percentage agreement over items both judges decided."""
        return 100.0 * self.agreements / self.both_labelled if self.both_labelled else 0.0


def agreement(
    primary: list[JudgeVerdict], secondary: list[JudgeVerdict]
) -> JudgeAgreement:
    """Compare two judges over the same responses, in order.

    Abstentions are excluded from the rate rather than counted as disagreement,
    and reported separately so a low rate cannot hide behind them.
    """
    if len(primary) != len(secondary):
        raise ValueError(f"length mismatch: {len(primary)} vs {len(secondary)}")

    both = agree = only_primary = only_secondary = abstained = 0
    for a, b in zip(primary, secondary):
        if a.abstained or b.abstained:
            abstained += 1
            continue
        both += 1
        if a.equivalent == b.equivalent:
            agree += 1
        elif a.equivalent:
            only_primary += 1
        else:
            only_secondary += 1
    return JudgeAgreement(both, agree, only_primary, only_secondary, abstained)
