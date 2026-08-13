"""Ollama client for local generation and LLM-as-judge scoring.

Uses the stdlib HTTP client so the audit stages stay dependency-light. Retries
transient failures, and returns judge abstentions as None rather than guessing,
so unparseable verdicts are visible in the results instead of silently counted.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = [
    "Generation",
    "JudgeVerdict",
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "EQUIVALENCE_TEMPLATE",
]

# Paper Appendix A.2.4, Figure 4, reproduced verbatim: the judge compares rather
# than re-solves, which limits leakage of the judge's own reasoning.
EQUIVALENCE_TEMPLATE = """### Question: {question}
### Ground Truth Answer: {ground_truth}
### Student Answer: {student_answer}

For the above question, please verify if the student's answer is equivalent to \
the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is \
equivalent to the ground truth answer.
If the student's answer is correct, output "Final Decision: Yes". If the \
student's answer is incorrect, output "Final Decision: No"."""


class OllamaError(RuntimeError):
    """The Ollama server was unreachable or returned an unusable response."""


@dataclass(frozen=True)
class OllamaConfig:
    """Connection and sampling settings.

    Args:
        model: Ollama model tag, e.g. ``qwen2.5:1.5b-instruct``.
        host: Server base URL.
        temperature: Sampling temperature; 0 for judging and data filtering.
        max_tokens: Generation cap (``num_predict``).
        timeout_s: Per-request timeout.
        retries: Attempts per request, including the first.
        backoff_s: Base delay between attempts, multiplied by attempt number.
    """

    model: str
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    retries: int = 3
    backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive; got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative; got {self.temperature}")
        if self.retries < 1:
            raise ValueError(f"retries must be at least 1; got {self.retries}")


@dataclass(frozen=True)
class Generation:
    """One completion and its cost."""

    text: str
    model: str
    tokens: int
    duration_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.duration_s if self.duration_s > 0 else 0.0


@dataclass(frozen=True)
class JudgeVerdict:
    """An equivalence judgement.

    Attributes:
        equivalent: True, False, or None when the judge did not emit a parseable
            decision. None is never coerced to False; it is reported separately.
        raw: The judge's full response, retained for the manual audit the
            protocol requires.
    """

    equivalent: bool | None
    raw: str

    @property
    def abstained(self) -> bool:
        return self.equivalent is None


class OllamaClient:
    """Minimal Ollama HTTP client."""

    def __init__(self, config: OllamaConfig) -> None:
        self.config = config

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.config.host.rstrip('/')}{path}"
        body = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.config.retries:
                    time.sleep(self.config.backoff_s * attempt)
        raise OllamaError(f"{url} failed after {self.config.retries} attempts: {last}")

    def is_available(self) -> bool:
        """Whether the server responds. Used for preflight, never for control flow."""
        try:
            url = f"{self.config.host.rstrip('/')}/api/version"
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def list_models(self) -> tuple[str, ...]:
        """Model tags available on the server."""
        url = f"{self.config.host.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"could not list models: {exc}") from exc
        return tuple(m["name"] for m in data.get("models", []))

    def require_model(self) -> None:
        """Raise unless the configured model is present.

        Ollama would otherwise attempt an implicit pull mid-run, turning a missing
        model into a multi-minute stall partway through a batch.
        """
        if not self.is_available():
            raise OllamaError(f"no Ollama server at {self.config.host}")
        available = self.list_models()
        if self.config.model not in available:
            raise OllamaError(
                f"model {self.config.model!r} not present; available: "
                f"{', '.join(available) or 'none'}. Run: ollama pull {self.config.model}"
            )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        """Complete a prompt."""
        options: dict[str, object] = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "num_predict": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        data = self._post(
            "/api/generate",
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
        )
        return Generation(
            text=data.get("response", ""),
            model=self.config.model,
            tokens=int(data.get("eval_count") or 0),
            duration_s=float(data.get("eval_duration") or 0) / 1e9,
        )

    def judge_equivalence(
        self, question: str, ground_truth: str, student_answer: str
    ) -> JudgeVerdict:
        """Score answer equivalence with the paper's compare-don't-solve template.

        Judging always runs at temperature 0: the protocol pins one judge
        configuration so that agreement statistics mean something.
        """
        prompt = EQUIVALENCE_TEMPLATE.format(
            question=question, ground_truth=ground_truth, student_answer=student_answer
        )
        raw = self.generate(prompt, temperature=0.0, max_tokens=64).text
        return JudgeVerdict(_parse_decision(raw), raw)


def _parse_decision(raw: str) -> bool | None:
    """Extract the judge's decision, or None if it never stated one."""
    lowered = raw.lower()
    marker = "final decision:"
    if marker in lowered:
        tail = lowered.split(marker, 1)[1].strip()
        if tail.startswith("yes"):
            return True
        if tail.startswith("no"):
            return False
    # Some small models drop the required prefix but still answer plainly.
    stripped = lowered.strip().strip(".*_ ")
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    return None
