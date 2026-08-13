"""Tests for judge backends, credential loading, and agreement statistics.

No network is used: the OpenAI backend is exercised against a stubbed transport,
and the local backend against a stub client.
"""

from __future__ import annotations

import json
import os
import urllib.error
from io import BytesIO

import pytest

from hero.env import load_env_file
from hero.judges import (
    JudgeAgreement,
    OllamaJudge,
    OpenAIJudge,
    agreement,
    resolve_judge,
)
from hero.llm import JudgeVerdict


def _openai_response(content: str, finish: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    ).encode()


class _FakeHTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


class TestEnvLoading:
    def test_loads_and_reports_names_only(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text('OPENAI_API_KEY="sk-secret"\nexport OTHER=plain\n# comment\n')
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OTHER", raising=False)
        names = load_env_file(path)
        assert set(names) == {"OPENAI_API_KEY", "OTHER"}
        assert os.environ["OPENAI_API_KEY"] == "sk-secret"
        assert os.environ["OTHER"] == "plain"

    def test_does_not_override_by_default(self, tmp_path, monkeypatch):
        """An already-set variable wins, so a stale file cannot silently shadow it."""
        path = tmp_path / ".env"
        path.write_text("TOKEN=from_file\n")
        monkeypatch.setenv("TOKEN", "from_env")
        assert load_env_file(path) == ()
        assert os.environ["TOKEN"] == "from_env"

    def test_override_when_requested(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("TOKEN=from_file\n")
        monkeypatch.setenv("TOKEN", "from_env")
        assert load_env_file(path, override=True) == ("TOKEN",)
        assert os.environ["TOKEN"] == "from_file"

    def test_ignores_blank_and_malformed_lines(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("\n\n#only a comment\nNOEQUALS\nGOOD=1\n")
        monkeypatch.delenv("GOOD", raising=False)
        assert load_env_file(path) == ("GOOD",)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="env file not found"):
            load_env_file(tmp_path / "absent")


class TestOpenAIJudge:
    def test_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="not set"):
            OpenAIJudge()

    def test_key_is_never_exposed_in_repr_or_name(self, api_key):
        judge = OpenAIJudge("gpt-4o")
        assert "sk-test" not in repr(judge)
        assert "sk-test" not in judge.name

    def test_parses_a_yes(self, api_key, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeHTTPResponse(_openai_response("Final Decision: Yes")),
        )
        judge = OpenAIJudge("gpt-4o")
        verdict = judge.judge("q", "42", "42")
        assert verdict.equivalent is True
        assert not verdict.truncated

    def test_flags_truncation(self, api_key, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeHTTPResponse(_openai_response("thinking...", "length")),
        )
        verdict = OpenAIJudge("gpt-4o").judge("q", "42", "42")
        assert verdict.abstained
        assert verdict.truncated

    def test_accumulates_usage_and_cost(self, api_key, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeHTTPResponse(_openai_response("Final Decision: No")),
        )
        judge = OpenAIJudge("gpt-4o")
        for _ in range(3):
            judge.judge("q", "1", "2")
        assert judge.calls == 3
        assert judge.prompt_tokens == 300
        assert judge.completion_tokens == 60
        # 300 prompt at $2.50/M plus 60 completion at $10.00/M.
        assert judge.estimated_cost_usd == pytest.approx((300 * 2.5 + 60 * 10) / 1e6)

    def test_unknown_model_costs_zero_rather_than_guessing(self, api_key, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeHTTPResponse(_openai_response("Final Decision: Yes")),
        )
        judge = OpenAIJudge("gpt-9-unreleased")
        judge.judge("q", "1", "1")
        assert judge.estimated_cost_usd == 0.0

    def test_client_errors_are_not_retried(self, api_key, monkeypatch):
        """A 401 will not fix itself; retrying wastes time and money."""
        calls = []

        def raise_401(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", raise_401)
        with pytest.raises(RuntimeError, match="401"):
            OpenAIJudge("gpt-4o").judge("q", "1", "1")
        assert len(calls) == 1

    def test_server_errors_are_retried_then_reported(self, api_key, monkeypatch):
        calls = []

        def raise_500(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError("url", 500, "Server Error", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", raise_500)
        monkeypatch.setattr("time.sleep", lambda _: None)
        with pytest.raises(RuntimeError, match="failed after"):
            OpenAIJudge("gpt-4o", retries=3).judge("q", "1", "1")
        assert len(calls) == 3

    def test_recovers_after_a_transient_failure(self, api_key, monkeypatch):
        attempts = []

        def flaky(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)
            return _FakeHTTPResponse(_openai_response("Final Decision: Yes"))

        monkeypatch.setattr("urllib.request.urlopen", flaky)
        monkeypatch.setattr("time.sleep", lambda _: None)
        assert OpenAIJudge("gpt-4o").judge("q", "1", "1").equivalent is True
        assert len(attempts) == 2


class TestResolveJudge:
    def test_openai_backend(self, api_key):
        assert resolve_judge("openai:gpt-4o").name == "openai:gpt-4o"

    def test_ollama_backend(self):
        judge = resolve_judge("ollama:qwen2.5:7b-instruct")
        assert isinstance(judge, OllamaJudge)
        # The model tag itself contains a colon; only the backend is split off.
        assert judge.name == "ollama:qwen2.5:7b-instruct"

    @pytest.mark.parametrize("spec", ["openai", "", "gpt-4o"])
    def test_missing_backend_rejected(self, spec):
        with pytest.raises(ValueError, match="backend"):
            resolve_judge(spec)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown judge backend"):
            resolve_judge("anthropic:claude")


class TestAgreement:
    def test_perfect_agreement(self):
        a = [JudgeVerdict(True, ""), JudgeVerdict(False, "")]
        result = agreement(a, list(a))
        assert result.rate == 100.0
        assert result.both_labelled == 2

    def test_total_disagreement(self):
        a = [JudgeVerdict(True, ""), JudgeVerdict(True, "")]
        b = [JudgeVerdict(False, ""), JudgeVerdict(False, "")]
        result = agreement(a, b)
        assert result.rate == 0.0
        assert result.primary_only_yes == 2
        assert result.secondary_only_yes == 0

    def test_abstentions_excluded_not_counted_as_disagreement(self):
        """Otherwise a truncated judge would look like a disagreeing one."""
        a = [JudgeVerdict(True, ""), JudgeVerdict(None, "")]
        b = [JudgeVerdict(True, ""), JudgeVerdict(True, "")]
        result = agreement(a, b)
        assert result.both_labelled == 1
        assert result.either_abstained == 1
        assert result.rate == 100.0

    def test_rate_is_zero_when_nothing_comparable(self):
        result = agreement([JudgeVerdict(None, "")], [JudgeVerdict(None, "")])
        assert result.both_labelled == 0
        assert result.rate == 0.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            agreement([JudgeVerdict(True, "")], [])

    def test_direction_of_disagreement_is_recorded(self):
        """Which judge is more permissive is the actionable part of A-5."""
        a = [JudgeVerdict(False, "")]
        b = [JudgeVerdict(True, "")]
        result = agreement(a, b)
        assert result.secondary_only_yes == 1
        assert result.primary_only_yes == 0

    def test_dataclass_is_frozen(self):
        result = JudgeAgreement(1, 1, 0, 0, 0)
        with pytest.raises(AttributeError):
            result.agreements = 5  # type: ignore[misc]
