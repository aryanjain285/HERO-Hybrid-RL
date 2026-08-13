"""Tests for the verifier-study metrics and the LLM client.

Metric arithmetic is verified against hand-computed confusion matrices, and
against the paper's own Table 1, which is internally consistent enough to serve
as a fixture.
"""

from __future__ import annotations

import json

import pytest

from hero.llm import EQUIVALENCE_TEMPLATE, JudgeVerdict, OllamaConfig, _parse_decision
from hero.study import ResponseRecord, StudyResult, score_verifiers
from hero.verifiers import Verdict


def record(uid: str, verdicts: dict[str, str], label: bool | None) -> ResponseRecord:
    return ResponseRecord(
        uid=uid,
        question="q",
        reference="42",
        level=3,
        sample_index=0,
        response="r",
        tokens=1,
        verdicts=verdicts,
        judge_label=label,
    )


class TestScoreVerifiers:
    def test_perfect_verifier(self):
        records = [
            record("a", {"v": Verdict.CORRECT}, True),
            record("b", {"v": Verdict.INCORRECT}, False),
        ]
        (m,) = score_verifiers(records, ("v",))
        assert (m.recall, m.precision, m.false_positive_rate, m.accuracy) == (
            100.0,
            100.0,
            0.0,
            100.0,
        )

    def test_conservative_verifier_has_low_recall_high_precision(self):
        """The math_reward.py profile: rejects correct answers, never over-credits."""
        records = [record(f"p{i}", {"v": Verdict.INCORRECT}, True) for i in range(9)]
        records.append(record("p9", {"v": Verdict.CORRECT}, True))
        (m,) = score_verifiers(records, ("v",))
        assert m.recall == pytest.approx(10.0)
        assert m.precision == pytest.approx(100.0)
        assert m.false_positive_rate == 0.0

    def test_permissive_verifier_has_high_recall_and_fpr(self):
        records = [record(f"t{i}", {"v": Verdict.CORRECT}, True) for i in range(8)]
        records += [record(f"f{i}", {"v": Verdict.CORRECT}, False) for i in range(4)]
        (m,) = score_verifiers(records, ("v",))
        assert m.recall == pytest.approx(100.0)
        assert m.false_positive_rate == pytest.approx(100.0)
        assert m.precision == pytest.approx(100 * 8 / 12)

    def test_confusion_matrix_counts(self):
        records = [
            record("a", {"v": Verdict.CORRECT}, True),
            record("b", {"v": Verdict.CORRECT}, False),
            record("c", {"v": Verdict.INCORRECT}, True),
            record("d", {"v": Verdict.INCORRECT}, False),
        ]
        (m,) = score_verifiers(records, ("v",))
        assert (m.true_positives, m.false_positives, m.false_negatives, m.true_negatives) == (
            1,
            1,
            1,
            1,
        )
        assert m.support == 4
        assert m.accuracy == pytest.approx(50.0)

    def test_errors_are_excluded_from_the_matrix(self):
        """A broken checker must not be scored as a strict one."""
        records = [
            record("a", {"v": Verdict.CORRECT}, True),
            record("b", {"v": Verdict.ERROR}, True),
        ]
        (m,) = score_verifiers(records, ("v",))
        assert m.errors == 1
        assert m.support == 1
        assert m.recall == pytest.approx(100.0)

    def test_unlabelled_records_are_excluded(self):
        records = [
            record("a", {"v": Verdict.CORRECT}, True),
            record("b", {"v": Verdict.CORRECT}, None),
        ]
        (m,) = score_verifiers(records, ("v",))
        assert m.support == 1

    def test_no_answer_counts_as_a_negative_prediction(self):
        records = [record("a", {"v": Verdict.NO_ANSWER}, True)]
        (m,) = score_verifiers(records, ("v",))
        assert m.false_negatives == 1
        assert m.recall == 0.0

    def test_undefined_ratios_are_zero_not_nan(self):
        """Empty denominators must not poison a results table."""
        (m,) = score_verifiers([], ("v",))
        assert (m.recall, m.precision, m.accuracy) == (0.0, 0.0, 0.0)

    def test_multiple_verifiers_scored_independently(self):
        records = [
            record("a", {"strict": Verdict.INCORRECT, "lenient": Verdict.CORRECT}, True)
        ]
        strict, lenient = score_verifiers(records, ("strict", "lenient"))
        assert strict.recall == 0.0
        assert lenient.recall == 100.0

    def test_reproduces_paper_table1_row(self):
        """math_verify (verl): recall 68.4, precision 100.0, FPR 0.0, acc 83.7.

        Table 1 is consistent with 387 positives out of 750 responses. Rebuilding
        that row from the implied confusion matrix checks the metric definitions
        match the paper's.
        """
        positives, total = 387, 750
        negatives = total - positives
        tp = round(0.684 * positives)
        records = [record(f"tp{i}", {"v": Verdict.CORRECT}, True) for i in range(tp)]
        records += [
            record(f"fn{i}", {"v": Verdict.INCORRECT}, True) for i in range(positives - tp)
        ]
        records += [record(f"tn{i}", {"v": Verdict.INCORRECT}, False) for i in range(negatives)]
        (m,) = score_verifiers(records, ("v",))
        assert m.recall == pytest.approx(68.4, abs=0.2)
        assert m.precision == pytest.approx(100.0)
        assert m.false_positive_rate == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(83.7, abs=0.3)


class TestStudyResultSerialisation:
    def test_round_trip(self, tmp_path):
        records = [record("a", {"v": Verdict.CORRECT}, True)]
        result = StudyResult(
            records=records,
            metrics=score_verifiers(records, ("v",)),
            base_rate=100.0,
            abstentions=0,
        )
        path = tmp_path / "out.json"
        result.to_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["base_rate"] == 100.0
        assert payload["metrics"][0]["name"] == "v"
        assert payload["records"][0]["uid"] == "a"

    def test_records_retain_judge_output_for_audit(self, tmp_path):
        """Manual re-adjudication is mandatory (A-5), so raw text must persist."""
        rec = record("a", {"v": Verdict.CORRECT}, True)
        rec.judge_raw = "Final Decision: Yes"
        result = StudyResult([rec], score_verifiers([rec], ("v",)), 100.0, 0)
        path = tmp_path / "out.json"
        result.to_json(path)
        assert "Final Decision: Yes" in path.read_text(encoding="utf-8")


class TestJudgeParsing:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Final Decision: Yes", True),
            ("Final Decision: No", False),
            ("final decision: yes", True),
            ("blah blah\nFinal Decision: No\n", False),
            ("Yes", True),
            ("No.", False),
            ("**Yes**", True),
        ],
    )
    def test_decisions(self, raw, expected):
        assert _parse_decision(raw) is expected

    @pytest.mark.parametrize("raw", ["", "I am not sure", "maybe", "Final Decision:"])
    def test_abstentions_are_none_not_false(self, raw):
        """Coercing an abstention to False would silently inflate false negatives."""
        assert _parse_decision(raw) is None

    def test_verdict_wrapper_flags_abstention(self):
        assert JudgeVerdict(None, "").abstained
        assert not JudgeVerdict(True, "").abstained


class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig(model="m")
        assert cfg.temperature == 0.0
        assert cfg.retries >= 1

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"model": ""}, "model"),
            ({"model": "m", "max_tokens": 0}, "max_tokens"),
            ({"model": "m", "temperature": -1}, "temperature"),
            ({"model": "m", "retries": 0}, "retries"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OllamaConfig(**kwargs)


class TestEquivalenceTemplate:
    def test_matches_the_paper_wording(self):
        """Pinned: changing the judge prompt invalidates cross-run comparisons."""
        assert "Do not solve the question by yourself" in EQUIVALENCE_TEMPLATE
        assert 'output "Final Decision: Yes"' in EQUIVALENCE_TEMPLATE

    def test_formats_all_three_fields(self):
        filled = EQUIVALENCE_TEMPLATE.format(
            question="Q", ground_truth="GT", student_answer="SA"
        )
        assert "Q" in filled and "GT" in filled and "SA" in filled
        assert "{" not in filled
