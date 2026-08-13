"""Tests for the rule-based verifiers.

The false-negative cases are transcribed from the paper's Table 10, which
publishes real ground-truth/prediction pairs together with which verifiers accept
each one. They serve as a labelled benchmark for this implementation.
"""

from __future__ import annotations

import pytest

from hero.verifiers import (
    ExactMatchVerifier,
    NormalisedMatchVerifier,
    SymbolicVerifier,
    Verdict,
    extract_answers,
    normalise,
)

EXACT = ExactMatchVerifier()
LENIENT = NormalisedMatchVerifier()


class TestExtraction:
    def test_single_boxed(self):
        assert extract_answers(r"so \boxed{42}") == ("42",)

    def test_nested_braces_survive(self):
        assert extract_answers(r"\boxed{\frac{1}{2}}") == (r"\frac{1}{2}",)

    def test_multiple_boxes_all_returned(self):
        """Paper Table 10 rows 2, 4 and 5 split answers across boxes."""
        got = extract_answers(r"\boxed{(6,3)}, \boxed{(9,3)}, \boxed{(9,5)}")
        assert got == ("(6,3)", "(9,3)", "(9,5)")

    def test_final_answer_phrase_fallback(self):
        assert extract_answers("Final Answer: 17") == ("17",)

    def test_last_number_fallback(self):
        assert extract_answers("we get 3 then 4 then 12") == ("12",)

    def test_boxed_preferred_over_prose(self):
        assert extract_answers(r"answer is 5 but \boxed{7}") == ("7",)

    @pytest.mark.parametrize("text", ["", "   ", "no digits here"])
    def test_nothing_to_extract(self, text):
        assert extract_answers(text) == ()

    def test_unclosed_box_is_ignored(self):
        assert extract_answers(r"\boxed{42") == ("42",)  # falls back to last number

    def test_final_answer_cue_does_not_cross_a_newline(self):
        """Regression: `\\s*` crossed the line break and captured the next line."""
        text = "#### Final answer:\n\nThus, the coordinates are:\n\\[\n(3, 5)\n\\]"
        assert extract_answers(text) == ("(3, 5)",)

    def test_final_answer_cue_does_not_capture_its_own_colon(self):
        """Regression: backtracking surrendered `[:=]?`, capturing ':' as the answer."""
        assert extract_answers("Final answer:\n\\[ 7 \\]") == ("7",)

    def test_display_math_beats_last_number_fallback(self):
        """`(3, \\frac{\\pi}{2})` must not degrade to '2'."""
        got = extract_answers("so we get\n\\[ (3, \\frac{\\pi}{2}) \\]")
        assert got == (r"(3, \frac{\pi}{2})",)

    def test_prose_lead_in_is_not_taken_as_an_answer(self):
        text = "The answer is the sum of all terms in the sequence\n\\[ 12 \\]"
        assert extract_answers(text) == ("12",)

    def test_short_word_answers_still_extract(self):
        """The prose guard must not reject legitimate one-word answers."""
        assert extract_answers("Final answer: even") == ("even",)


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (r"\frac{1}{2}", "(1)/(2)"),
            (r"\dfrac{1}{2}", "(1)/(2)"),
            (r"\left( 3 \right)", "(3)"),
            (r"\text{5 cm}", "5"),
            ("1,000", "1000"),
            ("50\\%", "50"),
            ("x = 7", "7"),
            ("  42.  ", "42"),
            (r"\sqrt{2}", "sqrt(2)"),
        ],
    )
    def test_canonicalisation(self, raw, expected):
        assert normalise(raw) == expected

    def test_idempotent(self):
        for raw in (r"\frac{1}{2}", "x = 7", "1,000"):
            once = normalise(raw)
            assert normalise(once) == once


class TestPrecisionRecallOrdering:
    """The premise of the paper: strict checkers reject correct answers."""

    def test_strict_rejects_reordered_set_but_lenient_accepts(self):
        """Set ordering is the cleanest strict/lenient separation.

        Both normalise punctuation, so the difference shows up only where the
        answer is a collection whose order carries no meaning.
        """
        assert not EXACT.verify(r"\boxed{3,1,2}", "1,2,3").is_pass
        assert LENIENT.verify(r"\boxed{3,1,2}", "1,2,3").is_pass

    def test_percent_sign_is_normalised_by_both(self):
        """Regression: stripping `%` before `\\%` left a stray backslash."""
        assert normalise(r"50\%") == "50"
        assert EXACT.verify(r"\boxed{50\%}", "50").is_pass
        assert LENIENT.verify(r"\boxed{50\%}", "50").is_pass

    def test_escaped_currency_is_normalised(self):
        """Regression, same class as the percent bug: `\\$78` left a stray backslash."""
        assert normalise(r"\$78") == "78"
        assert LENIENT.verify(r"\boxed{\$78}", "78").is_pass

    def test_tuple_assignment_prefix_is_stripped(self):
        """`(r, \\theta) = (3, 5)` states the same answer as `(3, 5)`."""
        assert normalise(r"(r, \theta) = (3, 5)") == "(3,5)"

    def test_known_greek_symbols_survive_as_sympy_names(self):
        assert normalise(r"\frac{\pi}{2}") == "(pi)/(2)"
        assert normalise(r"\theta") == "theta"

    def test_both_accept_an_identical_answer(self):
        assert EXACT.verify(r"\boxed{42}", "42").is_pass
        assert LENIENT.verify(r"\boxed{42}", "42").is_pass

    def test_neither_accepts_a_wrong_answer(self):
        assert not EXACT.verify(r"\boxed{41}", "42").is_pass
        assert not LENIENT.verify(r"\boxed{41}", "42").is_pass

    def test_lenient_never_accepts_where_strict_does_not_reject(self):
        """Lenient must dominate strict on recall, or the ordering is broken."""
        cases = [
            (r"\boxed{42}", "42"),
            (r"\boxed{50\%}", "50"),
            (r"\boxed{\frac{1}{2}}", "1/2"),
            (r"\boxed{1,000}", "1000"),
            (r"\boxed{x = 7}", "7"),
        ]
        for response, reference in cases:
            if EXACT.verify(response, reference).is_pass:
                assert LENIENT.verify(response, reference).is_pass


class TestPaperTable10:
    """Real false-negative cases published in the paper's Table 10.

    Column meanings there: ``math.py`` rejects every row; ``math_verify (verl)``
    accepts rows 1, 2, 4, 5, 6; the Math-Verify library accepts only 1 and 2.
    """

    def test_row1_boxed_function_is_accepted_by_both(self):
        """`f(x) = 2x` boxed, which the paper's math.py rejects.

        Both verifiers here accept it, because both extract from ``\\boxed{}`` and
        normalise before comparing. That is a real difference from math.py, which
        compares raw strings: ExactMatchVerifier is the strict end of *this*
        implementation, not a reimplementation of the paper's crudest checker.
        Recorded as a test so the distinction is not quietly assumed away.
        """
        assert EXACT.verify(r"\boxed{f(x) = 2x}", "f(x) = 2x").is_pass
        assert LENIENT.verify(r"\boxed{f(x) = 2x}", "f(x) = 2x").is_pass

    def test_row2_multi_box_tuple_list_is_recovered(self):
        """Four tuples across four boxes, same set as the reference."""
        response = r"\boxed{(6,3)}, \boxed{(9,3)}, \boxed{(9,5)}, \boxed{(54,5)}"
        reference = "(6,3),(9,3),(9,5),(54,5)"
        assert not EXACT.verify(response, reference).is_pass
        assert LENIENT.verify(response, reference).is_pass

    def test_row4_reordered_split_ranges_are_recovered(self):
        """Two boxes whose union equals the reference, in a different order.

        The Math-Verify library rejects this; math_verify (verl) accepts it. Set
        comparison is what makes the difference.
        """
        response = r"Final Answer: \boxed{-2,-1,0,1,2} and \boxed{10,11,12,13,14}"
        reference = "10, 11, 12, 13, 14, -2, -1, 0, 1, 2"
        assert not EXACT.verify(response, reference).is_pass
        assert LENIENT.verify(response, reference).is_pass

    def test_row5_reordered_tuples_are_recovered(self):
        response = (
            r"Final Answer: Two possible lists are \boxed{(3,5,101,107)} "
            r"and \boxed{(1,7,103,105)}"
        )
        reference = "(1,7,103,105),(3,5,101,107)"
        assert not EXACT.verify(response, reference).is_pass
        assert LENIENT.verify(response, reference).is_pass

    def test_row3_partial_set_is_correctly_rejected(self):
        """A strict subset of the reference set. Only o3 credited this row.

        Accepting it would be a false positive, so rejection is the right
        behaviour for a rule-based checker.
        """
        response = r"\boxed{(1,1,0)}, \boxed{(-1,-1,0)}"
        reference = "(0,1,1),(0,-1,-1),(1,0,1),(-1,0,-1),(1,1,0),(-1,-1,0)"
        result = LENIENT.verify(response, reference)
        assert not result.is_pass
        assert result.detail == "subset"

    def test_row6_renamed_parametric_family_is_rejected(self):
        """Renamed symbols are not textually recoverable; only o3 credited it."""
        response = r"\boxed{f(n)=cn+d}"
        reference = "f(x) = ax + b"
        assert not LENIENT.verify(response, reference).is_pass


class TestSetComparison:
    def test_tuples_are_not_flattened(self):
        """`(1,2),(3,4)` must not compare equal to `1,2,3,4`."""
        assert not LENIENT.verify(r"\boxed{1,2,3,4}", "(1,2),(3,4)").is_pass

    def test_order_insensitive(self):
        assert LENIENT.verify(r"\boxed{3,1,2}", "1,2,3").is_pass

    def test_numeric_tolerance_within_a_set(self):
        assert LENIENT.verify(r"\boxed{0.5, 2}", "1/2, 2").is_pass

    def test_superset_is_rejected_and_labelled(self):
        result = LENIENT.verify(r"\boxed{1,2,3}", "1,2")
        assert not result.is_pass
        assert result.detail == "superset"


class TestVerdictSemantics:
    def test_missing_answer_is_not_incorrect(self):
        """A formatting failure is distinguishable from a wrong answer (D-04)."""
        result = EXACT.verify("I cannot solve this.", "42")
        assert result.verdict is Verdict.NO_ANSWER
        assert not result.is_pass

    def test_only_correct_passes(self):
        assert Verdict.CORRECT.is_pass
        for verdict in (Verdict.INCORRECT, Verdict.NO_ANSWER, Verdict.ERROR):
            assert not verdict.is_pass

    def test_result_records_extraction_for_audit(self):
        result = LENIENT.verify(r"\boxed{7}", "42")
        assert result.extracted == ("7",)
        assert result.reference == ("42",)

    def test_invalid_tolerance_rejected(self):
        with pytest.raises(ValueError, match="rel_tol"):
            NormalisedMatchVerifier(rel_tol=-1.0)


@pytest.fixture(scope="module")
def symbolic():
    """One SymbolicVerifier for the module.

    Each instance spawns a subprocess on first use; sharing one keeps the suite
    fast and avoids repeated spawn churn.
    """
    with SymbolicVerifier() as verifier:
        yield verifier


class TestSymbolicVerifier:
    def test_algebraic_equivalence_beyond_normalisation(self, symbolic):
        assert symbolic.verify(r"\boxed{2*x + 2}", "2*(x+1)").is_pass

    def test_equivalent_radicals(self, symbolic):
        """The case a 1.5B judge got wrong in the M0 study: sqrt(117) = 3*sqrt(13)."""
        assert symbolic.verify(r"\boxed{\sqrt{117}}", r"3\sqrt{13}").is_pass

    def test_delegates_to_normalisation_first(self, symbolic):
        result = symbolic.verify(r"\boxed{42}", "42")
        assert result.is_pass
        assert result.strategy == "scalar"

    def test_wrong_answer_still_rejected(self, symbolic):
        assert not symbolic.verify(r"\boxed{2*x + 3}", "2*(x+1)").is_pass

    def test_unparseable_candidate_does_not_raise(self, symbolic):
        assert not symbolic.verify(r"\boxed{??!!}", "42").is_pass

    def test_latex_input_does_not_error(self, symbolic):
        """Regression: sympy raises TokenError on backslashes, which is content,
        not infrastructure, so it must not surface as Verdict.ERROR."""
        result = symbolic.verify(r"\boxed{\text{Evelyn}}", r"\text{Bob}")
        assert result.verdict is Verdict.INCORRECT

    def test_close_is_idempotent(self):
        verifier = SymbolicVerifier()
        verifier.verify(r"\boxed{2*x}", "x*2")
        verifier.close()
        verifier.close()

    def test_invalid_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout_s"):
            SymbolicVerifier(timeout_s=0)
