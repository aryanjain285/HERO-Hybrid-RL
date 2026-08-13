# Milestone 0 results: verifier study

Executed 13 Aug 2026. Reproduces the paper's Table 1 *methodology* at reduced
scale on local hardware: generate responses to real problems, score each with
every rule verifier, label true correctness with an LLM judge using the paper's
template, then measure recall and precision against those labels.

## Setup

| | |
|---|---|
| Problems | 30 from MATH-500, all levels |
| Responses | 60 (2 samples per problem, temperature 0.7, seeded) |
| Generator | `qwen2.5:1.5b-instruct` via Ollama, CPU, ~48 tok/s |
| Judge | `gpt-4o`, paper's Figure 4 compare-don't-solve template, temperature 0 |
| Judge cost | $0.13 for 61 calls (40,597 prompt + 3,136 completion tokens) |
| Labelled | 60/60, zero abstentions |
| Judge base rate | 38.3% of responses correct |

Reproduce with `scripts/run_m0.sh`, or re-score verifier changes against the fixed
labels for free via `python analysis/rescore_study.py runs/m0_gpt4o.json`.

## Results

Intervals are Wilson 95%, as PRD §9.2 requires. Point estimates alone would be
misleading at n=23 positives.

| verifier | recall | precision | accuracy | FPR |
|---|---|---|---|---|
| `raw_match` | 60.9 [40.8, 77.8] | 100.0 [78.5, 100.0] | 85.0 [73.9, 91.9] | 0.0 |
| `exact_match` | 95.7 [79.0, 99.2] | 100.0 [85.1, 100.0] | 98.3 [91.1, 99.7] | 0.0 |
| `normalised_match` | 95.7 [79.0, 99.2] | 100.0 [85.1, 100.0] | 98.3 [91.1, 99.7] | 0.0 |
| `symbolic` | 100.0 [85.7, 100.0] | 100.0 [85.7, 100.0] | 100.0 [94.0, 100.0] | 0.0 |

**What reproduces.** The precision/recall trade-off that motivates HERO. Literal
matching loses 39% of genuinely correct answers at perfect precision; progressively
more permissive checking recovers all of them without a single false positive. The
`raw_match` and `symbolic` recall intervals do not overlap, so that gap is not
noise. Nine correct answers were recovered beyond literal matching purely by
formatting tolerance.

**What does not compare to the paper.** Absolute numbers. The paper measures 750
responses over 250 deliberately hard-to-verify HardVerify-Math problems, and its
checkers are different implementations — `math_reward.py` compares raw strings and
scores 10.1% recall, whereas even `raw_match` here extracts from `\boxed{}` first.
MATH-500 is also easier and better formatted, so recall is uniformly higher.

## Judge reliability (PRD §9.3, audit A-5)

The study was run twice, once with a local `qwen2.5:1.5b-instruct` judge and once
with GPT-4o, over the same problems.

| | |
|---|---|
| Comparable items | 58 |
| Agreement | 82.8% [71.1, 90.4] |
| Local says correct, GPT-4o disagrees | 7 |
| GPT-4o says correct, local disagrees | 3 |

Every disagreement inspected by hand favours GPT-4o. Examples:

* reference `3\sqrt{13}`, response `\sqrt{117}` — equal, since √117 = √(9·13). The
  local judge said no.
* reference `\frac{14}{3}`, response `\frac{5}{3}` — unequal. The local judge said yes.
* reference `1,-2`, response `1` — a partial answer. The local judge said yes.

The weak judge errs in **both** directions, so it is not a bias that could be
corrected for. Two consequences for the project:

1. The judge must be stronger than the generator. A 1.5B judge scoring 1.5B
   generations produced a table whose verifier "false positives" were mostly judge
   mistakes.
2. Audit A-5 is not hypothetical. It reproduced here on the first attempt, which is
   direct support for the PRD's mandate to dual-judge and hand-audit rather than
   trust judge scores. `runs/judge_audit.csv` holds the 60-row audit sheet.

## Defects this milestone found in the harness

Every one was found by running the study rather than by reading the code, and each
is now a regression test.

| Defect | Effect |
|---|---|
| `%` stripped before `\%` | `50\%` normalised to `50\`, failing a correct answer |
| `$` stripped before `\$` | `\$78` normalised to `\78`; same bug class, found later |
| "Final answer:" cue crossed a newline | Captured the following prose line as the answer |
| Backtracking in the same cue | Surrendered `[:=]?` and captured `:` as the answer |
| No display-math fallback | `(3, \frac{\pi}{2})` degraded to `2` via the last-number rule |
| Broad `except Exception` in `SymbolicVerifier` | A subprocess pool that failed to spawn made the verifier silently degrade to normalised matching, appearing only as unexplained lost recall |
| Judge capped at 64 tokens | Two abstentions were truncations mid-reasoning, not uncertainty |

Recall rose from 87.0 to 100.0 across these fixes, measured against a fixed label
set so the improvements are attributable to the verifiers and not to re-generation.

## Limitations

* 60 responses is small; every interval is wide and no result here should be quoted
  without one.
* One generator and one benchmark. The paper's regime distinction
  (easy-to-verify vs hard-to-verify training data) is not exercised.
* MATH-500 substitutes for HardVerify-Math, which is not published standalone.
* Ground truth is a single strong judge, not human labels. Audit A-14 applies to
  this study exactly as it does to the paper's Table 1: the label provenance is a
  project decision (D-10), not a given.
