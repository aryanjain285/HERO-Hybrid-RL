"""Milestone 0: reproduce the paper's Table 1 methodology on local models.

Pipeline: sample responses to MATH-500 problems, score each with every rule
verifier, label true correctness with the paper's judge template, then report
recall, precision, false-positive rate, and accuracy per verifier.

Scale is set by ``--problems`` and ``--samples``; defaults are small enough to
finish in minutes on CPU. The paper uses 750 responses over 250 HardVerify-Math
problems, so absolute numbers are not comparable -- the reproducible claim is the
precision/recall ordering across verifiers, and the presence of false negatives.

Requires a running Ollama server. Usage:
    python analysis/verifier_study.py --problems 40 --samples 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hero.data import load_math500
from hero.env import load_env_file
from hero.judges import OpenAIJudge, resolve_judge
from hero.llm import OllamaClient, OllamaConfig, OllamaError
from hero.study import (
    apply_verifiers,
    generate_responses,
    label_responses,
    score_verifiers,
    StudyResult,
)
from hero.verifiers import (
    ExactMatchVerifier,
    NormalisedMatchVerifier,
    RawMatchVerifier,
    SymbolicVerifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=int, default=40)
    parser.add_argument("--samples", type=int, default=2, help="responses per problem")
    parser.add_argument("--min-level", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct")
    parser.add_argument(
        "--judge",
        default="ollama:qwen2.5:7b-instruct",
        help="'openai:gpt-4o' for the paper's protocol, or 'ollama:<model>'",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="dotenv file supplying OPENAI_API_KEY; values are never logged",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default=None, help="write full results as JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser


def print_table(result: StudyResult) -> None:
    print(f"\n{'=' * 78}")
    print("Verifier comparison (paper Table 1 methodology, judge-labelled)")
    print("=" * 78)
    header = f"{'verifier':>20} | {'recall':>7} | {'prec.':>7} | {'FPR':>6} | {'acc.':>6} | {'err':>4}"
    print(header)
    print("-" * 78)
    for m in result.metrics:
        print(
            f"{m.name:>20} | {m.recall:7.1f} | {m.precision:7.1f} | "
            f"{m.false_positive_rate:6.1f} | {m.accuracy:6.1f} | {m.errors:4d}"
        )
    print(
        f"\nlabelled responses: {result.metrics[0].support if result.metrics else 0}"
        f"  judge base rate: {result.base_rate:.1f}%"
        f"  abstentions: {result.abstentions}"
    )


def print_false_negatives(result: StudyResult, limit: int = 5) -> None:
    """Show cases the strict verifier rejected but the judge accepted.

    These are the false negatives that motivate HERO: a correct answer scoring 0
    contributes a wrong gradient, not merely a missing one.
    """
    strict = ExactMatchVerifier.name
    lenient = NormalisedMatchVerifier.name
    cases = [
        r
        for r in result.records
        if r.judge_label is True
        and r.verdicts.get(strict) != "correct"
        and r.verdicts.get(lenient) == "correct"
    ]
    print(f"\n{'=' * 78}")
    print(f"False negatives recovered by normalisation ({len(cases)} found)")
    print("=" * 78)
    if not cases:
        print("none in this sample")
        return
    for record in cases[:limit]:
        tail = record.response.strip().replace("\n", " ")[-110:]
        print(f"\n  reference : {record.reference}")
        print(f"  response  : ...{tail}")
        print(f"  strict    : {record.verdicts.get(strict)}")


def print_judge_disagreements(result: StudyResult, limit: int = 6) -> None:
    """Cases where the strongest verifier and the judge disagree.

    Audit A-5 requires these to be re-adjudicated by hand rather than assumed to
    be verifier errors: the judge is the label source, so a judge mistake is
    recorded as a verifier false positive. A local 1.5B judge, for instance, ruled
    sqrt(117) not equivalent to 3*sqrt(13).
    """
    cases = [
        r
        for r in result.records
        if r.is_labelled and (r.verdicts.get("symbolic") == "correct") != bool(r.judge_label)
    ]
    print(f"\n{'=' * 78}")
    print(f"Symbolic verifier vs judge disagreements ({len(cases)}) -- audit by hand")
    print("=" * 78)
    if not cases:
        print("none")
        return
    for record in cases[:limit]:
        tail = record.response.strip().replace("\n", " ")[-100:]
        print(f"\n  reference : {record.reference}")
        print(f"  response  : ...{tail}")
        print(f"  symbolic  : {record.verdicts.get('symbolic')}   judge: {record.judge_label}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress = not args.quiet

    if args.env_file:
        loaded = load_env_file(args.env_file)
        print(f"env: loaded {len(loaded)} variable(s) from {args.env_file}")

    generator = OllamaClient(
        OllamaConfig(model=args.model, host=args.host, max_tokens=args.max_tokens)
    )
    try:
        generator.require_model()
        judge = resolve_judge(args.judge, host=args.host)
        judge.preflight()
    except (OllamaError, RuntimeError, ValueError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    print(f"generator: {generator.config.model}")
    print(f"judge    : {judge.name}")
    problems = load_math500(limit=args.problems, min_level=args.min_level)
    print(f"problems : {len(problems)} at level >= {args.min_level}")
    print(f"responses: {len(problems) * args.samples}")

    started = time.time()
    print("\ngenerating responses")
    records = generate_responses(
        generator,
        problems,
        samples_per_problem=args.samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        progress=progress,
    )

    print("\napplying verifiers")
    with SymbolicVerifier(timeout_s=5.0) as symbolic:
        verifiers = (
            RawMatchVerifier(),
            ExactMatchVerifier(),
            NormalisedMatchVerifier(),
            symbolic,
        )
        names = tuple(v.name for v in verifiers)
        apply_verifiers(records, verifiers)

    print("\njudging true correctness")
    abstentions, truncations = label_responses(judge, records, progress=progress)

    labelled = [r for r in records if r.is_labelled]
    base_rate = 100.0 * sum(bool(r.judge_label) for r in labelled) / max(len(labelled), 1)
    result = StudyResult(
        records=records,
        metrics=score_verifiers(records, names),
        base_rate=base_rate,
        abstentions=abstentions,
        truncations=truncations,
        judge_name=judge.name,
    )

    print_table(result)
    print_false_negatives(result)
    print_judge_disagreements(result)
    if isinstance(judge, OpenAIJudge):
        print(
            f"\njudge cost: ${judge.estimated_cost_usd:.4f} over {judge.calls} calls "
            f"({judge.prompt_tokens} prompt + {judge.completion_tokens} completion tokens)"
        )
    if truncations:
        print(
            f"\nWARNING: {truncations} judge response(s) hit the token cap. "
            "Raise the judge's max_tokens; these are configuration faults, not data."
        )
    print(f"\nelapsed: {time.time() - started:.0f}s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_json(args.out)
        print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
