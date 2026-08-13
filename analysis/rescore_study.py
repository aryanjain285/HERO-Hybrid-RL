"""Re-score a saved study against its stored judge labels.

Verifier changes can be evaluated instantly and for free: responses and labels
are fixed, so only the rule-based verdicts are recomputed. This keeps the
labelled set constant across verifier revisions, which is what makes recall
figures comparable between them.

Run as a file, not via stdin: SymbolicVerifier spawns a subprocess, and
multiprocessing cannot re-import a ``<stdin>`` main module.

    python analysis/rescore_study.py runs/m0_gpt4o.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hero.study import apply_verifiers, load_study, score_verifiers
from hero.verifiers import (
    ExactMatchVerifier,
    NormalisedMatchVerifier,
    RawMatchVerifier,
    SymbolicVerifier,
    Verdict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="rewrite with new verdicts")
    parser.add_argument("--show", type=int, default=8, help="disagreements to print")
    args = parser.parse_args(argv)

    result = load_study(args.study)
    labelled = [r for r in result.records if r.is_labelled]
    print(f"study    : {args.study}")
    print(f"judge    : {result.judge_name or 'unknown'}")
    print(f"records  : {len(result.records)} ({len(labelled)} labelled)")
    print(f"base rate: {result.base_rate:.1f}%")

    with SymbolicVerifier(timeout_s=5.0) as symbolic:
        verifiers = (
            RawMatchVerifier(),
            ExactMatchVerifier(),
            NormalisedMatchVerifier(),
            symbolic,
        )
        names = tuple(v.name for v in verifiers)
        for record in result.records:
            record.verdicts = {}
        apply_verifiers(result.records, verifiers)
        result.metrics = score_verifiers(result.records, names)

    # Intervals, not bare point estimates (PRD 9.2, audit A-6).
    print(
        f"\n{'verifier':>18} | {'recall (95% CI)':>22} | {'precision (95% CI)':>22} | "
        f"{'err':>4}"
    )
    print("-" * 78)
    for m in result.metrics:
        print(
            f"{m.name:>18} | {str(m.recall_interval):>22} | "
            f"{str(m.precision_interval):>22} | {m.errors:4d}"
        )
    print(
        f"\n{'verifier':>18} | {'accuracy (95% CI)':>22} | {'FPR':>6} | {'support':>7}"
    )
    print("-" * 78)
    for m in result.metrics:
        print(
            f"{m.name:>18} | {str(m.accuracy_interval):>22} | "
            f"{m.false_positive_rate:6.1f} | {m.support:7d}"
        )

    errors = [
        r for r in result.records if r.verdicts.get("symbolic") == Verdict.ERROR
    ]
    if errors:
        print(
            f"\nWARNING: {len(errors)} symbolic verdict(s) are ERROR. The subprocess "
            "pool is unavailable, so this verifier is not actually running."
        )

    strict = RawMatchVerifier.name
    recovered = [
        r
        for r in result.records
        if r.judge_label is True
        and r.verdicts.get(strict) != Verdict.CORRECT
        and r.verdicts.get("symbolic") == Verdict.CORRECT
    ]
    print(f"\nfalse negatives recovered beyond literal matching: {len(recovered)}")

    remaining = [
        r
        for r in result.records
        if r.judge_label is True and r.verdicts.get("symbolic") != Verdict.CORRECT
    ]
    print(f"remaining false negatives: {len(remaining)}")
    for record in remaining[: args.show]:
        tail = record.response.strip().replace("\n", " ")[-90:]
        print(f"\n  reference : {record.reference}")
        print(f"  response  : ...{tail}")
        print(f"  extracted : {record.verdicts.get('symbolic')}")

    if args.out:
        result.to_json(args.out)
        print(f"\nrewritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
