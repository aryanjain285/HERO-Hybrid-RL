"""Dual-judge agreement and the manual audit sheet (PRD 9.3, audit A-5).

Compares two judges' labels over the same responses, reports agreement with a
confidence interval, and exports the disagreements plus a sample of agreements as
a CSV for hand adjudication. The PRD makes that audit mandatory: unaudited judge
scores can reward formatting rather than reasoning.

    python analysis/judge_agreement.py runs/m0_local.json runs/m0_gpt4o.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hero.judges import agreement
from hero.llm import JudgeVerdict
from hero.stats import wilson_interval
from hero.study import load_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path, help="study JSON from judge A")
    parser.add_argument("secondary", type=Path, help="study JSON from judge B")
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument(
        "--audit-sample",
        type=int,
        default=50,
        help="agreed items to include alongside every disagreement (PRD 9.3)",
    )
    args = parser.parse_args(argv)

    first = load_study(args.primary)
    second = load_study(args.secondary)

    by_uid_a = {(r.uid, r.sample_index): r for r in first.records}
    by_uid_b = {(r.uid, r.sample_index): r for r in second.records}
    shared = sorted(set(by_uid_a) & set(by_uid_b))
    if not shared:
        print("no responses in common; the studies used different samples")
        return 1

    identical_text = sum(
        1 for key in shared if by_uid_a[key].response == by_uid_b[key].response
    )
    print(f"judge A : {first.judge_name or args.primary.name}")
    print(f"judge B : {second.judge_name or args.secondary.name}")
    print(f"shared responses: {len(shared)}")
    print(f"identical response text: {identical_text}/{len(shared)}")
    if identical_text != len(shared):
        print(
            "  NOTE: judges saw different text for some items, so disagreement\n"
            "  conflates judge behaviour with generation differences."
        )

    verdicts_a = [JudgeVerdict(by_uid_a[k].judge_label, by_uid_a[k].judge_raw) for k in shared]
    verdicts_b = [JudgeVerdict(by_uid_b[k].judge_label, by_uid_b[k].judge_raw) for k in shared]
    result = agreement(verdicts_a, verdicts_b)
    interval = wilson_interval(result.agreements, result.both_labelled)

    print(f"\n{'=' * 70}")
    print("Judge agreement (PRD 9.3)")
    print("=" * 70)
    print(f"comparable items    : {result.both_labelled}")
    print(f"agreements          : {result.agreements}")
    print(f"agreement rate      : {interval}  (Wilson 95%)")
    print(f"A says yes, B no    : {result.primary_only_yes}")
    print(f"B says yes, A no    : {result.secondary_only_yes}")
    print(f"either abstained    : {result.either_abstained}")

    disagreements = [
        key
        for key, va, vb in zip(shared, verdicts_a, verdicts_b)
        if not va.abstained and not vb.abstained and va.equivalent != vb.equivalent
    ]
    print(f"\ndisagreements: {len(disagreements)}")
    for key in disagreements[:8]:
        record = by_uid_a[key]
        tail = record.response.strip().replace("\n", " ")[-90:]
        print(f"\n  reference : {record.reference}")
        print(f"  response  : ...{tail}")
        print(f"  A={by_uid_a[key].judge_label}  B={by_uid_b[key].judge_label}")

    if args.audit_csv:
        agreed = [k for k in shared if k not in set(disagreements)]
        selected = disagreements + agreed[: args.audit_sample]
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "uid", "sample", "reference", "response",
                    "judge_a", "judge_b", "disagree", "human_verdict", "notes",
                ]
            )
            for key in selected:
                record = by_uid_a[key]
                writer.writerow(
                    [
                        record.uid, record.sample_index, record.reference,
                        record.response.strip(),
                        by_uid_a[key].judge_label, by_uid_b[key].judge_label,
                        key in set(disagreements), "", "",
                    ]
                )
        print(f"\naudit sheet -> {args.audit_csv} ({len(selected)} rows)")
        print("Fill 'human_verdict' with TRUE/FALSE; it becomes a report appendix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
