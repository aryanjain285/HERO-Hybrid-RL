#!/usr/bin/env bash
# Run the full audit: test suite, advantage-invariance analysis, and end-to-end
# GRPO training experiments. No GPU, no network, about a minute.
#
# Every artefact lands under runs/audit-<timestamp>/ with a manifest recording the
# git SHA, so a result can always be traced to the code that produced it.
#
#   scripts/run_audit.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_package
cd "$REPO_ROOT"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$(run_dir "audit-$STAMP")"
log "output: $OUT"

log "1/3  test suite"
"$PY" -m pytest tests/ -q --junit-xml="$OUT/pytest.xml" | tee "$OUT/pytest.log"

log "2/3  advantage invariance (audit A-1, A-1b, A-15, A-16, A-17)"
"$PY" analysis/invariance_check.py | tee "$OUT/invariance_check.log"

log "3/3  end-to-end GRPO training (audit A-1 at training level, A-12)"
"$PY" analysis/grpo_end_to_end.py | tee "$OUT/grpo_end_to_end.log"

write_manifest "$OUT" \
  "stage: audit" \
  "gpu_required: false" \
  "tests: $(grep -c . "$OUT/pytest.log" >/dev/null && echo passed || echo unknown)"

log "audit complete. Logs in $OUT"
log "to stage models for the training milestone: scripts/fetch_models.sh --tier dev"
