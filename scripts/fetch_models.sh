#!/usr/bin/env bash
# Download the model weights for a compute tier into a shared cache.
#
# The tier definitions live in hero/registry.py, so this script never carries its
# own copy of a model list that could drift from the configs.
#
#   scripts/fetch_models.sh --tier dev          # Qwen3-1.7B + AceMath RM + verifier
#   scripts/fetch_models.sh --tier smoke        # smallest end-to-end set
#   scripts/fetch_models.sh --tier dev --dry-run
#
# Downloads are resumable and skipped when already complete, so re-running after
# an interrupted transfer is safe. Set HF_HOME to control the cache location;
# gated repositories need HF_TOKEN.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TIER="dev"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --tier) TIER="${2:-}"; [[ -n "$TIER" ]] || die "--tier needs a value"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

require_package
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

log "tier: $TIER"
log "cache: $HF_HOME"
"$PY" -m hero.cli models --tier "$TIER" --field table | sed 's/^/       /'

mapfile -t MODELS < <("$PY" -m hero.cli models --tier "$TIER" --field hf_id)
(( ${#MODELS[@]} > 0 )) || die "tier $TIER resolved to no models"

if (( DRY_RUN )); then
  log "dry run: nothing downloaded"
  exit 0
fi

# huggingface_hub is not a runtime dependency of the audit stages, so install it
# on demand rather than making every user of the CPU path carry it.
if ! "$PY" -c 'import huggingface_hub' 2>/dev/null; then
  log "installing huggingface_hub"
  "$PY" -m pip install --quiet huggingface_hub
fi

[[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN unset; gated repositories will fail."

for model in "${MODELS[@]}"; do
  log "fetching $model"
  # download() resumes partial transfers and no-ops when the snapshot is complete.
  "$PY" - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
path = snapshot_download(repo_id=repo, resume_download=True)
print(f"       -> {path}")
PY
done

write_manifest "$(run_dir "fetch-$TIER")" \
  "stage: fetch_models" \
  "tier: $TIER" \
  "hf_home: $HF_HOME" \
  "models: ${MODELS[*]}"

log "all models present for tier $TIER"
