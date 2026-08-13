#!/usr/bin/env bash
# Download model weights for a compute tier into a shared cache.
#
# Tier definitions live in hero/registry.py, so this script never carries its own
# copy of a model list that could drift from the configs.
#
#   scripts/fetch_models.sh --tier dev            # Qwen3-1.7B + AceMath RM + verifier
#   scripts/fetch_models.sh --tier smoke          # smallest end-to-end set
#   scripts/fetch_models.sh --only qwen3-0.6b     # one model by registry key
#   scripts/fetch_models.sh --tier dev --dry-run
#
# A tier fetch is preflighted with `hero.cli check-models`, which confirms every
# repository is reachable and reports the disk needed before any bytes move.
#
# Downloads are resumable and skipped when already complete, so re-running after
# an interrupted transfer is safe. Set HF_HOME to control the cache location;
# gated repositories need HF_TOKEN.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TIER="dev"
DRY_RUN=0
ONLY=""
while (( $# )); do
  case "$1" in
    --tier) TIER="${2:-}"; [[ -n "$TIER" ]] || die "--tier needs a value"; shift 2 ;;
    --only) ONLY="${2:-}"; [[ -n "$ONLY" ]] || die "--only needs a registry key"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

require_package
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

# huggingface_hub is not a runtime dependency of the audit stages, so install it
# on demand rather than making every user of the CPU path carry it.
if ! "$PY" -c 'import huggingface_hub' 2>/dev/null; then
  log "installing huggingface_hub"
  "$PY" -m pip install --quiet huggingface_hub
fi

if [[ -n "$ONLY" ]]; then
  log "single model: $ONLY"
  mapfile -t MODELS < <("$PY" -m hero.cli models --key "$ONLY" --field hf_id)
else
  log "tier: $TIER"
  # Preflight against the Hub before committing to a large transfer: catches
  # renamed repos, missing tokens for gated ones, and undersized disks.
  log "verifying repositories and sizing the download"
  "$PY" -m hero.cli check-models --tier "$TIER" | sed 's/^/       /' \
    || die "one or more repositories are unreachable; see above."
  mapfile -t MODELS < <("$PY" -m hero.cli models --tier "$TIER" --field hf_id)
fi

(( ${#MODELS[@]} > 0 )) || die "nothing to fetch"
log "cache: $HF_HOME"

if (( DRY_RUN )); then
  log "dry run: nothing downloaded"
  exit 0
fi

[[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN unset; gated repositories will fail."

# Large-model transfers fail often enough that a single attempt is not a pipeline.
# Observed failure modes: the Xet CAS backend erroring mid-transfer on restricted
# or proxied networks, and ordinary connection resets. Each attempt resumes from
# the cache, so a retry costs only the remaining bytes; the final attempt disables
# Xet entirely and falls back to plain HTTPS range requests, which traverse
# corporate proxies more reliably.
ATTEMPTS="${HERO_FETCH_ATTEMPTS:-3}"

fetch_one() {
  local model="$1" attempt=1 rc=0
  while (( attempt <= ATTEMPTS )); do
    if (( attempt == ATTEMPTS )); then
      warn "final attempt for $model with Xet disabled (plain HTTPS)"
      export HF_HUB_DISABLE_XET=1
    fi
    log "fetching $model (attempt $attempt/$ATTEMPTS)"
    rc=0
    "$PY" - "$model" <<'PY' || rc=$?
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id=sys.argv[1], max_workers=4)
print(f"       -> {path}")
PY
    if (( rc == 0 )); then
      unset HF_HUB_DISABLE_XET
      return 0
    fi
    warn "attempt $attempt failed (exit $rc)"
    (( attempt++ ))
    (( attempt <= ATTEMPTS )) && sleep $(( attempt * 5 ))
  done
  unset HF_HUB_DISABLE_XET
  return 1
}

for model in "${MODELS[@]}"; do
  fetch_one "$model" || die "could not fetch $model after $ATTEMPTS attempts. If the
       error mentions CAS or Xet, retry with HF_HUB_DISABLE_XET=1 set in the
       environment; if it mentions gating, accept the licence and set HF_TOKEN."
done

write_manifest "$(run_dir "fetch-${ONLY:-$TIER}")" \
  "stage: fetch_models" \
  "selection: ${ONLY:-tier=$TIER}" \
  "hf_home: $HF_HOME" \
  "models: ${MODELS[*]}"

log "all models present"
