#!/usr/bin/env bash
# Milestone 0: the verifier study, using a local Ollama model.
#
# Generates responses to MATH-500 problems, scores them with every rule verifier,
# labels true correctness with the paper's judge template, and reports Table 1
# metrics. CPU-only; no GPU or API key required.
#
#   scripts/run_m0.sh                                   # defaults: 30 problems x 2
#   scripts/run_m0.sh --problems 100 --samples 3
#   scripts/run_m0.sh --model qwen2.5:7b-instruct
#
# Requires a running Ollama server with the model pulled. The script pulls it if
# missing, since an implicit mid-run pull would otherwise stall the batch.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PROBLEMS=30
SAMPLES=2
MIN_LEVEL=1
MODEL="qwen2.5:1.5b-instruct"
JUDGE_MODEL=""
HOST="${OLLAMA_HOST:-http://localhost:11434}"

while (( $# )); do
  case "$1" in
    --problems) PROBLEMS="${2:?}"; shift 2 ;;
    --samples) SAMPLES="${2:?}"; shift 2 ;;
    --min-level) MIN_LEVEL="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --judge-model) JUDGE_MODEL="${2:?}"; shift 2 ;;
    --host) HOST="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

require_package
cd "$REPO_ROOT"

# Ollama installs outside PATH on Windows and some Linux setups; resolve it before
# assuming the CLI is callable.
OLLAMA_BIN="$(command -v ollama || true)"
for candidate in \
  "$HOME/AppData/Local/Programs/Ollama/ollama.exe" \
  "/c/Users/$USER/AppData/Local/Programs/Ollama/ollama.exe" \
  "/usr/local/bin/ollama"
do
  [[ -n "$OLLAMA_BIN" ]] && break
  [[ -x "$candidate" ]] && OLLAMA_BIN="$candidate"
done

curl -sf -m 10 "$HOST/api/version" >/dev/null \
  || die "no Ollama server at $HOST. Start it with: ollama serve"
log "ollama: $(curl -s -m 10 "$HOST/api/version")"

ensure_model() {
  local model="$1"
  if curl -s -m 20 "$HOST/api/tags" | grep -q "\"$model\""; then
    log "model present: $model"
    return 0
  fi
  [[ -n "$OLLAMA_BIN" ]] || die "model $model missing and no ollama CLI found to pull it"
  log "pulling $model (first run only)"
  "$OLLAMA_BIN" pull "$model" || die "could not pull $model"
}

ensure_model "$MODEL"
[[ -n "$JUDGE_MODEL" ]] && ensure_model "$JUDGE_MODEL"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$(run_dir "m0-$STAMP")"
log "output: $OUT"

ARGS=(
  --problems "$PROBLEMS"
  --samples "$SAMPLES"
  --min-level "$MIN_LEVEL"
  --model "$MODEL"
  --host "$HOST"
  --out "$OUT/verifier_study.json"
)
[[ -n "$JUDGE_MODEL" ]] && ARGS+=(--judge-model "$JUDGE_MODEL")

"$PY" analysis/verifier_study.py "${ARGS[@]}" | tee "$OUT/verifier_study.log"

write_manifest "$OUT" \
  "stage: m0_verifier_study" \
  "model: $MODEL" \
  "judge_model: ${JUDGE_MODEL:-$MODEL}" \
  "problems: $PROBLEMS" \
  "samples_per_problem: $SAMPLES" \
  "min_level: $MIN_LEVEL"

log "M0 complete. Results in $OUT"
