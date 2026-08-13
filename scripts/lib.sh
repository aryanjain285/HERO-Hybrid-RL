#!/usr/bin/env bash
# Shared helpers for the HERO pipeline scripts. Sourced, never executed.
#
# Every script that sources this file inherits strict mode, so an unset variable
# or a failed command in a pipeline aborts rather than silently continuing with
# half-built state.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HERO_VENV:-$REPO_ROOT/.venv}"
RUNS_DIR="${HERO_RUNS:-$REPO_ROOT/runs}"

# Target platforms are Linux and macOS (bin/), but resolving Scripts/ as well lets
# the CPU-only stages be exercised under Git Bash, which is what made it possible
# to test these scripts before handing them to a VM.
#
# Must be re-invoked after creating a virtualenv: at source time the interpreter
# does not exist yet, so the layout cannot be detected.
resolve_python() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PY="$VENV_DIR/bin/python"
  elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
    PY="$VENV_DIR/Scripts/python.exe"
  else
    PY="$VENV_DIR/bin/python"  # absent; setup.sh will create it
    return 1
  fi
  return 0
}

resolve_python || true

log()  { printf '\033[1;34m[hero]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

require_venv() {
  [[ -x "$PY" ]] || die "no virtualenv at $VENV_DIR. Run scripts/setup.sh first."
}

# Confirms the package imports before a long job starts, so a broken install
# fails in seconds rather than after model downloads.
require_package() {
  require_venv
  "$PY" -c 'import hero' 2>/dev/null || die "hero package not importable. Re-run scripts/setup.sh."
}

require_gpu() {
  have nvidia-smi || die "nvidia-smi not found; this stage needs an NVIDIA GPU."
  local count
  count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
  (( count > 0 )) || die "no GPU visible to nvidia-smi."
  log "GPUs visible: $count"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/       /'
}

# Fails early and loudly on a missing credential rather than midway through a
# paid job with half the requests spent.
require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "environment variable $name is not set. See docs/pipeline.md."
}

run_dir() {
  local name="$1"
  local dir="$RUNS_DIR/$name"
  mkdir -p "$dir"
  printf '%s' "$dir"
}

# Records exactly what produced a result, so a stale artefact is identifiable.
write_manifest() {
  local dir="$1"; shift
  {
    printf 'timestamp_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_sha: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'git_dirty: %s\n' "$(git -C "$REPO_ROOT" diff --quiet 2>/dev/null && echo no || echo yes)"
    printf 'host: %s\n' "$(uname -sr)"
    printf 'python: %s\n' "$("$PY" --version 2>&1)"
    for kv in "$@"; do printf '%s\n' "$kv"; done
  } > "$dir/manifest.yaml"
  log "manifest -> $dir/manifest.yaml"
}
