#!/usr/bin/env bash
# Create the virtualenv and install the project. Idempotent and safe to re-run.
#
# CPU-only by default, which is all the audit stages need. Pass --with-gpu to add
# the training stack (torch, vllm, verl); that requires Linux with CUDA and pulls
# several gigabytes.
#
#   scripts/setup.sh                # audit stages only
#   scripts/setup.sh --with-gpu     # plus the training stack
#
# Usable on Linux and macOS. The GPU extras are Linux-only: neither vllm nor verl
# ships macOS or Windows wheels, which is why the audit stack is kept independent
# of them.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WITH_GPU=0
for arg in "$@"; do
  case "$arg" in
    --with-gpu) WITH_GPU=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done

log "repo root: $REPO_ROOT"
cd "$REPO_ROOT"

# --- interpreter -------------------------------------------------------------- #
# 3.11 is the floor: hero.rewards uses StrEnum and match statements.
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if have "$candidate"; then
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [[ "$(printf '%s\n3.11\n' "$ver" | sort -V | head -1)" == "3.11" ]]; then
      PYTHON_BIN="$candidate"; break
    fi
  fi
done
[[ -n "$PYTHON_BIN" ]] || die "need Python >= 3.11 on PATH; found none."
log "interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# --- virtualenv --------------------------------------------------------------- #
if [[ -x "$PY" ]]; then
  log "reusing virtualenv at $VENV_DIR"
else
  log "creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  resolve_python || die "venv created but no interpreter found under $VENV_DIR"
fi
log "venv interpreter: $PY"

"$PY" -m pip install --quiet --upgrade pip setuptools wheel
log "installing hero (editable) with dev extras"
"$PY" -m pip install --quiet -e ".[dev]"

if (( WITH_GPU )); then
  [[ "$(uname -s)" == "Linux" ]] || die "--with-gpu requires Linux; vllm and verl have no macOS wheels."
  require_gpu
  log "installing training stack: torch, vllm, verl"
  # Unpinned deliberately: CUDA wheel compatibility is host-specific. Pin these in
  # requirements-gpu.txt once a target VM image is chosen (decision D-01 depends on
  # the verl commit, so record it there).
  "$PY" -m pip install torch
  "$PY" -m pip install vllm
  "$PY" -m pip install verl
  log "verl version: $("$PY" -c 'import verl; print(getattr(verl, "__version__", "unknown"))')"
  warn "pin the verl commit in requirements-gpu.txt before any real run; audit A-1"
  warn "hinges on algorithm.norm_adv_by_std_in_grpo, whose default may change."
fi

# --- verify ------------------------------------------------------------------- #
log "verifying installation"
"$PY" -c 'import hero; print(f"hero exports {len(hero.__all__)} names")'
log "running test suite"
"$PY" -m pytest tests/ -q

log "setup complete."
log "next: scripts/run_audit.sh   (no GPU needed, ~1 minute)"
