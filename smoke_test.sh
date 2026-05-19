#!/usr/bin/env bash
# Smoke-test runner: verifies each training entry-point can complete 1 train+val
# batch without errors.  Run from the repo root.
# Usage: bash smoke_test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
TIMEDOUT=0

# Conda environment that has lightning + torch_scatter + torch_geometric
CONDA_ENV="CityGraphs"
PY="conda run -n $CONDA_ENV python"

# Portable 60-second timeout wrapper
_run_timeout() {
    # Uses perl alarm (available on macOS/Linux) or GNU timeout, falling back
    # to direct exec.  Exit code 124 means timeout (matching GNU timeout).
    local cmd="$*"
    if command -v timeout &>/dev/null; then
        timeout 60 bash -c "$cmd"
    elif command -v gtimeout &>/dev/null; then
        gtimeout 60 bash -c "$cmd"
    else
        # perl-based: SIGALRM exits with code 142; we normalise to 124
        perl -e '
            alarm 60;
            $SIG{ALRM} = sub { exit 124 };
            system(@ARGV) == 0 or exit ($? >> 8 || 1);
        ' -- bash -c "$cmd"
    fi
}

run_smoke() {
    local name="$1"
    local dir="$2"
    local cmd="$3"
    echo ""
    echo "══════════════════════════════════════════"
    echo "  $name"
    echo "══════════════════════════════════════════"
    pushd "$dir" > /dev/null
    set +e
    WANDB_MODE=disabled _run_timeout "conda run -n $CONDA_ENV $cmd" 2>&1
    local code=$?
    set -e
    if [ "$code" -eq 0 ]; then
        echo "  --> PASS"
        PASS=$((PASS+1))
    elif [ "$code" -eq 124 ]; then
        echo "  --> TIMEOUT (>60 s)"
        TIMEDOUT=$((TIMEDOUT+1))
    else
        echo "  --> FAIL (exit $code)"
        FAIL=$((FAIL+1))
    fi
    popd > /dev/null
}

# ── ECHO benchmark ────────────────────────────────────────────────────────────
run_smoke "ECHO ECHO_train.py (sssp, GHR)" \
    "$REPO_ROOT/benchmarks/echo/scripts" \
    "python ECHO_train.py --task sssp --gnn_type GHR --hidden_dim 32 --l_steps 2 --h_steps 2 --lr 1e-3 --batch_size 4 --device cpu --smoke-test"

# ── LRIM benchmark ────────────────────────────────────────────────────────────
run_smoke "LRIM example-setup/LRIM_train.py" \
    "$REPO_ROOT/benchmarks/lrim/example-setup" \
    "python LRIM_train.py --dataset_name lrim_16_0.6_10k --hidden_dim 32 --L_steps 2 --H_steps 2 --batch_size 4 --smoke-test"

# ── LRGB ──────────────────────────────────────────────────────────────────────
run_smoke "LRGB LRGB_train.py" \
    "$REPO_ROOT/benchmarks/lrgb" \
    "python LRGB_train.py --config LRGB.yaml --smoke-test"

# ── RGG ablations ─────────────────────────────────────────────────────────────
run_smoke "RGG OOR_train.py" \
    "$REPO_ROOT/experiments/rgg" \
    "python OOR_train.py --smoke-test"

run_smoke "RGG train_weighted.py" \
    "$REPO_ROOT/experiments/rgg" \
    "python train_weighted.py --smoke-test"

# ── City network ──────────────────────────────────────────────────────────────
run_smoke "City network train.py" \
    "$REPO_ROOT/benchmarks/city_network" \
    "python train.py --smoke-test"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  SMOKE TEST SUMMARY"
echo "══════════════════════════════════════════"
echo "  PASS   : $PASS"
echo "  FAIL   : $FAIL"
echo "  TIMEOUT: $TIMEDOUT"
echo "══════════════════════════════════════════"
