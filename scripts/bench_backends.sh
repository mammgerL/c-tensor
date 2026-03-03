#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing command: $1" >&2
        exit 1
    fi
}

need_cmd make
need_cmd python3

PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
fi

ensure_data() {
    if [[ -f mnist_train.csv && -f mnist_test.csv ]]; then
        echo "[info] dataset exists: mnist_train.csv + mnist_test.csv"
        return
    fi

    echo "[info] dataset missing, trying to generate via make data"
    if ! "$PYTHON_BIN" -c 'import torch, torchvision, numpy' >/dev/null 2>&1; then
        echo "[error] Python deps missing. install first:" >&2
        echo "        pip install -r requirements.txt" >&2
        echo "        pip install torch torchvision" >&2
        exit 1
    fi
    make data PYTHON="$PYTHON_BIN"
}

run_accelerate() {
    echo "[bench] Accelerate build/run"
    make clean >/dev/null
    make train >/dev/null
    /usr/bin/time -p ./train 2>&1 | tee bench_accelerate.log
}

run_openmp() {
    echo "[bench] OpenMP build/run"
    make clean >/dev/null
    if ! make openmp >/dev/null 2>&1; then
        echo "[error] make openmp failed. On macOS install libomp:" >&2
        echo "        brew install libomp" >&2
        exit 1
    fi
    /usr/bin/time -p ./train 2>&1 | tee bench_openmp.log
}

summarize() {
    local accel_elapsed openmp_elapsed
    accel_elapsed="$(rg -o 'Elapsed Time: [0-9.]+ seconds' bench_accelerate.log | awk '{print $3}' | tail -n1 || true)"
    openmp_elapsed="$(rg -o 'Elapsed Time: [0-9.]+ seconds' bench_openmp.log | awk '{print $3}' | tail -n1 || true)"

    echo
    echo "========== Benchmark Summary =========="
    if [[ -n "$accel_elapsed" ]]; then
        echo "Accelerate elapsed: ${accel_elapsed}s"
    else
        echo "Accelerate elapsed: not found in log"
    fi

    if [[ -n "$openmp_elapsed" ]]; then
        echo "OpenMP elapsed:    ${openmp_elapsed}s"
    else
        echo "OpenMP elapsed:    not found in log"
    fi

    if [[ -n "$accel_elapsed" && -n "$openmp_elapsed" ]]; then
        python3 - <<PY
acc = float("$accel_elapsed")
omp = float("$openmp_elapsed")
if acc > 0:
    print(f"Speedup (OpenMP / Accelerate): {omp/acc:.2f}x")
else:
    print("Speedup: undefined (accelerate time <= 0)")
PY
    fi

    echo "Logs: bench_accelerate.log, bench_openmp.log"
}

ensure_data
run_accelerate
run_openmp
summarize
