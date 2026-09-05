#!/usr/bin/env bash
# PF-10 — run SubspaceNet + DeepCNN benchmark pair (same seed, separate OL jobs).
# Writes benchmark_metrics.json in each arm dir and benchmark_pair.json in OUT_ROOT.
#
# Usage:
#   ./scripts/run_paper_benchmark_pair.sh \
#     configs/Used_for_paper/paper_experimental_smoke.yaml \
#     configs/Used_for_paper/paper_experimental_baseline_deepcnn.yaml \
#     experiments/results/smoke_runs/journal_paper/PF-10
#
# Optional env: SEED=42 (default 42)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SN_CFG="${1:-configs/Used_for_paper/paper_experimental_smoke.yaml}"
CNN_CFG="${2:-configs/Used_for_paper/paper_experimental_baseline_deepcnn.yaml}"
OUT_ROOT="${3:-experiments/results/smoke_runs/journal_paper/PF-10}"
SEED="${SEED:-42}"

SN_OUT="${OUT_ROOT}/subspacenet"
CNN_OUT="${OUT_ROOT}/deepcnn"

echo "=== PF-10 benchmark pair seed=${SEED} ==="
echo "SubspaceNet config: ${SN_CFG}"
echo "DeepCNN config:     ${CNN_CFG}"
echo "Output root:        ${OUT_ROOT}"

python3 main_v2.py run -c "${SN_CFG}" --goal online_learning --trajectory \
  -O "simulation.seed=${SEED}" \
  -o "${SN_OUT}"

python3 main_v2.py run -c "${CNN_CFG}" --goal online_learning --trajectory \
  -O "simulation.seed=${SEED}" \
  -o "${CNN_OUT}"

python3 - <<PY
from pathlib import Path
from utils.benchmark_export import merge_benchmark_pair

merge_benchmark_pair(
    Path("${SN_OUT}"),
    Path("${CNN_OUT}"),
    Path("${OUT_ROOT}"),
    seed=int("${SEED}"),
)
print("Merged benchmark_pair.json")
PY

echo "Done."
echo "  ${SN_OUT}/benchmark_metrics.json"
echo "  ${CNN_OUT}/benchmark_metrics.json"
echo "  ${OUT_ROOT}/benchmark_pair.json"
