#!/usr/bin/env bash
# T1 full-axis OL smoke (N=6,9,18), dataset_size=1, eta jump to 0.9
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="experiments/debug_logs"
mkdir -p "$LOG_DIR"
OUT="experiments/results/smoke_runs/journal_paper/T1_full"
LOG="${LOG_DIR}/smoke_T1_eta09_$(date +%Y%m%d_%H%M%S).log"

run () {
  local name="$1" cfg="$2" sub="$3"
  echo "=== ${name} ===" | tee -a "$LOG"
  python3 main_v2.py run -c "$cfg" --goal online_learning --trajectory \
    -o "${OUT}/${sub}" 2>&1 | tee -a "$LOG"
  local exit_code=${PIPESTATUS[0]}
  if [[ "$exit_code" -ne 0 ]]; then
    echo "FAILED: ${name} (exit ${exit_code})" | tee -a "$LOG"
    exit "$exit_code"
  fi
}

run T1_subspacenet configs/Used_for_paper/paper_T1_antenna_sweep.yaml subspacenet
run T1_deepcnn configs/Used_for_paper/paper_T1_antenna_sweep_deepcnn.yaml deepcnn

python3 - <<PY | tee -a "$LOG"
from pathlib import Path
from utils.benchmark_export import merge_benchmark_pair
from utils.plotting.benchmark_pair import plot_benchmark_pair_comparisons

root = Path("${OUT}")
merge_benchmark_pair(root / "subspacenet", root / "deepcnn", root, seed=42)
paths = plot_benchmark_pair_comparisons(root, root, axis="n", scenario=9)
for k, p in paths.items():
    print(f"{k}: {p}")
PY

echo "T1 smoke done -> ${OUT}" | tee -a "$LOG"
