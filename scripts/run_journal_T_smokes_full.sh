#!/usr/bin/env bash
# PF-10 — full-axis T1/T2/T3 OL smokes (all scenario_config.values, dataset_size=1).
# Not thin: no --sweep/-v override; model_paths align with values by index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="experiments/debug_logs"
mkdir -p "$LOG_DIR"

run_full_smoke () {
  local name="$1" cfg="$2" out="$3"
  local log="${LOG_DIR}/smoke_full_${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== FULL ${name} -> ${out} ===" | tee "$log"
  python3 main_v2.py run -c "$cfg" --goal online_learning --trajectory \
    -o "$out" 2>&1 | tee -a "$log"
}

mkdir -p experiments/results/smoke_runs/journal_paper/{T1,T2,T3}_full/{subspacenet,deepcnn}

run_full_smoke T1_subspacenet configs/Used_for_paper/paper_T1_antenna_sweep.yaml \
  experiments/results/smoke_runs/journal_paper/T1_full/subspacenet

run_full_smoke T1_deepcnn configs/Used_for_paper/paper_T1_antenna_sweep_deepcnn.yaml \
  experiments/results/smoke_runs/journal_paper/T1_full/deepcnn

run_full_smoke T2_subspacenet configs/Used_for_paper/paper_T2_snr_sweep.yaml \
  experiments/results/smoke_runs/journal_paper/T2_full/subspacenet

run_full_smoke T2_deepcnn configs/Used_for_paper/paper_T2_snr_sweep_deepcnn.yaml \
  experiments/results/smoke_runs/journal_paper/T2_full/deepcnn

run_full_smoke T3_subspacenet configs/Used_for_paper/paper_T3_target_sweep.yaml \
  experiments/results/smoke_runs/journal_paper/T3_full/subspacenet

run_full_smoke T3_deepcnn configs/Used_for_paper/paper_T3_target_sweep_deepcnn.yaml \
  experiments/results/smoke_runs/journal_paper/T3_full/deepcnn

echo "All full-axis T1/T2/T3 smokes done."
