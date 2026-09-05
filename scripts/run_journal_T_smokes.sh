#!/usr/bin/env bash
# PF-10 — thin T1/T2/T3 OL smokes (dataset_size=1, one sweep point each).
# Full paper runs: same yamls without -v override, bump online_learning.dataset_size to 20.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="experiments/debug_logs"
mkdir -p "$LOG_DIR" experiments/results/smoke_runs/journal_paper/{T1,T2,T3}/{subspacenet,deepcnn}

run_smoke () {
  local name="$1" cfg="$2" axis="$3" val="$4" model="$5" out="$6"
  local log="${LOG_DIR}/smoke_${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== ${name} -> ${out} ===" | tee "$log"
  python3 main_v2.py run -c "$cfg" --goal online_learning --trajectory \
    --sweep 1d --axis "$axis" -v "$val" \
    -m "$model" -o "$out" 2>&1 | tee -a "$log"
}

SN="experiments/results/basemodels_for_journal_paper/subspacenet/snr_10.0/checkpoints/final_SubspaceNet_20250916_084930.pt"
DC="experiments/results/basemodels_for_journal_paper/deepcnn/snr_10.0/checkpoints/final_DeepCNN_20260820_173256.pt"

run_smoke T1_subspacenet configs/Used_for_paper/paper_T1_antenna_sweep.yaml n 9 "$SN" \
  experiments/results/smoke_runs/journal_paper/T1/subspacenet

run_smoke T1_deepcnn configs/Used_for_paper/paper_T1_antenna_sweep_deepcnn.yaml n 9 "$DC" \
  experiments/results/smoke_runs/journal_paper/T1/deepcnn

run_smoke T2_subspacenet configs/Used_for_paper/paper_T2_snr_sweep.yaml snr 10 "$SN" \
  experiments/results/smoke_runs/journal_paper/T2/subspacenet

run_smoke T2_deepcnn configs/Used_for_paper/paper_T2_snr_sweep_deepcnn.yaml snr 10 "$DC" \
  experiments/results/smoke_runs/journal_paper/T2/deepcnn

run_smoke T3_subspacenet configs/Used_for_paper/paper_T3_target_sweep.yaml m 3 "$SN" \
  experiments/results/smoke_runs/journal_paper/T3/subspacenet

run_smoke T3_deepcnn configs/Used_for_paper/paper_T3_target_sweep_deepcnn.yaml m 3 "$DC" \
  experiments/results/smoke_runs/journal_paper/T3/deepcnn

echo "All T1/T2/T3 smokes done."
