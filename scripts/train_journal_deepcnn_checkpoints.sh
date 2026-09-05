#!/usr/bin/env bash
# PF-09 — train journal paper DeepCNN checkpoints for T1/T3 sweeps.
# Full-quality: paper_deepcnn_training.yaml (10 epochs, 1024 samples).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="configs/Used_for_paper/paper_deepcnn_training.yaml"

output_dir_for () {
  local n="$1" m="$2" snr="${3:-10}"
  if [[ "$n" == "9" && "$m" == "3" ]]; then
    echo "experiments/results/basemodels_for_journal_paper/deepcnn/snr_${snr}.0"
  else
    echo "experiments/results/basemodels_for_journal_paper/deepcnn/N${n}_M${m}_SNR${snr}.0"
  fi
}

train_variant () {
  local label="$1" n="$2" m="$3"
  shift 3
  local out
  out="$(output_dir_for "$n" "$m")"
  echo "=== Training DeepCNN $label -> $out ==="
  python3 main_v2.py run -c "$CFG" --goal train -o "$out" "$@"
}

# T1: N sweep (M=3, SNR=10)
train_variant "N=6 M=3"  6 3 -O system_model.N=6
train_variant "N=18 M=3" 18 3 -O system_model.N=18

# T3: M sweep (N=9, SNR=10)
train_variant "N=9 M=2" 9 2 -O system_model.M=2
train_variant "N=9 M=4" 9 4 -O system_model.M=4
train_variant "N=9 M=5" 9 5 -O system_model.M=5

echo "Done. Checkpoints under experiments/results/basemodels_for_journal_paper/deepcnn/"
