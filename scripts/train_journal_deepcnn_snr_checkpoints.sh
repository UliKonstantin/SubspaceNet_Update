#!/usr/bin/env bash
# PF-09 T2 — train journal DeepCNN checkpoints per SNR (N=9 M=3).
# Reduced SNR set: -5, 0, 5, 10 (matches trimmed T2 DeepCNN plan).
# Full-quality: paper_deepcnn_training.yaml (10 epochs, 1024 samples).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="configs/Used_for_paper/paper_deepcnn_training.yaml"
OUT_ROOT="experiments/results/basemodels_for_journal_paper/deepcnn"

train_snr () {
  local snr="$1"
  local out="${OUT_ROOT}/snr_${snr}.0"
  if compgen -G "${out}/checkpoints/final_DeepCNN_"*.pt > /dev/null; then
    echo "=== Skip SNR=${snr} (final checkpoint exists in ${out}) ==="
    return 0
  fi
  echo "=== Training DeepCNN N=9 M=3 SNR=${snr} -> ${out} ==="
  python3 main_v2.py run -c "$CFG" --goal train -o "$out" -O "system_model.snr=${snr}"
}

for snr in -5 0 5 10; do
  train_snr "$snr"
done

echo "Done. Checkpoints under ${OUT_ROOT}/snr_*/checkpoints/final_DeepCNN_*.pt"
