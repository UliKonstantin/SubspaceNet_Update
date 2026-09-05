#!/usr/bin/env bash
# Wait for snr_-5.0 to finish, kill the full 16-SNR batch, then train 0/5/10 only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="experiments/debug_logs/pf09_deepcnn_snr_subset_$(date +%Y%m%d_%H%M%S).log"
BATCH_PID="${1:?batch bash PID required}"

OUT="${ROOT}/experiments/results/basemodels_for_journal_paper/deepcnn/snr_-5.0/checkpoints"
echo "[watch] waiting for snr_-5.0 final (batch PID=${BATCH_PID})" | tee -a "$LOG"
while ! compgen -G "${OUT}/final_DeepCNN_"*.pt > /dev/null; do
  if ! kill -0 "$BATCH_PID" 2>/dev/null; then
    echo "[watch] batch PID ${BATCH_PID} already exited before -5 final appeared" | tee -a "$LOG"
    break
  fi
  sleep 20
done

if kill -0 "$BATCH_PID" 2>/dev/null; then
  echo "[watch] snr_-5.0 done — cancelling batch (PID ${BATCH_PID})" | tee -a "$LOG"
  kill "$BATCH_PID" 2>/dev/null || true
  sleep 3
  pkill -f 'main_v2.py run -c configs/Used_for_paper/paper_deepcnn_training.yaml --goal train -o experiments/results/basemodels_for_journal_paper/deepcnn/snr_-4.0' 2>/dev/null || true
fi

echo "[watch] starting subset train: 0, 5, 10" | tee -a "$LOG"
exec >>"$LOG" 2>&1
for snr in 0 5 10; do
  out="experiments/results/basemodels_for_journal_paper/deepcnn/snr_${snr}.0"
  if compgen -G "${out}/checkpoints/final_DeepCNN_"*.pt > /dev/null; then
    echo "=== Skip SNR=${snr} (final exists) ==="
    continue
  fi
  echo "=== Training DeepCNN N=9 M=3 SNR=${snr} -> ${out} ==="
  python3 main_v2.py run -c configs/Used_for_paper/paper_deepcnn_training.yaml \
    --goal train -o "$out" -O "system_model.snr=${snr}"
done
echo "Subset SNR training done."
