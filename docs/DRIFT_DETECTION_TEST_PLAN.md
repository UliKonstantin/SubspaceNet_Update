# Drift detection experiments

Paper (15)–(16): trigger on streaming **z-score of G_w** (max log-GLR), not on τ* ≈ η.

| Symbol | Meaning | Role |
|--------|---------|------|
| **w\*** | Window where `z > z_th` first latches | **Detection time** (latency); starts `time_to_learn` countdown |
| **τ\*** | `argmax_τ` log-GLR on MSIE prefix at w\* | **Estimated changepoint location** (accuracy); can be far in the past |
| **η window** | Scheduled distribution change (config) | Ground truth; not what GLRT optimizes |

Trigger condition today: `z > z_th` only. Problem: z can fire while τ\* sits at an ancient split (e.g. w\*=38, τ\*=16), so adaptation starts late with a stale changepoint story.

Plots now distinguish **w\*** (crimson `:`) vs **τ\* at trigger** (teal `-.`).

---

## Test queue (in order)

### A. Recent-τ gate on trigger *(implemented)*

**Status:** `drift_recent_tau_k` in config (default `None` = off). T1 paper configs use **K=12**.

**Goal:** Cap latency + reject ancient τ\* without changing the GLRT prefix.

**Trigger only if both:**
1. `z > z_th`
2. `argmax τ ≥ n − K` (post-warmup index; changepoint in last **K** MSIE samples of prefix), e.g. **K=5**

**Expected effect:**
- Rejects “split at w=12” when evaluating at w=60 with high z.
- May delay or suppress triggers until τ\* moves into the recent tail.
- Small code change: gate in `Online_learning.py` after z compute; config `drift_recent_tau_k` (default `None` = off).

**Metrics:** trigger rate, mean/median (w\* − η), |τ\* − η|, post-adapt RMSPE vs baseline.

**Smoke:** T1_full_eta45, N∈{6, 18}, SubspaceNet + DeepCNN.

---

### B. Prefix lookback + smaller min segment *(if A insufficient)*

**Goal:** Localize GLRT to recent history so G_w reflects η-adjacent structure.

**Changes:**
- GLRT on **last L=15** MSIE windows only (not full prefix from warmup).
- `min_segment_size = 4` (was 5).

**Config knobs:** `drift_glrt_lookback_windows: 15`, `drift_min_segment_size: 4`.

**Risk:** shorter prefix → noisier G_w / z; needs retune of guard/baseline.

---

### C. Plot markers *(done)*

- **w\***: z-trigger time on window-index axes.
- **τ\***: `glrt_at_detection.changepoint_window` at trigger.
- **Fix:** `glrt_adaptation_at_detection_glrt.png` — no w\* vline on τ-candidate axis; τ\* only.

---

## Implementation notes for A

```python
# After adapt_changepoint computed, before latching w*:
n = len(adaptation_losses)  # post-warmup length
recent_ok = (adapt_changepoint >= n - K) if K is not None else True
if z > z_th and recent_ok and w* is None:
    latch w*
```

Do **not** reset z baseline when gate fails — keep streaming; only suppress latch.

---

## Success criteria (informal)

| Experiment | Pass hint |
|------------|-----------|
| A | Fewer triggers with τ\* ≪ η; w\* closer to η without killing recall on N=6 η@45 |
| B | G_w peak near η; N=18 starts triggering where A still fails |
| C | Plots readable; τ\* and w\* visually separable |
