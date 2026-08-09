# README Recipe Verification Plan

Manual end-to-end verification of every `main_v2.py` recipe in [README.md](../README.md) before TBD work.

**Strategy:** One fast yaml per recipe (same flags/routing as paper yaml, minimal data). Run via CLI. Confirm exit 0 + expected artifacts + plots look sane. **Tear down completely after each step** before starting the next.

---

## Directories (ephemeral)

| Path | Purpose |
|------|---------|
| `tmp/recipe_verify/configs/` | Fast yaml clones (create during plan execution) |
| `tmp/recipe_verify/runs/` | Per-step output only (`step_XX_name/`) |

**Do not commit** `tmp/recipe_verify/` contents. Add to `.gitignore` later if needed.

---

## Paper checkpoint (eval / OL steps only)

Steps **3+** that need a loaded model use the **existing paper checkpoint** — not anything produced by Step 0 or other train steps.

Same path as `configs/Used_for_paper/SineAccel_base_model_Online_learning_eta_sweep_config.yaml` (`scenario_config.model_paths[0]`):

```bash
export PAPER_MODEL="experiments/results/base_model_random_data_snr_10_SubspaceNet_esprit_N9_M3_SNR10.0_Far_ESPRIT/checkpoints/final_SubspaceNet_20250916_084930.pt"
```

**Preflight (before Step 3):**
```bash
test -f "$PAPER_MODEL" || echo "MISSING: paper checkpoint required for eval/OL steps"
```

| Steps | Model source |
|-------|----------------|
| **0, 1, 2** (train) | Train from scratch; checkpoints are **throwaway** — deleted on cleanup |
| **3–6, 9** | `-m "$PAPER_MODEL"` |
| **7–8** | `scenario_config.model_paths: ["$PAPER_MODEL"]` in fast yaml |

**Never delete** `$PAPER_MODEL` during cleanup.

---

## Global trim rules (all fast yamls)

Apply these vs paper configs unless a step says otherwise:

| Field | Paper-ish | Fast verify |
|-------|-----------|-------------|
| `dataset.samples_size` | 256–1024 | **1–2** |
| `training.epochs` | 10–30 | **1** |
| `training.batch_size` | 256 | **2** |
| `trajectory.trajectory_length` | 100–300 | **12–20** (enough windows for GLRT + OL) |
| `online_learning.dataset_size` | 1–10 | **1–2** |
| `online_learning.window_size` / `stride` | 5 / 3 | **3 / 2** |
| `online_learning.time_to_learn` | 3–10 | **2** (learning starts early) |
| `online_learning.max_iterations` | 2–10 | **1–2** |
| `online_learning.enable_lr_sweep` | true (paper) | **false** except Step 8 |
| `scenario_config.values` | 6+ | **2** values only |
| `evaluation.sweep_values` | 7+ SNRs | **2** values: e.g. `[0, 10]` |
| `logging.level` | INFO | **WARNING** |
| `simulation.save_plots` | varies | **true** when step expects plots |
| `simulation.plot_results` | varies | **true** when step expects plots |

**Trajectory length note:** Use at least `(time_to_learn + window_size + 3)` steps so drift detection + at least one OL update can fire.

---

## Success criteria (every step)

1. **CLI:** `python3 main_v2.py run ...` exits **0**, log ends with `Run completed successfully`.
2. **No errors:** No traceback; no `status: error` in returned pipeline (check log).
3. **Artifacts:** Files listed in step table exist under `-o` output dir.
4. **Plots:** PNGs open without empty/broken figures; axes have data (not flat NaN lines); titles match scenario (SNR/eta/kalman/training).
5. **Logic smoke:** Result dict keys match sweep values (e.g. 2 SNR → 2 subdirs or 2 top-level keys).

**Plotting gap (document, don’t fail steps yet):** Plot dispatch is **split across the codebase** today. Do **not** block recipe steps on fixing this; capture artifacts as-is, then consolidate in the **end-of-plan plotting pass** (see below).

| Where plots fire today | Examples |
|------------------------|----------|
| `cli/postprocess.py` | Sweep-only: `loss_vs_snr.png`, kalman 2D heatmap, eta/LR sweep, 4D grid; single eval KF plot (partial) |
| `simulation/runners/Online_learning.py` | GLRT PNGs, trajectory PNGs, calls into `plot_averaged_*` mid-pipeline |
| `simulation/runners/training.py` | Training loss/accuracy curves |
| `simulation/core.py` | (metrics only; JSON save on eval — no plot yet) |

**Target architecture (end-of-plan refactor — TBD #6 + #7):**

1. **Runners return data only** — no matplotlib in `Online_learning.py`, `core.py`, or sweep loops.
2. **Single dispatch point:** `cli/postprocess.py` (or `utils/plotting/dispatch.py`) receives `(result, sim.results, request, output_dir)` **after** the run completes.
3. **Registry by goal + sweep:** e.g. `POSTPROCESS_PLOTS[Goal.ONLINE_LEARNING][SweepType.NONE]` → list of plot functions + required result keys.
4. **`simulation.save_plots` gates all PNG generation** (one flag, one place).
5. **Adding a new plot** = one function in `utils/plotting/` + one registry entry — no runner edits.

**When to do it:** After Steps 0–9 verification passes (or pauses at a stable checkpoint). Re-run **Step 3, 4, 5, 6, 7, 8, 9** once after refactor to regenerate all expected PNGs from the unified dispatch and update execution log plot checks.

**Interim (current session):** KF comparison plots were added ad hoc (`averaged_kf_gain_comparison.png` in OL runner path; `eval_kf_gain_comparison.png` in postprocess for single eval). Treat as **provisional** until the consolidation pass moves them into the registry.

---

## Execution order

Run **sequentially**. Complete cleanup before next step.

```
Step 0  →  A · fast train smoke (throwaway ckpt — not reused)
Step 1  →  A · single train
Step 2  →  A · SNR train sweep
Step 3  →  B · single eval          (-m $PAPER_MODEL)
Step 4  →  B · SNR eval sweep       (-m $PAPER_MODEL)
Step 5  →  B · kalman 2D eval       (-m $PAPER_MODEL)
Step 6  →  C · single OL            (-m $PAPER_MODEL)
Step 7  →  C · eta sweep (no LR)    (model_paths = $PAPER_MODEL)
Step 8  →  C · eta + LR sweep       (model_paths = $PAPER_MODEL)
Step 9  →  C · 4D grid              (-m $PAPER_MODEL)
Step 10 →  D · full pipeline (optional / expect partial)
```

---

## Step 0 — A · Fast train smoke

Verifies the training pipeline end-to-end. Checkpoint is **discarded** after cleanup; later steps use `$PAPER_MODEL`.

| Item | Value |
|------|-------|
| **Source yaml** | `configs/Used_for_paper/Random_basemodel_training_config.yaml` |
| **Fast config** | `tmp/recipe_verify/configs/00_train_smoke.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/00_train_smoke.yaml --goal train -o tmp/recipe_verify/runs/step_00_train_smoke` |
| **Trim** | Global rules; `epochs: 3`, `samples_size: 2`, `trajectory_length: 10`, `save_model: true`, `save_plots: true` |

**Expect:**
- `checkpoints/saved_SubspaceNet_*.pt` or `best_*.pt`
- `plots/loss_curve_*.png`, `plots/accuracy_curve_*.png`

**Cleanup (include checkpoint — do not keep for later steps):**
```bash
rm -rf tmp/recipe_verify/runs/step_00_train_smoke
rm -f tmp/recipe_verify/configs/00_train_smoke.yaml
```

---

## Step 1 — A · Single train

| Item | Value |
|------|-------|
| **README** | `Random_basemodel_training_config.yaml --goal train` |
| **Fast config** | `tmp/recipe_verify/configs/01_train_single.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/01_train_single.yaml --goal train -o tmp/recipe_verify/runs/step_01_train_single` |

**Clone from:** `configs/Used_for_paper/Random_basemodel_training_config.yaml`  
**Must keep:** `simulation.train_model: true`, `trajectory.enabled: true`, `training.enabled: true`  
**Trim:** global rules; `save_plots: true`, `save_model: true`

**Expect:**
- `checkpoints/*.pt`
- `plots/loss_curve_*.png`, `plots/accuracy_curve_*.png`
- Loss curve: 1 epoch, finite y-values (rad scale ~0.1–2)

**Cleanup (include any checkpoints produced — throwaway):**
```bash
rm -rf tmp/recipe_verify/runs/step_01_train_single
rm -f tmp/recipe_verify/configs/01_train_single.yaml
```

---

## Step 2 — A · SNR train sweep

| Item | Value |
|------|-------|
| **README** | `Random_base_model_training_snr_scenario_config.yaml --goal train --sweep 1d --axis snr -v ...` |
| **Fast config** | `tmp/recipe_verify/configs/02_train_snr_sweep.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/02_train_snr_sweep.yaml --goal train --sweep 1d --axis snr -v 5 -v 10 -o tmp/recipe_verify/runs/step_02_train_snr` |

**Clone from:** `configs/Used_for_paper/Random_base_model_training_snr_scenario_config.yaml`  
**Trim:** 2 SNR values via CLI only (yaml can keep defaults); 1 epoch; small dataset  
**Note:** No `scenario_config` required when passing `-v` on CLI.

**Expect:**
- Subdirs: `snr_5.0/`, `snr_10.0/` (float tags)
- Each contains `checkpoints/*.pt` if `save_model: true`
- Optional training plots per subdir

**Cleanup (include any checkpoints produced — throwaway):**
```bash
rm -rf tmp/recipe_verify/runs/step_02_train_snr
rm -f tmp/recipe_verify/configs/02_train_snr_sweep.yaml
```

---

## Step 3 — B · Single eval

| Item | Value |
|------|-------|
| **README** | `default_eval_config.yaml --goal evaluate -m CHECKPOINT` |
| **Fast config** | `tmp/recipe_verify/configs/03_eval_single.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/03_eval_single.yaml --goal evaluate -m "$PAPER_MODEL" -o tmp/recipe_verify/runs/step_03_eval_single` |

**Clone from:** `configs/evaluation_configs/default_eval_config.yaml`  
**Trim:** `samples_size: 2`, `trajectory_length: 12`, `training.enabled: false`, `evaluate_model: true`, `load_model: true`, `save_plots: true`, `evaluation.save_results: true`  
**Plots (interim — postprocess + core):** `eval_kf_gain_comparison.png` when `save_plots: true`; `evaluation_results_*.json` when `save_results: true`. Full dispatch consolidation is **end-of-plan**.

**Expect:**
- No crash loading checkpoint
- Log shows evaluation completed (scalar `dnn_test_loss` + `ekf_test_loss`)
- Optional: per-step DNN vs EKF plot + JSON with `dnn_trajectory_results`

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_03_eval_single
rm -f tmp/recipe_verify/configs/03_eval_single.yaml
```

---

## Step 4 — B · SNR eval sweep

| Item | Value |
|------|-------|
| **README** | `snr_sweep_config.yaml --goal evaluate -m MODEL --sweep 1d --axis snr` |
| **Fast config** | `tmp/recipe_verify/configs/04_eval_snr_sweep.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/04_eval_snr_sweep.yaml --goal evaluate -m "$PAPER_MODEL" --sweep 1d --axis snr -o tmp/recipe_verify/runs/step_04_eval_snr` |

**Clone from:** `configs/evaluation_configs/snr_sweep_config.yaml`  
**Trim:** `evaluation.sweep_values: [0, 10]` (2 values); small dataset; `save_plots: true`

**Expect (postprocess):**
- `loss_vs_snr.png` in output root
- Plot: 2 points, monotonic-ish loss vs SNR (not required — just non-empty)

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_04_eval_snr
rm -f tmp/recipe_verify/configs/04_eval_snr_sweep.yaml
```

---

## Step 5 — B · Kalman 2D eval

| Item | Value |
|------|-------|
| **README** | `default_eval_config.yaml --goal evaluate -m MODEL --sweep 2d_kalman` |
| **Fast config** | `tmp/recipe_verify/configs/05_eval_kalman_2d.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/05_eval_kalman_2d.yaml --goal evaluate -m "$PAPER_MODEL" --sweep 2d_kalman -o tmp/recipe_verify/runs/step_05_eval_kalman2d` |

**Clone from:** `configs/evaluation_configs/default_eval_config.yaml`  
**Trim:** Use CLI defaults for 2D grid (small grid in `cli/constants.py`); shorten trajectory  
**Optional CLI trim:** pass smaller grids if flags exist, else accept ~7×7 default (slowish — may override via yaml kalman defaults only)

**Expect (postprocess):**
- `kalman_noise_2d_heatmap.png`
- `kalman_noise_analysis.png` (if generated by plotter)

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_05_eval_kalman2d
rm -f tmp/recipe_verify/configs/05_eval_kalman_2d.yaml
```

---

## Step 6 — C · Single online learning

| Item | Value |
|------|-------|
| **README** | `online_learning_config.yaml --goal online_learning --trajectory -m MODEL` |
| **Fast config** | `tmp/recipe_verify/configs/06_ol_single.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/06_ol_single.yaml --goal online_learning -m "$PAPER_MODEL" -o tmp/recipe_verify/runs/step_06_ol_single` |

**Clone from:** `configs/online_learning_config.yaml` (or `tests/configs/test_online_learning_single.yaml` as template)  
**Trim:** 1 trajectory, `trajectory_length: 15`, `time_to_learn: 2`, GLRT thresholds relaxed if needed  
**Plots (interim — runner inline, not postprocess):** Always generated today from `Online_learning.py` when OL completes:

| File | Content |
|------|---------|
| `averaged_online_learning_comparison_main_loss.png` | RMSPE(EKF, GT) — pretrained vs Algorithm 1 |
| `averaged_online_learning_comparison_training_loss.png` | MSIE — pretrained vs Algorithm 1 |
| `averaged_kf_gain_comparison.png` | SubspaceNet-only vs EKF vs GT + KF gain panel |
| `glrt_adaptation_loss_averaged_{loss,glrt}.png` | Post-hoc GLRT on MSIE stream |
| `glrt_reference_metric_averaged_{loss,glrt}.png` | Post-hoc GLRT on supervised EKF stream |
| `online_learning_trajectory_*.png` | If `plot_trajectory: true` |

**Extended handoff runs (optional, same Step 6):** Use CLI overrides for 600-step sine-accel + DC + `doa_range=360` scenarios under `tmp/recipe_verify/runs/step_06_600_handoff_*`. Document params in execution log.

**Expect:**
- `status: success` in log
- `online_learning_results` populated (log or debug)
- At least main/training/KF comparison PNGs present

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_06_ol_single
rm -f tmp/recipe_verify/configs/06_ol_single.yaml
```

---

## Step 7 — C · Eta sweep (no LR)

| Item | Value |
|------|-------|
| **README** | eta sweep without `--lr-sweep` |
| **Fast config** | `tmp/recipe_verify/configs/07_ol_eta_sweep.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/07_ol_eta_sweep.yaml --goal online_learning --sweep 1d --axis eta -o tmp/recipe_verify/runs/step_07_ol_eta` |

**Clone from:** `configs/Used_for_paper/SineAccel_base_model_Online_learning_eta_sweep_config.yaml`  
**Trim:** `scenario_config.values: [0.5, 1.0]`, `enable_lr_sweep: false`, keep paper `model_paths: ["$PAPER_MODEL"]`, 1 trajectory  
**Physics (align with Step 6):** `sine_accel_kappa: [1,-1,0.67]`, `sine_accel_dc_offset_range: [-15,15]`, `kalman_filter.process_noise_std_dev: 0.03` (= `sine_accel_noise_std`), `num_gd_steps: 3`, adaptive LR block, `online_learning.trajectory_length: 120` (≥ `drift_warmup_windows` + margin — short length caused empty scenario/GLRT plots)  
**Trajectory length:** OL uses `online_learning.trajectory_length`; `trajectory.trajectory_length` is fallback for batch eval/train — keep both equal in OL yamls or omit `trajectory.trajectory_length` only if OL-only and eval disabled  
**Template:** extend `tests/configs/test_online_learning_eta_sweep.yaml` with `save_plots: true`

**Expect (postprocess):**
- `eta_scenario_drift_detection_comparison.png`
- `performance_improvement_table_eta.png`
- `scenario_results_comparison.png` (or `glrt_scenario_results.png`)
- Subdirs `eta_0.5/`, `eta_1.0/` (float keys)

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_07_ol_eta
rm -f tmp/recipe_verify/configs/07_ol_eta_sweep.yaml
```

---

## Step 8 — C · Eta + LR sweep (paper-critical)

| Item | Value |
|------|-------|
| **README** | `SineAccel_..._eta_sweep_config.yaml --goal online_learning --sweep 1d --axis eta --lr-sweep` |
| **Fast config** | `tmp/recipe_verify/configs/08_ol_eta_lr_sweep.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/08_ol_eta_lr_sweep.yaml --goal online_learning --sweep 1d --axis eta --lr-sweep -o tmp/recipe_verify/runs/step_08_ol_eta_lr` |

**Clone from:** paper eta sweep yaml  
**Trim:**
- `scenario_config.values: [0.5, 1.0]` (2 η)
- `static_lr_list: [0.001, 0.01]` (2 LR only)
- `enable_lr_sweep: true`
- 1 trajectory, short length
- `model_paths: ["$PAPER_MODEL"]` (paper path, unchanged)

**Expect (postprocess):**
- All Step 7 plots **plus** `lr_sweep_heatmap.png`
- `lr_sweep_heatmap_data.json` in sim results / output dir
- Nested eta/lr subdirs under output

**Not in scope yet (TBD #1–2):** `optimal_lr_vs_eta.png`, `glrt_observable_to_optimal_lr.png` from scratch scripts — do not fail Step 8 if missing.

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_08_ol_eta_lr
rm -f tmp/recipe_verify/configs/08_ol_eta_lr_sweep.yaml
```

---

## Step 9 — C · 4D grid

| Item | Value |
|------|-------|
| **README** | `online_learning_config.yaml --goal online_learning --sweep 4d_grid -m MODEL` |
| **Fast config** | `tmp/recipe_verify/configs/09_ol_4d_grid.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/09_ol_4d_grid.yaml --goal online_learning -m "$PAPER_MODEL" --sweep 4d_grid --eta-values 0.0 --eta-values 0.01 --process-noise-values 0.001 --process-noise-values 0.01 --kf-process-noise-values 0.001 --kf-measurement-noise-values 0.001 -o tmp/recipe_verify/runs/step_09_ol_4d` |

**Clone from:** `configs/online_learning_config.yaml`  
**Trim:** **2×2×1×1** mini grid via CLI flags above (not full 5×3×3×3 paper grid)

**Expect (postprocess):**
- At least one `eta_comparison_pn*_kfpn*_kfmn*_*.png`
- Exit 0; grid cells in result dict

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_09_ol_4d
rm -f tmp/recipe_verify/configs/09_ol_4d_grid.yaml
```

---

## Step 10 — D · Full pipeline (optional)

| Item | Value |
|------|-------|
| **README** | `default_config.yaml --goal full` |
| **Fast config** | `tmp/recipe_verify/configs/10_full.yaml` |
| **Command** | `python3 main_v2.py run -c tmp/recipe_verify/configs/10_full.yaml --goal full -o tmp/recipe_verify/runs/step_10_full` |

**Do not use raw `default_config.yaml`** (`train_model: false`, no OL).  
**Build instead:** merge train + eval + OL flags from Steps 1+3+6 into one yaml with global trim; eval/OL phases use `$PAPER_MODEL` (not a checkpoint produced in-step).

**Expect:** train checkpoint + eval completes + OL completes (longer step).  
**Mark as optional** until README fixed.

**Cleanup:**
```bash
rm -rf tmp/recipe_verify/runs/step_10_full
rm -f tmp/recipe_verify/configs/10_full.yaml
```

---

## Final teardown

After all steps:

```bash
rm -rf tmp/recipe_verify/runs tmp/recipe_verify/configs
# Remove any stray dataset caches if created under data/ (shouldn't with save_dataset: false)
# Do NOT rm $PAPER_MODEL
```

---

## Execution log

### Step 4 — SNR eval sweep
- Date: 2026-05-30
- Commit: `52f41e4` (+ uncommitted fixes: `_resolve_model_path_override`, plot labels, `TestSweepModelPath`)
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/04_eval_snr_sweep.yaml --goal evaluate -m "$PAPER_MODEL" --sweep 1d --axis snr -o tmp/recipe_verify/runs/step_04_eval_snr`
- Exit code: 0
- Artifacts present: [x] yes — `loss_vs_snr.png`, per-SNR subdirs
- Plots OK (visual): [x] yes
- Logic OK: [x] yes — 2 SNR points, DNN+EKF metrics; `-m` no longer overwritten by null yaml path
- Cleanup done: [x] yes — temp configs/runs removed with EKF experiment teardown

### Step 5 — Kalman 2D eval
- Date: 2026-05-30
- Commit: `52f41e4`
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/05_eval_kalman_2d.yaml --goal evaluate -m "$PAPER_MODEL" --sweep 2d_kalman -v 0.1 -v 0.5 -v 1.0 -v 0.1 -v 0.5 -v 1.0 -o tmp/recipe_verify/runs/step_05_eval_kalman2d`
- Exit code: 0
- Artifacts present: [x] yes — `kalman_noise_2d_heatmap.png`, `kalman_noise_analysis.png`, 9× `kalman_noise_m*_p*/` subdirs
- Plots OK (visual): [x] yes — 3×3 heatmaps (DNN/EKF/ESPRIT), non-empty cells
- Logic OK: [x] yes — 3×3 meas×proc grid; EKF ≤ DNN on all 9 cells; sine_accel + extended KF
- Cleanup done: [ ] no — artifacts kept for review

### Step 6 — Single online learning
- Date: 2026-05-30
- Commit: `52f41e4`
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/06_ol_single.yaml --goal online_learning -m "$PAPER_MODEL" -o tmp/recipe_verify/runs/step_06_ol_single`
- Exit code: 0
- Artifacts present: [x] yes — `averaged_online_learning_comparison_main_loss.png`, `averaged_online_learning_comparison_training_loss.png`
- Plots OK (visual): [x] yes — 7 windows processed, drift/η updates fired from window 3+
- Logic OK: [x] yes — OL pipeline completed; supervised + unsupervised + EKF metrics per window
- Cleanup done: [ ] no — artifacts kept for review

### Step 6 — Extended handoff baseline (600-step, z=400, doa360, DC)
- Date: 2026-08-09
- Command: `06_ol_single.yaml` + overrides: `trajectory_length=600`, `drift_z_threshold=400`, `eta_update_interval_windows=500`, `doa_range=360`, `plot_trajectory=true` → `step_06_600_handoff_sine_eta500_dc_doa360_z400`
- Exit code: 0 (~13s)
- Artifacts: [x] GLRT ×4, main/training loss, trajectory PNGs, **`averaged_kf_gain_comparison.png`** (KF analysis)
- Analysis note: MSIE elevated even without drift because SubspaceNet-only error is already large; KF plot separates snapshot vs posterior vs GT
- Cleanup done: [ ] no — kept for paper/debug

### Step 7 — Eta sweep (no LR)
- Date: 2026-08-09
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/07_ol_eta_sweep.yaml --goal online_learning --sweep 1d --axis eta -m "$PAPER_MODEL" -o tmp/recipe_verify/runs/step_07_ol_eta`
- Exit code: 0
- Artifacts present: [x] yes — scenario/drift/GLRT/KF plots per eta subdir
- Plots OK (visual): [x] yes
- Logic OK: [x] yes — live drift after first_z; η@32
- Cleanup done: [ ] no

### Step 8 — Eta + LR sweep
- Date: 2026-08-09
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/08_ol_eta_lr_sweep.yaml --goal online_learning --sweep 1d --axis eta --lr-sweep -m "$PAPER_MODEL" -o tmp/recipe_verify/runs/step_08_ol_eta_lr`
- Exit code: 0
- Artifacts present: [x] yes — `lr_sweep_heatmap.png`, `lr_sweep_heatmap_data.json`
- Plots OK (visual): [x] yes — heatmap populated after window-index loss extract fix
- Logic OK: [x] yes — 2η × (2 static + adaptive)
- Cleanup done: [ ] no

### Step 9 — 4D grid mini
- Date: 2026-08-09
- Command: see Step 9 table (repeat each `--*-values` flag per value)
- Exit code: 0
- Artifacts present: [x] yes — 4 grid cells, `eta_comparison_plots/eta_comparison_pn*_*.png`
- Plots OK (visual): [x] yes — 4D plot adapter for trajectory-based OL results
- Logic OK: [x] yes — 2×2×1×1 mini grid
- Cleanup done: [ ] no

### Step 10 — Full pipeline
- Date: 2026-08-09
- Command: `python3 main_v2.py run -c tmp/recipe_verify/configs/10_full.yaml --goal full -o tmp/recipe_verify/runs/step_10_full`
- Exit code: 0 (~45s with `samples_size: 100`, `epochs: 5`, `batch_size: 10`)
- Phases: train → eval → OL in one `sim.run()`; model handoff in-memory (`load_model: false`)
- Artifacts: [x] checkpoints, loss/accuracy curves (5 epochs, visible markers), eval JSON (`dnn=0.036`, `ekf=0.034`), OL plots (main/training/KF/GLRT ×4)
- Fixes applied this session:
  - Training: record train accuracy in `_train_epoch`; plot 1-based epochs + markers (was empty-looking at 1 epoch)
  - Eval: `BatchExtendedKalmanFilter1D.predict()` no longer flattens masked states to 1D (was `ValueError: batch (30) vs sources (3)`)
- Plots OK (visual): [x] yes
- Logic OK: [x] yes — full pipeline train + eval + OL complete
- Cleanup done: [ ] no

---

## Execution log template

Copy per step:

```markdown
### Step XX — <name>
- Date:
- Commit: `git rev-parse --short HEAD`
- Command: (paste)
- Exit code:
- Artifacts present: [ ] yes  [ ] no — list missing
- Plots OK (visual): [ ] yes  [ ] no — notes
- Logic OK: [ ] yes  [ ] no — notes
- Cleanup done: [ ] yes
```

---

## Mapping to existing automated tests

| Step | Existing pytest (partial coverage) |
|------|-----------------------------------|
| 0, 1 | `tests/test_integration.py::test_training_runs_to_completion` |
| 2 | `tests/test_integration.py::test_training_snr_scenario`, `test_cli_training_parity.py` |
| 6–7 | `tests/test_cli_e2e.py` eta/single OL |
| 8 | `tests/test_cli_paper_parity.py` (trimmed numeric, not full plots) |

This plan adds **plot/visual** verification and covers **eval / kalman / 4D** not fully exercised in CI.

---

## After plan passes

1. **Plot consolidation (priority):** Refactor per [Plotting gap](#plotting-gap-document-dont-fail-steps-yet) — runners return data, `cli/postprocess.py` (or registry) owns all PNGs. Re-run Steps 3–9 for visual regression.
2. Fix README (`default_config` train/full, `-m` requirements).
3. Proceed with [TBD.md](./TBD.md) items 1–2 (LR/GLRT analysis plots into consolidated dispatch).
4. Optionally promote stable fast yamls to `tests/recipe_verify/configs/` (committed) with pytest driver.

---

## Next step (current position)

**Recipe verification complete (Steps 0–10).** Follow-ups: teardown, commit, plot consolidation (TBD #6), LR/GLRT analysis (TBD #1–2).
