# TBD Backlog

Tracked follow-ups after CLI v2 and refactoring. Ordered roughly by dependency / impact.

---

## 1. Embed LR optimality analysis (`scratch.py`)

**Current state:** Root-level one-off script. Reads `lr_sweep_heatmap_data.json`, picks best LR per η, fits log-space sigmoid, plots:
- `optimal_lr_vs_eta.png`
- `loss_vs_eta_per_lr.png`

**Target design:**

| Piece | Location | Notes |
|-------|----------|-------|
| Data loading + best-LR selection | `utils/lr_analysis.py` | `load_heatmap_data(path)`, `best_lr_per_eta(data) -> DataFrame` |
| Sigmoid fit (η → LR*) | `utils/lr_analysis.py` | `fit_eta_to_lr_sigmoid(etas, lrs) -> FitResult` (params + R²) |
| Plots | `utils/plotting.py` | `plot_optimal_lr_vs_eta(...)`, `plot_loss_vs_eta_per_lr(...)` — reuse existing matplotlib style helpers |
| Postprocess hook | `cli/postprocess.py` | When `goal=online_learning`, `sweep=1d`, `axis=eta`, `lr_sweep=True`: call after `plot_lr_sweep_heatmap` |
| Optional CLI | `main_v2.py analyze` or `run --analyze-lr` | Re-run analysis on existing results dir without re-simulating |

**Acceptance:**
- Paper eta+LR sweep produces the two PNGs automatically in output dir
- Unit test on synthetic `lr_sweep_heatmap_data.json` (fit + plot smoke)
- Delete `scratch.py` after cutover

**Related:** `plot_lr_sweep_heatmap` in `utils/plotting.py` already handles heatmap; this adds optimality curve + per-LR loss curves.

---

## 2. Embed GLRT → optimal LR analysis (`scratch_glrt_analysis.py`)

**Current state:** Root-level script. Joins `drift_detection_dicts.json` + `lr_sweep_heatmap_data.json`, fits sigmoid(observable → LR*) for:
- `main_log_glr`
- `glr_diff` (GLR − baseline)

Outputs `glrt_observable_to_optimal_lr.png`.

**Target design:**

| Piece | Location | Notes |
|-------|----------|-------|
| Join heatmap + drift dicts | `utils/lr_analysis.py` | `build_glrt_lr_mapping(heatmap_path, drift_path) -> mapping table` |
| Sigmoid fit (observable → LR*) | `utils/lr_analysis.py` | `fit_observable_to_lr_sigmoid(x, log_lrs) -> FitResult` |
| Compare observables | same module | Return R² for `main_log_glr` vs `glr_diff`; log which wins |
| Plot | `utils/plotting.py` | `plot_glrt_observable_to_optimal_lr(mapping, fits, output_dir)` |
| Postprocess hook | `cli/postprocess.py` | After eta LR sweep if both JSON files exist in `output_dir` |
| Wire to adaptive LR | `simulation/runners/Online_learning.py` | **Optional phase 2:** use fitted sigmoid params from config (`adaptive_lr_*` already in YAML) — validate params match fitted values |

**Acceptance:**
- Automatic plot after OL eta+LR sweep when drift dicts saved
- Test with fixture JSON (2–3 η values)
- Delete `scratch_glrt_analysis.py` after cutover

**Note:** This closes the loop between GLRT drift detection and adaptive LR tuning documented in `GLRT_DRIFT_DETECTION_AND_ADAPTIVE_LEARNING.md`.

---

## 3. Refactor `sandbox.py` (GLRT module)

**Current state:** Misnamed file mixing production GLRT + dev plotting + commented demo + CLI.

**Target design:**

```
simulation/drift/
  glrt.py              ← glrt_changepoint_detection (from sandbox)
  plotting.py          ← plot_results (GLRT loss/GLRT stat figures)
  drift_metrics.py     ← plot_drift_detection_metrics (optional CLI)
```

- Update imports in `Online_learning.py`, `tests/test_integration.py`
- Remove ~150 lines of commented demo code
- Keep backward compat alias `simulation.runners.sandbox` → re-export from `drift/` for one release, then remove

---

## 4. Dead code cleanup

| Item | Action |
|------|--------|
| `experiments/runner.py` | Delete; remove unused import from `main.py` |
| `main.py` dead imports | Remove `save_config`, `run_experiment`, unused top-level `plot_scenario_results` |
| `simulation/losses/` | Delete empty package or add real loss modules when needed |
| `get_kalman_filter` helpers | Keep if public API; else remove unused exports from `kalman_filter/__init__.py` |
| `tests/kalman_filter/standalone_test_*.py` | Move to `tests/kalman_filter/legacy/` or delete if redundant with `test_extended.py` |

---

## 5. CLI v2 cutover (Phase 6)

- [ ] Update `.vscode/launch.json` → `main_v2.py run ...`
- [ ] Deprecate `main.py` commands (banner warning)
- [ ] Remove duplicate sweep logic from `main.py` after parity sign-off
- [ ] Full paper config manual run: 6-eta + LR sweep numeric + plot parity

**Already done:** thin-config parity, paper-config numeric parity (trimmed), 51 tests.

---

## 6. Unified plot dispatch (end-of-recipe refactor)

**Current state:** Plotting is **spread across** `cli/postprocess.py` (sweeps + partial single eval), `Online_learning.py` (GLRT, averaged OL, trajectory, KF comparison), and `training.py` (loss curves). Adding a plot requires hunting call sites.

**Target:**
- Runners / `Simulation` **return structured results only** (no matplotlib).
- **One dispatch** after run completes: `cli/postprocess.py` or `utils/plotting/dispatch.py` registry keyed by `(goal, sweep_type)`.
- **`simulation.save_plots`** gates all PNG generation in that single place.
- New plot = implement in `utils/plotting/` + register — no runner edits.

**Interim (2026-08):** `eval_kf_gain_comparison.png` (single eval, postprocess); `averaged_kf_gain_comparison.png` (OL, still inline in runner). Move both to registry during this task.

**When:** After recipe Steps 0–9 pass; re-run Steps 3–9 for plot regression. See [RECIPE_VERIFICATION_PLAN.md](./RECIPE_VERIFICATION_PLAN.md) plotting gap section.

---

## 7. Stage 10 — split `utils/plotting.py`

Large file (~2200 lines). Split by domain after scratch plots are moved in:

```
utils/plotting/
  __init__.py          ← re-export public API
  evaluation.py
  online_learning.py
  sweeps.py
  lr_analysis.py       ← plots from tasks 1–2
  style.py             ← shared rcParams / serif theme (dedupe scratch style blocks)
```

Do **after** tasks 1–2 so new plots land in the right place.

---

## 8. Stage 2 — import cleanup

- Remove duplicate imports across runners
- Lazy imports in CLI (`run --help` still loads torch today)
- Fix `tests/kalman_filter/*` collection error (`ModuleNotFoundError: src` — DCD_MUSIC path)

---

## 9. Repo hygiene

| Item | Action |
|------|--------|
| `experiments/debug_logs/` | Add to `.gitignore`; optional rotation in EKF logger |
| `diagrams/` | Commit source (`.tex`) only; gitignore build artifacts (`.aux`, `.log`, …) |
| `simulation/runners/.png` | Delete orphan file |
| `.DS_Store` | gitignore |

---

## 10. Testing gaps

- [ ] Full LR-sweep paper config numeric parity (legacy vs v2, all 6 η × 7 LR runs)
- [ ] Kalman 2D evaluate e2e test
- [ ] 4D grid result shape + postprocess plot test
- [ ] Tests for tasks 1–2 (LR/GLRT analysis utils)

---

## Suggested execution order

```
1–2  LR/GLRT analysis embed (user priority)
3    sandbox → drift/ refactor (unblocks clean imports)
4    dead code cleanup (quick win)
6    unified plot dispatch (after recipe 0–9)
5    CLI cutover
7–8  plotting split + import cleanup (during/after #6)
9    repo hygiene (anytime)
10   extended tests (parallel with above)
```

---

## Risk register (carry forward)

- `goal=None` legacy path in `_run_sweep_iteration` can diverge from explicit `--goal` if YAML flags disagree
- Grid 4D results shape differs from kalman_2d (`sim.results` not always set)
- `main.py` remains authoritative until task 5 cutover
