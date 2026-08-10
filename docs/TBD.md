# TBD Backlog

Tracked follow-ups after CLI v2 and refactoring. Ordered roughly by dependency / impact.

---

## 1. Embed LR optimality analysis (`scratch.py`)

**Status:** Done (2026-08-09)

- `utils/lr_analysis.py` — data loading, best-LR selection, sigmoid fits
- `utils/plotting.py` — `plot_optimal_lr_vs_eta`, `plot_loss_vs_eta_per_lr`
- Auto-generated after eta+LR sweep via `cli/postprocess.py` + `main.py`
- Tests: `tests/test_lr_analysis.py`
- `scratch.py` deleted

---

## 2. Embed GLRT → optimal LR analysis (`scratch_glrt_analysis.py`)

**Status:** Done (2026-08-09)

- `build_glrt_lr_mapping` + `plot_glrt_observable_to_optimal_lr` in same postprocess hook
- Reads `drift_detection_dicts.json` from output dir when present
- Output: `glrt_observable_to_optimal_lr.png`
- `scratch_glrt_analysis.py` deleted

Adaptive LR sigmoid params remain **manual YAML tuning** (`adaptive_lr_*`); no auto-calibration from sweep fits.

---

## 3. Refactor `sandbox.py` (GLRT module)

**Status:** Done (2026-08-09)

```
simulation/drift/
  glrt.py              ← glrt_changepoint_detection
  plotting.py          ← plot_results
  drift_metrics.py     ← plot_drift_detection_metrics (+ pipeline hook)
```

- `simulation.runners.sandbox` re-exports from `simulation.drift` (CLI preserved)
- `plot_drift_detection_metrics_in_output_dir` wired in `utils/plot_dispatch.py` for 1D OL sweeps
- Output: `drift_detection_metrics_vs_eta.png` when `drift_detection_dicts.json` exists
- Tests: `tests/test_drift_metrics.py`

---

## 4. Dead code cleanup

**Status:** Done (2026-08-09)

| Item | Action |
|------|--------|
| `experiments/runner.py` | Already deleted (CLI v2 cutover) |
| `main.py` dead imports | Already removed; module deprecated (see header) |
| `simulation/losses/` | Deleted empty package |
| `get_kalman_filter` helpers | Kept — used by `tests/kalman_filter/test_helpers.py` |
| `tests/kalman_filter/standalone_test_*.py` | Moved to `tests/kalman_filter/legacy/` |
| `simulation/runners/.png` | Deleted orphan file |
| `data.py` inline trajectory verification PNG | Removed; no plotting from data loader |

---

## 5. CLI v2 cutover (Phase 6)

**Status:** Done (2026-08-09)

- [x] `.vscode/launch.json` → `main_v2.py run ...` recipes
- [x] Deprecate `main.py` commands (`cli/legacy_bridge.warn_deprecated`)
- [x] Remove duplicate sweep plotting from `main.py` → `legacy_postprocess` → `plot_dispatch`
- [x] Delete unused `experiments/runner.py`
- [x] Full paper config: `experiments/results/paper_eta_sweep_v3/` — 6 η × 7 LR, all aggregate + per-subdir plots

**Already done:** thin-config parity, paper-config numeric parity (trimmed), 51+ tests.

---

## 6. Unified plot dispatch (end-of-recipe refactor)

**Status:** Done (2026-08-09)

- `utils/plot_dispatch.py` — registry keyed by `(goal, sweep_type)`, `save_plots` gate
- Runners return structured results only; `cli/postprocess.py` delegates to dispatch
- Per-subdir OL plots restored for 1D sweeps (`dispatch_one_d_sweep_iteration_plots`)
- Tests: `tests/test_plot_dispatch.py`

---

## 7. Stage 10 — split `utils/plotting.py`

**Status:** Done (2026-08-09)

```
utils/plotting/
  __init__.py          ← re-export public API (backward compat)
  evaluation.py        ← eval + kalman 2D
  online_learning.py   ← OL averaged, GLRT, KF gain, training curves
  sweeps.py            ← 1D/4D scenario sweeps, heatmap, improvement tables
  lr_plots.py          ← optimal LR / GLRT observable plots
  trajectory.py        ← OL trajectory figures (dispatch-only)
  style.py             ← shared paper rcParams + save_figure
```

Monolithic `utils/plotting.py` removed. Call sites unchanged: `from utils.plotting import …`

---

## 8. Stage 2 — import cleanup

**Status:** Done (2026-08-09)

- [x] `tests/conftest.py` — workspace + DCD_MUSIC on `sys.path` for all tests
- [x] Fix `tests/kalman_filter/test_helpers.py` — patch `create_from_config` (was stale `from_config`)
- [x] Lazy CLI — `_LazyCLI` static command list for top-level `--help`; heavy imports on subcommand only
- [x] Runner import dedupe — `Online_learning.py`, `evaluation.py`, `training.py`

---

## 9. Repo hygiene

**Status:** Done (2026-08-09)

| Item | Action |
|------|--------|
| `experiments/debug_logs/` | Added to `.gitignore` |
| `diagrams/` | Gitignore LaTeX build artifacts (`.aux`, `.log`, …); source `.tex` committed |
| `tmp/`, `.cursor/` | Added to `.gitignore` |
| `.DS_Store` | Already in `.gitignore`; removed from git index |
| Local diagram PDF/PNG binaries | Removed from working tree (regenerate from `.tex`) |

---

## 10. Testing gaps (deferred — do last)

**Status:** Backlog

- [ ] **Full LR-sweep paper config numeric parity** (legacy vs v2, all 6 η × 7 LR runs) — only item worth heavy investment
- [ ] Kalman 2D evaluate e2e test (low priority unless mode is used)
- [ ] 4D grid result shape + postprocess plot test (low priority)
- [x] LR/GLRT analysis utils — `tests/test_lr_analysis.py`

---

## 11. Plot visual polish (paper-ready figures)

**Status:** Done (2026-08-09)

Systematic pass over all pipeline-generated plots for publication quality. **All figures must go through `plot_dispatch` / `utils/plotting/` — no inline `savefig` in runners or data loaders.**

**Architecture:**
- [x] `utils/plotting/style.py` — `apply_paper_plot_style()`, `save_figure()`, shared labels/colors/fig sizes
- [x] OL trajectory plots in `utils/plotting/trajectory.py`, invoked from `plot_dispatch._plot_iteration_if_ol`
- [x] Removed inline trajectory verification plotting from `simulation/runners/data.py`
- [x] Paper style + dpi 300 on main/training loss, KF gain, GLRT, drift metrics, eta scenario comparison, trajectory XY/DOA plots
- [x] Sweep aggregates (`sweeps.py`, `lr_plots.py`) — scenario comparison, heatmap, improvement tables, GLRT violins
- [x] Training/eval (`evaluation.py`, `online_learning.py`) — unified `save_figure` export
- [x] `scripts/replot_paper_aggregates.py` — offline replot from JSON artifacts
- [x] `scenario_results_stub.json` saved during eta-sweep postprocess for offline aggregate replot

**Deferred (non-blocking):**
- Per-subdir OL replot from v3 without full result dicts (re-run sweep or restore stubs)
- Optional PDF export toggle

**Deliverable:** re-dispatch v3 step 8 or `python scripts/replot_paper_aggregates.py <output_dir>` for aggregate refresh

---

## Suggested execution order

```
10   full legacy parity (last; only if regression proof needed)
     DCD_MUSIC submodule cleanup (local dirty state)
```

TBD #11 plot polish is complete. Circle back to DCD_MUSIC + TBD #10 when ready.

---

## Risk register

| Risk | Status |
|------|--------|
| `goal=None` in `_run_sweep_iteration` diverges from `--goal` | **Fixed** — uses `cli.resolver.infer_goal()` before legacy fallback |
| Grid 4D `sim.results` not set (unlike kalman_2d) | **Fixed** — `cli/runner.py` sets `sim.results["grid_4d"]` |
| `main.py` vs `main_v2.py` | **Mitigated** — `main.py` deprecated; header + `warn_deprecated` on use |
| `DCD_MUSIC` submodule dirty | **Local only** — see submodule note below |

**DCD_MUSIC submodule:** parent repo tracks submodule SHA; working tree shows `M src/system_model.py` inside submodule — commit or discard inside `DCD_MUSIC/` separately from parent repo.
