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

**Optional phase 2:** wire fitted sigmoid params into adaptive LR config in `Online_learning.py`.

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
| `main.py` dead imports | Already removed |
| `simulation/losses/` | Deleted empty package |
| `get_kalman_filter` helpers | Kept — used by `tests/kalman_filter/test_helpers.py` |
| `tests/kalman_filter/standalone_test_*.py` | Moved to `tests/kalman_filter/legacy/` |
| `simulation/runners/.png` | Deleted orphan file |

---

## 5. CLI v2 cutover (Phase 6)

**Status:** Mostly done (2026-08-09)

- [x] `.vscode/launch.json` → `main_v2.py run ...` recipes
- [x] Deprecate `main.py` commands (`cli/legacy_bridge.warn_deprecated`)
- [x] Remove duplicate sweep plotting from `main.py` → `legacy_postprocess` → `plot_dispatch`
- [x] Delete unused `experiments/runner.py`
- [ ] Full paper config manual run: 6-eta + LR sweep numeric + plot parity (use launch config **v2: paper eta sweep**)

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
  online_learning.py   ← OL averaged, GLRT, KF gain, trajectory, training curves
  sweeps.py            ← 1D/4D scenario sweeps, heatmap, improvement tables
  lr_plots.py          ← optimal LR / GLRT observable plots
```

Monolithic `utils/plotting.py` removed. Call sites unchanged: `from utils.plotting import …`

---

## 8. Stage 2 — import cleanup

- Remove duplicate imports across runners
- Lazy imports in CLI (`run --help` still loads torch today)
- Fix `tests/kalman_filter/*` collection error (`ModuleNotFoundError: src` — DCD_MUSIC path)

---

## 9. Repo hygiene

**Status:** Done (2026-08-09)

| Item | Action |
|------|--------|
| `experiments/debug_logs/` | Added to `.gitignore` |
| `diagrams/` | Gitignore LaTeX build artifacts (`.aux`, `.log`, …); source `.tex` can be committed |
| `tmp/`, `.cursor/` | Added to `.gitignore` |
| `.DS_Store` | Already in `.gitignore`; removed from git index |

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
