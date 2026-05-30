# CLI Logic Routing Spec (Decision Tree Only)

Pure routing logic for all intended CLI use cases. No execution mapping, no code references.

---

## Top-Level Intent

```
What is the primary goal?
│
├─ A) Train a base (supervised) model
├─ B) Evaluate a fixed model (no weight updates)
├─ C) Run online learning on a fixed model (weight updates on stream)
└─ D) Run a full pipeline (A, then optionally B, then optionally C)
```

---

## Branch A — Supervised Base Model Training

```
A) Train base model
│
├─ Load existing weights before training?
│   ├─ No  → train from scratch
│   └─ Yes → fine-tune / resume from checkpoint
│
├─ Data generation mode
│   ├─ Static batch dataset (fixed DOA samples per example)
│   └─ Trajectory dataset (time-varying DOA per trajectory)
│
└─ Parameter sweep?
    ├─ No  → single training run at config defaults
    └─ Yes → sweep one system parameter per run
            │
            ├─ Valid sweep axes (1D):
            │   • SNR
            │   • M (number of sources)
            │   • T (snapshots per sample)
            │   • eta (calibration error — if training under mismatch)
            │   • trajectory_length (if trajectory mode)
            │
            └─ Per sweep value — model policy:
                ├─ Retrain a separate model per value
                └─ Reuse one base model (same weights for all values)  ← usually invalid for training unless comparing init only
```

**Invalid under A:**
- Sweep axis = `kalman_noise` (2D KF tuning — evaluation concern, not training)
- Sweep axis = `4d_grid` (online-learning research grid)
- Nested LR sweep (online-learning only)
- `load checkpoint` + `retrain per value=false` without a meaningful comparison goal

---

## Branch B — Evaluate Fixed Model

```
B) Evaluate fixed model
│
├─ Precondition: checkpoint MUST exist (required)
│
├─ Data generation mode
│   ├─ Static batch test set
│   └─ Trajectory test set
│
└─ Parameter sweep?
    ├─ No  → single evaluation at config defaults
    └─ Yes
        │
        ├─ 1D sweep (pick one axis):
        │   • SNR
        │   • M
        │   • T
        │   • eta (calibration / steering-vector mismatch)
        │   • trajectory_length
        │
        └─ 2D sweep (special):
            • kalman_noise
              – axis 1: measurement_noise_std_dev
              – axis 2: process_noise_std_dev
```

**Invalid under B:**
- No checkpoint provided
- Sweep axis = `4d_grid`
- Nested LR sweep
- Weight updates enabled (that is Branch C, not B)
- `retrain per sweep value` (evaluation never trains)

---

## Branch C — Online Learning

```
C) Online learning on fixed model
│
├─ Precondition: checkpoint MUST exist (required)
│
├─ Data mode: streaming trajectory (implicit — not static batch)
│
├─ Online loss / supervision mode (config choice, not routing branch):
│   ├─ Supervised online loss (vs ground truth)
│   └─ Unsupervised online loss (vs pre-EKF / self-reference)
│
├─ Drift trigger: GLRT z-score threshold (always part of OL logic)
│
├─ Learning rate policy
│   ├─ Fixed LR
│   └─ Adaptive LR (sigmoid from GLRT deviation)
│
└─ Parameter sweep?
    ├─ No  → single OL session at config defaults
    └─ Yes
        │
        ├─ 1D sweep (pick one axis):
        │   • eta (primary paper use case — calibration drift injection)
        │   • SNR
        │   • M
        │   • T
        │   • trajectory process noise (via trajectory noise params)
        │   • KF process / measurement noise (single-axis if not 2D/4D)
        │
        ├─ Nested sub-sweep (only when 1D axis = eta):
        │   └─ LR sweep enabled?
        │       ├─ No
        │       └─ Yes
        │           ├─ Static LR list (multiple runs per eta)
        │           └─ Optional: one adaptive-LR run per eta
        │
        └─ 4D grid (special OL research sweep):
            • process_noise (trajectory)
            × kf_process_noise
            × kf_measurement_noise
            × eta
```

**Invalid under C:**
- No checkpoint provided
- Static batch data only (no trajectory stream)
- Nested LR sweep when sweep axis ≠ eta
- `kalman_noise` 2D grid as the top-level sweep pattern (belongs under B; can appear as one axis inside 4D grid)
- Base model training during OL run

---

## Branch D — Full Pipeline

```
D) Full pipeline
│
├─ Stage 1 — Train base model (same choices as Branch A, single run only)
│
├─ Stage 2 — Evaluate? (config flag)
│   ├─ No
│   └─ Yes → same as Branch B (single run, no sweep at pipeline level)
│
└─ Stage 3 — Online learning? (config flag)
    ├─ No
    └─ Yes → same as Branch C (single run, no sweep at pipeline level)
```

**Invalid under D:**
- Start from checkpoint-only with `train=false` (that is B or C, not D)
- Parameter sweep at pipeline level (sweeps belong on A, B, or C branches individually)
- 4D grid or kalman_noise 2D as full-pipeline mode

---

## Unified Sweep Taxonomy (cross-branch)

| Sweep pattern | Dimensions | Valid in A (train) | Valid in B (eval) | Valid in C (OL) |
|---------------|------------|:------------------:|:-----------------:|:---------------:|
| Single run (no sweep) | 0 | ✓ | ✓ | ✓ |
| SNR | 1D | ✓ | ✓ | ✓ |
| M | 1D | ✓ | ✓ | ✓ |
| T | 1D | ✓ | ✓ | ✓ |
| eta | 1D | ✓ | ✓ | ✓ |
| trajectory_length | 1D | ✓ | ✓ | ✓ |
| kalman_noise | 2D | ✗ | ✓ | ✗ |
| eta + LR sub-sweep | 1D + nested | ✗ | ✗ | ✓ |
| 4d_grid | 4D | ✗ | ✗ | ✓ |

---

## Model / Checkpoint Policy Matrix

| Primary goal | Checkpoint required? | Weight updates? | Sweep retrain per value? |
|--------------|---------------------|-----------------|--------------------------|
| A — Train | Optional (fine-tune) | Yes | Optional (usually yes) |
| B — Evaluate | **Required** | No | N/A (never) |
| C — Online learning | **Required** | Yes (on trigger) | N/A (same checkpoint) |
| D — Full pipeline | Optional at start | Yes in stage 1 (and 3 if enabled) | Only if A sub-branch sweeps |

---

## Data Mode Matrix

| Primary goal | Static batch | Trajectory stream |
|--------------|:------------:|:-----------------:|
| A — Train | ✓ | ✓ |
| B — Evaluate | ✓ | ✓ |
| C — Online learning | ✗ | **Required** |
| D — Full pipeline | ✓ (stage 1/2) | ✓ (stage 3 if OL enabled) |

---

## Complete Decision Tree (compact)

```
START
│
├─ Goal = Train base model? ────────────────────────────────► Branch A
│     └─ [see Branch A tree]
│
├─ Goal = Evaluate fixed model? ───────────────────────────► Branch B
│     └─ checkpoint? ─ No ──► INVALID
│     └─ checkpoint? ─ Yes ─► [see Branch B tree]
│
├─ Goal = Online learning? ────────────────────────────────► Branch C
│     └─ checkpoint? ─ No ──► INVALID
│     └─ checkpoint? ─ Yes ─► [see Branch C tree]
│
└─ Goal = Full pipeline? ──────────────────────────────────► Branch D
      └─ [see Branch D tree]

Within any branch that allows "Parameter sweep? = Yes":
│
├─ Sweep type?
│   ├─ 1D (SNR | M | T | eta | trajectory_length)
│   ├─ 2D kalman_noise          → only if Goal = Evaluate
│   ├─ 4d_grid                  → only if Goal = Online learning
│   └─ eta + LR nested sweep    → only if Goal = Online learning AND axis = eta
│
└─ Values source?
    ├─ Explicit list (CLI)
    ├─ Config-defined list
    └─ Built-in defaults (fallback)
```

---

## Explicitly Excluded Combinations (never valid)

1. **Evaluate or OL without a checkpoint**
2. **OL on static-only data** (no trajectory)
3. **Training with kalman_noise 2D sweep** as primary sweep
4. **Training or evaluation with 4d_grid sweep**
5. **LR nested sweep outside OL + eta axis**
6. **Evaluation with weight updates**
7. **Full pipeline + top-level parameter sweep** (sweep one branch at a time)
8. **Sweep retrain per value under evaluate-only**
9. **4d_grid under evaluate or train branches**
10. **Conflicting intent:** `train=false` + `goal=train base model`

---

## Minimal CLI Surface (logical, not implementation)

One entry point should be enough if it exposes:

| Parameter | Role |
|-----------|------|
| `goal` | train \| evaluate \| online_learning \| full |
| `model_path` | required when goal ∈ {evaluate, online_learning}; optional for train (fine-tune) |
| `data_mode` | static \| trajectory |
| `sweep` | none \| 1d \| 2d_kalman \| 4d_grid |
| `sweep_axis` | snr \| m \| t \| eta \| trajectory_length (when sweep=1d) |
| `sweep_values` | list or config-backed |
| `lr_sweep` | yes/no (only: goal=online_learning AND sweep_axis=eta) |
| `retrain_per_sweep` | yes/no (only: goal=train) |

All current multi-command overlap collapses to different **preset bundles** of these logical parameters — not separate routing trees.
