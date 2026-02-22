"""Analyze which GLRT observable best proxies for eta, and fit sigmoid: observable → optimal LR."""
import json
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── Load drift detection data ──
drift_path = "experiments/results/online_learning_eta_sweep_optimality_test/drift_detection_dicts.json"
with open(drift_path, "r") as f:
    dicts = json.load(f)

# ── Load heatmap data (optimal LR per eta) ──
heatmap_path = "experiments/results/online_learning_eta_sweep_optimality_test/lr_sweep_heatmap_data.json"
with open(heatmap_path, "r") as f:
    heatmap = json.load(f)

# Find optimal LR per eta from heatmap
from collections import OrderedDict
eta_best = OrderedDict()
for eta, lr, lr_type, loss in zip(heatmap["eta_values"], heatmap["lr_values"], heatmap["lr_types"], heatmap["avg_losses"]):
    if eta not in eta_best or loss < eta_best[eta]["loss"]:
        eta_best[eta] = {"lr": lr, "lr_type": lr_type, "loss": loss}

# Average GLRT observables per eta from drift dicts (only static LR runs — use lr_at_detection to filter)
eta_glrt = defaultdict(list)
for d in dicts:
    eta_glrt[d["eta"]].append(d)

# Build the mapping table: avg_observable → optimal_lr (exclude eta=0.0 control)
print(f"{'eta':>6}  {'avg_log_glr':>12}  {'avg_glr_diff':>12}  {'optimal_lr':>12}  {'type':>10}  {'loss':>10}")
print("-" * 72)

map_etas = []
map_log_glrs = []
map_glr_diffs = []
map_opt_lrs = []

for eta in sorted(eta_best.keys()):
    if eta < 0.01:  # skip control
        continue
    if eta not in eta_glrt:
        continue
    group = eta_glrt[eta]
    avg_log_glr = np.mean([d["main_log_glr"] for d in group])
    avg_glr_diff = np.mean([d["main_log_glr"] - d["baseline_mean"] for d in group])
    opt_lr = eta_best[eta]["lr"]

    map_etas.append(eta)
    map_log_glrs.append(avg_log_glr)
    map_glr_diffs.append(avg_glr_diff)
    map_opt_lrs.append(opt_lr)

    print(f"{eta:>6.2f}  {avg_log_glr:>12.4f}  {avg_glr_diff:>12.4f}  {opt_lr:>12.6f}  {eta_best[eta]['lr_type']:>10}  {eta_best[eta]['loss']:>10.6f}")

map_log_glrs = np.array(map_log_glrs)
map_glr_diffs = np.array(map_glr_diffs)
map_opt_lrs = np.array(map_opt_lrs)
log_opt_lrs = np.log10(map_opt_lrs)

# ── Fit sigmoid: log10(LR*) = f(main_log_glr) ──
def sigmoid(x, L_low, L_high, k, x0):
    return L_low + (L_high - L_low) / (1.0 + np.exp(-k * (x - x0)))

# Fit using main_log_glr
p0_glr = [log_opt_lrs.min(), log_opt_lrs.max(), 0.5, 75.0]
bounds_glr = ([-5, -3, 0.001, 30], [-1, 0, 10, 90])
popt_glr, pcov_glr = curve_fit(sigmoid, map_log_glrs, log_opt_lrs, p0=p0_glr, bounds=bounds_glr, maxfev=10000)
perr_glr = np.sqrt(np.diag(pcov_glr))
r2_glr = 1 - np.sum((log_opt_lrs - sigmoid(map_log_glrs, *popt_glr))**2) / np.sum((log_opt_lrs - np.mean(log_opt_lrs))**2)

# Fit using glr_diff
p0_diff = [log_opt_lrs.min(), log_opt_lrs.max(), 0.5, 60.0]
bounds_diff = ([-5, -3, 0.001, 20], [-1, 0, 10, 85])
popt_diff, pcov_diff = curve_fit(sigmoid, map_glr_diffs, log_opt_lrs, p0=p0_diff, bounds=bounds_diff, maxfev=10000)
perr_diff = np.sqrt(np.diag(pcov_diff))
r2_diff = 1 - np.sum((log_opt_lrs - sigmoid(map_glr_diffs, *popt_diff))**2) / np.sum((log_opt_lrs - np.mean(log_opt_lrs))**2)

print(f"\n{'='*60}")
print(f"Sigmoid fit: log10(LR*) = L_min + (L_max-L_min)/(1+exp(-k*(x-x0)))")
print(f"{'='*60}")

print(f"\n--- Using main_log_glr as input ---")
print(f"  L_min = {popt_glr[0]:.4f} ± {perr_glr[0]:.4f}  (LR_min = {10**popt_glr[0]:.6f})")
print(f"  L_max = {popt_glr[1]:.4f} ± {perr_glr[1]:.4f}  (LR_max = {10**popt_glr[1]:.6f})")
print(f"  k     = {popt_glr[2]:.4f} ± {perr_glr[2]:.4f}")
print(f"  x0    = {popt_glr[3]:.4f} ± {perr_glr[3]:.4f}")
print(f"  R²    = {r2_glr:.4f}")

print(f"\n--- Using glr_diff (glr - baseline) as input ---")
print(f"  L_min = {popt_diff[0]:.4f} ± {perr_diff[0]:.4f}  (LR_min = {10**popt_diff[0]:.6f})")
print(f"  L_max = {popt_diff[1]:.4f} ± {perr_diff[1]:.4f}  (LR_max = {10**popt_diff[1]:.6f})")
print(f"  k     = {popt_diff[2]:.4f} ± {perr_diff[2]:.4f}")
print(f"  x0    = {popt_diff[3]:.4f} ± {perr_diff[3]:.4f}")
print(f"  R²    = {r2_diff:.4f}")

# ── Plot both fits ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: main_log_glr → LR*
x_dense = np.linspace(map_log_glrs.min() * 0.9, map_log_glrs.max() * 1.05, 300)
ax1.plot(x_dense, 10**sigmoid(x_dense, *popt_glr), color="#888888", linewidth=1.8, zorder=3,
         label=r"$\mathrm{LR}^*(G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;\frac{\Delta\log}{1+e^{-k(G-G_0)}}}$"
               f"\n$R^2 = {r2_glr:.3f}$")
for eta, glr, lr in zip(map_etas, map_log_glrs, map_opt_lrs):
    ax1.scatter(glr, lr, s=100, c="#3A76AF", edgecolors="black", linewidths=0.7, zorder=6)
    ax1.annotate(f"$\\eta$={eta}", (glr, lr), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCC", alpha=0.85, lw=0.5))
ax1.set_xlabel("Main Log-GLR at Detection ($G$)")
ax1.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
ax1.set_title("LR* vs Log-GLR", fontweight="bold")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.25, linewidth=0.6)
ax1.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#CCC")

# Plot 2: glr_diff → LR*
x_dense2 = np.linspace(map_glr_diffs.min() * 0.9, map_glr_diffs.max() * 1.05, 300)
ax2.plot(x_dense2, 10**sigmoid(x_dense2, *popt_diff), color="#888888", linewidth=1.8, zorder=3,
         label=r"$\mathrm{LR}^*(\Delta G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;\frac{\Delta\log}{1+e^{-k(\Delta G-\Delta G_0)}}}$"
               f"\n$R^2 = {r2_diff:.3f}$")
for eta, diff, lr in zip(map_etas, map_glr_diffs, map_opt_lrs):
    ax2.scatter(diff, lr, s=100, c="#E07B39", marker="D", edgecolors="black", linewidths=0.7, zorder=6)
    ax2.annotate(f"$\\eta$={eta}", (diff, lr), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCC", alpha=0.85, lw=0.5))
ax2.set_xlabel(r"GLR Diff at Detection ($\Delta G = G - \bar{G}_{\mathrm{base}}$)")
ax2.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
ax2.set_title(r"LR* vs GLR Diff", fontweight="bold")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.25, linewidth=0.6)
ax2.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#CCC")

plt.tight_layout()
out_path = "experiments/results/online_learning_eta_sweep_optimality_test/glrt_observable_to_optimal_lr.png"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved plot to {out_path}")
