import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit

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
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})

json_path = "experiments/results/online_learning_eta_sweep/lr_sweep_heatmap_data.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Group by eta, find optimal LR (lowest loss) per eta
eta_entries = defaultdict(list)
for eta, lr, lr_type, loss in zip(
    data["eta_values"], data["lr_values"], data["lr_types"], data["avg_losses"]
):
    eta_entries[eta].append({"lr": lr, "lr_type": lr_type, "loss": loss})

etas = np.array(sorted(eta_entries.keys()))
best_lrs = []
best_losses = []
best_types = []

for eta in etas:
    best = min(eta_entries[eta], key=lambda e: e["loss"])
    best_lrs.append(best["lr"])
    best_losses.append(best["loss"])
    best_types.append(best["lr_type"])

best_lrs = np.array(best_lrs)
best_losses = np.array(best_losses)

# ── Fit: sigmoid in log-space ──
# log(LR*(eta)) = L_low + (L_high - L_low) / (1 + exp(-k*(eta - eta0)))
log_lrs = np.log10(best_lrs)

def sigmoid_log(eta, L_low, L_high, k, eta0):
    return L_low + (L_high - L_low) / (1.0 + np.exp(-k * (eta - eta0)))

p0 = [np.log10(0.001), np.log10(0.035), 15.0, 0.75]
bounds = ([-5, -3, 0.1, 0.3], [-1, 0, 100, 1.5])
popt, pcov = curve_fit(sigmoid_log, etas, log_lrs, p0=p0, bounds=bounds, maxfev=10000)
L_low, L_high, k_fit, eta0_fit = popt
perr = np.sqrt(np.diag(pcov))

eta_dense = np.linspace(etas.min() * 0.8, etas.max() * 1.05, 300)
lr_fit = 10 ** sigmoid_log(eta_dense, *popt)

r_squared = 1 - np.sum((log_lrs - sigmoid_log(etas, *popt))**2) / np.sum((log_lrs - np.mean(log_lrs))**2)

# ── Plot ──
fig, ax = plt.subplots(figsize=(7, 5))

# Fit curve
ax.plot(eta_dense, lr_fit, color="#888888", linewidth=1.8, linestyle="-", zorder=3,
        label=r"$\mathrm{LR}^*(G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;\frac{\log_{10}\mathrm{LR}_{\max}\,-\,\log_{10}\mathrm{LR}_{\min}}{1\,+\,e^{-k(G\,-\,G_0)}}}$")

# Data points colored by type
for i, (eta, lr, loss, lr_type) in enumerate(zip(etas, best_lrs, best_losses, best_types)):
    color = "#E07B39" if lr_type == "adaptive" else "#3A76AF"
    marker = "D" if lr_type == "adaptive" else "o"
    label = "Adaptive" if lr_type == "adaptive" and i == next(j for j, t in enumerate(best_types) if t == "adaptive") else \
            "Static" if lr_type == "static" and i == next(j for j, t in enumerate(best_types) if t == "static") else None
    ax.scatter(eta, lr, c=color, s=100, marker=marker, zorder=6,
               edgecolors="black", linewidths=0.7, label=label)

# Loss annotations
for eta, lr, loss in zip(etas, best_lrs, best_losses):
    ax.annotate(f"{loss:.3f}", (eta, lr),
                textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=10, color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCCCCC", alpha=0.85, lw=0.5))

ax.set_xlabel(r"Calibration Error $\eta$")
ax.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
ax.set_title(r"Optimal LR vs $\eta$", fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.25, linewidth=0.6)
ax.set_xlim(etas.min() - 0.08, etas.max() + 0.08)

ax.legend(loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")

plt.tight_layout()
out_path = "experiments/results/online_learning_eta_sweep/optimal_lr_vs_eta.png"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")

# Summary
print(f"\nSigmoid fit (log10-space):")
print(f"  log10(LR*) = L_low + (L_high - L_low) / (1 + exp(-k*(eta - eta0)))")
print(f"  L_low  = {L_low:.4f} ± {perr[0]:.4f}  (LR = {10**L_low:.6f})")
print(f"  L_high = {L_high:.4f} ± {perr[1]:.4f}  (LR = {10**L_high:.6f})")
print(f"  k      = {k_fit:.4f} ± {perr[2]:.4f}")
print(f"  eta0   = {eta0_fit:.4f} ± {perr[3]:.4f}")
print(f"  R² = {r_squared:.4f}")
print(f"\n{'Eta':>6}  {'Best LR':>12}  {'Type':>10}  {'Loss':>10}")
print("-" * 44)
for eta, lr, loss, lr_type in zip(etas, best_lrs, best_losses, best_types):
    print(f"{eta:>6.2f}  {lr:>12.6f}  {lr_type:>10}  {loss:>10.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Loss vs eta, one curve per LR
# ══════════════════════════════════════════════════════════════════════════════

# Group data by (lr_row_id) to get one curve per LR run
from collections import OrderedDict

lr_row_ids = data["lr_row_ids"]
curves = OrderedDict()

for eta, lr, lr_type, lr_row_id, loss in zip(
    data["eta_values"], data["lr_values"], data["lr_types"],
    lr_row_ids, data["avg_losses"]
):
    if lr_row_id not in curves:
        curves[lr_row_id] = {"lr_type": lr_type, "lr_value": lr, "lr_values": [], "etas": [], "losses": []}
    curves[lr_row_id]["etas"].append(eta)
    curves[lr_row_id]["losses"].append(loss)
    curves[lr_row_id]["lr_values"].append(lr)

# Sort each curve by eta
for c in curves.values():
    order = np.argsort(c["etas"])
    c["etas"] = np.array(c["etas"])[order]
    c["losses"] = np.array(c["losses"])[order]

# Style palette: distinct colors and markers for static, separate style for adaptive
static_styles = [
    {"color": "#2166AC", "marker": "o",  "ls": "-"},
    {"color": "#4393C3", "marker": "s",  "ls": "--"},
    {"color": "#92C5DE", "marker": "^",  "ls": "-."},
    {"color": "#B2ABD2", "marker": "v",  "ls": ":"},
    {"color": "#762A83", "marker": "p",  "ls": "-"},
]
adaptive_style = {"color": "#D6604D", "marker": "D", "ls": "-"}

fig2, ax2 = plt.subplots(figsize=(8, 5.5))

static_idx = 0
for row_id, c in curves.items():
    if c["lr_type"] == "adaptive":
        style = adaptive_style
        lr_min = min(c["lr_values"])
        lr_max = max(c["lr_values"])
        label = f"Adaptive (LR: {lr_min:.4f} - {lr_max:.4f})"
    else:
        style = static_styles[static_idx % len(static_styles)]
        static_idx += 1
        label = f"Static LR = {c['lr_value']}"

    ax2.plot(c["etas"], c["losses"], color=style["color"], marker=style["marker"],
             linestyle=style["ls"], linewidth=2.2, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6,
             label=label, zorder=4)

ax2.set_xlabel(r"Calibration Error $\eta$")
ax2.set_ylabel("Post-Learning RMSPE")
ax2.set_title(r"Post-Learning Loss vs $\eta$ for Each Learning Rate", fontweight="bold")
ax2.grid(True, alpha=0.25, linewidth=0.6)
ax2.set_xlim(etas.min() - 0.05, etas.max() + 0.05)
ax2.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC", fontsize=11)

plt.tight_layout()
out_path2 = "experiments/results/online_learning_eta_sweep/loss_vs_eta_per_lr.png"
fig2.savefig(out_path2, bbox_inches="tight")
plt.close(fig2)
print(f"\nSaved to {out_path2}")
