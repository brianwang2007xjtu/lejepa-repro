"""Build the summary figure used in REPRODUCTION_REPORT.md (CPU only)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
sweep = [json.loads(l) for l in open(f"{R}/ablations/sweep.jsonl")]
expB = json.load(open(f"{R}/expB_in100_vits16_128/eval_last2cls_imagenet100.json"))
expC = json.load(open(f"{R}/expC_linear_probe.json"))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---------------- left: main runs
ax = axes[0]
names = ["Exp B\n4 views @128\n40 ep", "Exp C\n8 views 224+96\n50 ep"]
lp = [expB["linear_top1"], expC["linear_top1"]]
knn = [expB["knn_acc"], expC["knn_acc"]]
x = range(len(names))
b1 = ax.bar([i - 0.18 for i in x], lp, 0.36, label="frozen linear probe", color="C0")
b2 = ax.bar([i + 0.18 for i in x], knn, 0.36, label="k-NN (k=20)", color="C1")
ax.axhline(1.0, ls="--", c="gray", lw=1, label="chance (1%)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}",
                ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("ImageNet-100 top-1 (%)"); ax.set_ylim(0, 80)
ax.set_title("Pretraining runs (ViT-S/16, ImageNet-100)")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# ---------------- right: ablation sweep
ax = axes[1]
order = ["abl_base_lamb0.02", "abl_lamb0.05", "abl_slices64",
         "abl_lamb0.005", "abl_lamb0.2", "abl_views2"]
label = {"abl_base_lamb0.02": "baseline λ=.02",
         "abl_lamb0.05": "λ=.05", "abl_slices64": "M=64",
         "abl_lamb0.005": "λ=.005", "abl_lamb0.2": "λ=.2",
         "abl_views2": "V=2"}
d = {r["name"]: r for r in sweep}
vals = [d[n]["linear_top1"] for n in order]
colors = ["C0", "C0", "C2", "C0", "C3", "C3"]
bars = ax.barh(range(len(order)), vals, color=colors)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([label[n] for n in order])
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 0.6, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", fontsize=10)
ax.set_xlim(0, 37)
ax.set_xlabel("frozen linear probe top-1 (%)")
ax.set_title("Ablations (12 epochs, 4 views @128)  —  ±2–3 pt noise")
ax.axvline(d["abl_base_lamb0.02"]["linear_top1"], ls=":", c="gray", lw=1)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
os.makedirs(R, exist_ok=True)
fig.savefig(f"{R}/summary_figure.png", dpi=140)
print("wrote results/summary_figure.png")
