"""Collect a run's metrics into a loss-curve figure and a markdown summary table.

Runs entirely on CPU; needs no GPU and no dataset.  This is the script that turns a run
directory into report-ready artefacts, so that results survive even if the training host is
lost (a lesson learned the hard way: the first Exp B run lived only on the remote disk).

    python scripts/collect_results.py --run-dir checkpoints/expB_in100_vits16_128 \
        --out-dir results
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch


def load_metrics(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows: List[Dict], key: str):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:  # skip NaN
            xs.append(r["step"])
            ys.append(float(v))
    return xs, ys


def smooth(ys: List[float], k: int = 25) -> List[float]:
    if k <= 1 or len(ys) < k:
        return ys
    out = []
    acc = 0.0
    for i, y in enumerate(ys):
        acc += y
        if i >= k:
            acc -= ys[i - k]
        out.append(acc / min(i + 1, k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.run_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_metrics(os.path.join(args.run_dir, "metrics.jsonl"))
    print(f"{name}: {len(rows)} log records, last step {rows[-1]['step']}")

    # ---------------- markdown summary
    keys = [
        ("train/pred", "prediction loss"),
        ("train/sigreg", "SIGReg"),
        ("train/lejepa", "LeJEPA total"),
        ("train/probe", "online probe loss"),
        ("eval/online_probe_acc", "online probe acc (val)"),
        ("diag/effective_rank", "effective rank"),
        ("diag/feature_std_min", "min feature std"),
        ("diag/mean_norm", "||mean embedding||"),
        ("diag/cov_offdiag", "mean |off-diag corr|"),
    ]
    md = [f"# Run summary — `{name}`", "", f"- log records: {len(rows)}",
          f"- last step: {rows[-1]['step']}", ""]
    md.append("| metric | first | last | min | max |")
    md.append("|---|---|---|---|---|")
    for key, label in keys:
        _, ys = series(rows, key)
        if not ys:
            continue
        md.append(
            f"| {label} (`{key}`) | {ys[0]:.4f} | {ys[-1]:.4f} | "
            f"{min(ys):.4f} | {max(ys):.4f} |"
        )
        print(f"  {label:28s} first={ys[0]:.4f} last={ys[-1]:.4f} "
              f"min={min(ys):.4f} max={max(ys):.4f}")

    # eval table
    eval_rows = [r for r in rows if "eval/online_probe_acc" in r]
    if eval_rows:
        md += ["", "## Online probe accuracy (validation split)", "",
               "| step | accuracy | n |", "|---|---|---|"]
        for r in eval_rows:
            md.append(f"| {r['step']} | {r['eval/online_probe_acc']*100:.2f}% | {r.get('eval/n','')} |")
        md.append("")
        accs = [r["eval/online_probe_acc"] * 100 for r in eval_rows]
        print(f"  final online probe acc = {accs[-1]:.2f}%  (first {accs[0]:.2f}%)")

    out_md = os.path.join(args.out_dir, f"{name}_summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {out_md}")

    # ---------------- figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    for key, label in [("train/pred", "prediction"), ("train/sigreg", "SIGReg")]:
        xs, ys = series(rows, key)
        if ys:
            ax.plot(xs, smooth(ys), label=label, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title("LeJEPA loss terms")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    xs, ys = series(rows, "train/lejepa")
    if ys:
        ax.plot(xs, smooth(ys), color="C3", lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("total loss (log)")
    ax.set_title("LeJEPA objective  $\\lambda$SIGReg + (1-$\\lambda$)pred")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if eval_rows:
        ax.plot([r["step"] for r in eval_rows],
                [r["eval/online_probe_acc"] * 100 for r in eval_rows],
                "o-", color="C2", lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("top-1 (%)")
    ax.set_title("online linear probe (validation)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for key, label in [("diag/effective_rank", "effective rank"),
                       ("diag/feature_std_min", "min feature std")]:
        xs, ys = series(rows, key)
        if ys:
            ax.plot(xs, ys, label=label, lw=1.4)
    ax.set_xlabel("step")
    ax.set_title("collapse diagnostics")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"{name}_curves.png")
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
