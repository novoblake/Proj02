"""Aggregate multi-seed ablation runs: OUT_ROOT/<config>/s<seed>/metrics.json.

Prints mean ± std of final and best metrics per config as Markdown, and the
relative delta of every ablated variant against `full` (higher-is-better for
d1/d2/d3, lower-is-better for the rest).
"""
import json
import sys
from pathlib import Path

import numpy as np

HIGHER_BETTER = {"d1", "d2", "d3"}
FRACTIONS = 0.05  # use last 5% epochs for the "final" reading if desired


def read_metrics(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows


def best_epoch(rows):
    if not rows:
        return {}
    best = min(rows, key=lambda r: r.get("rmse", float("inf")))
    return best


def main(root):
    root = Path(root)
    configs = {}
    for cfg_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for seed_dir in sorted(cfg_dir.glob("s*")):
            f = seed_dir / "metrics.json"
            if not f.exists():
                continue
            rows = read_metrics(f)
            best = best_epoch(rows)
            if best:
                configs.setdefault(cfg_dir.name, []).append(best)

    if not configs:
        sys.exit(f"no metrics.json found under {root}")

    keys = sorted({k for runs in configs.values() for r in runs for k in r})
    metrics = [k for k in keys if k not in ("epoch",)]
    print("| config | seeds | " + " | ".join(metrics) + " |")
    print("|" + "---|" * (len(metrics) + 2))
    base = configs.get("full") or next(iter(configs.values()))
    for name, runs in configs.items():
        cells = []
        for k in metrics:
            vals = np.array([r[k] for r in runs if k in r], dtype=float)
            if vals.size == 0:
                cells.append("—")
            else:
                cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
        print(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")

    print("\nDelta vs full (negative = worse):")
    ref = {k: float(np.mean([r[k] for r in base if k in r])) for k in metrics if k in base[0]}
    for name, runs in configs.items():
        if name == "full" or not runs:
            continue
        deltas = []
        for k in metrics:
            if k not in ref or k not in runs[0]:
                continue
            vals = np.array([r[k] for r in runs])
            d = vals.mean() - ref[k]
            if k in HIGHER_BETTER:
                d = -d
            deltas.append(f"{k}: {d:+.3f}")
        print(f"- {name}: {'  '.join(deltas)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/ablation")
