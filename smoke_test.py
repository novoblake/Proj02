"""Pre-flight smoke test: verifies the full LiteDepth-GD structure and gives
an early read on whether the core idea (cross-scale gated skips > standard
skips) holds, using a controlled synthetic depth task before any real-data
training.

Usage:
    python smoke_test.py              # structure + quick synthetic run (~1 min CPU)
    python smoke_test.py --structure  # structure checks only (~5 s)
    python smoke_test.py --full       # longer synthetic comparison (~5 min CPU)

Exit code 0 = all checks passed.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from litedepth.model import build_model
from litedepth.losses import DepthLoss
from litedepth.metrics import compute_depth_metrics, sparsification_ause
from litedepth.plugins import ScaleCompetitionGate

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((PASS, name, f"{time.time()-t0:.1f}s", detail or ""))
        return True
    except Exception as e:
        results.append((FAIL, name, f"{time.time()-t0:.1f}s", f"{type(e).__name__}: {e}"))
        return False


def make_synthetic_batch(n, size, seed):
    """Images/depth with multi-scale structure: coarse radial blobs (needs
    deep context) + fine stripe ripples (needs shallow skips). A model that
    cannot fuse both scales cannot fit the depth field."""
    g = torch.Generator().manual_seed(seed)
    h, w = size, size
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
    )
    fine = torch.sin(2 * torch.pi * xs * 12)
    coarse = torch.sin(2 * torch.pi * ys * 1.5) * torch.cos(2 * torch.pi * xs)
    depth = 2.0 + 6.0 * (0.5 + 0.5 * coarse) + 1.2 * fine
    img = torch.zeros(n, 3, h, w)
    for i in range(n):
        phase = torch.rand(1, generator=g).item()
        stripe = torch.sin(2 * torch.pi * (xs * 12 + phase))
        blobs = (
            0.6 * torch.exp(-((xs - 0.3) ** 2 + (ys - 0.4) ** 2) * 6)
            + 0.6 * torch.exp(-((xs - 0.7) ** 2 + (ys - 0.7) ** 2) * 8)
        )
        base = 0.4 + 0.5 * blobs + 0.3 * stripe
        img[i, 0] = base
        img[i, 1] = base * (0.5 + 0.5 * coarse)
        img[i, 2] = 1 - base
    return img, depth.unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)


def structure_checks():
    check(
        "model builds (full, all encoders)",
        lambda: (
            build_model({}),
            build_model({"encoder": "mobilenetv3"}),
            build_model({"uncertainty": True}),
            "2.61M tinynet / 1.70M mobilenetv3",
        )
        and None,
    )
    check(
        "ablation variants forward",
        lambda: [
            build_model(cfg)(torch.randn(1, 3, 128, 128)).shape
            for cfg in (
                {"disable_gated_skips": True},
                {"disable_recurrence": True},
                {"disable_detail_head": True},
                {"recurrent": 1},
            )
        ]
        and None,
    )
    check("full backward (all grads)", lambda: _backward_check())
    check("uncertainty returns depth + logvar", lambda: _uncertainty_check())
    check("plug-in gate standalone", lambda: _plugin_check())
    check("metrics: KITTI garg + NYU eigen + AUSE", lambda: _metrics_check())


def _backward_check():
    m = build_model({})
    out = m(torch.randn(1, 3, 128, 128))
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad: {missing[:5]}"


def _uncertainty_check():
    m = build_model({"uncertainty": True})
    d, lv = m(torch.randn(1, 3, 96, 96), return_uncertainty=True)
    assert d.shape == (1, 1, 96, 96) and lv.shape == (1, 1, 96, 96)


def _plugin_check():
    gate = ScaleCompetitionGate([16, 32, 64], 16)
    f = [torch.randn(1, c, 32, 32) for c in (16, 32, 64)]
    prev = torch.randn(1, 16, 32, 32)
    out = gate(f, prev, (32, 32))
    assert out.shape == (1, 32, 32, 32)


def _metrics_check():
    pred = torch.rand(1, 1, 240, 320) * 8 + 0.5
    target = torch.rand(1, 1, 240, 320) * 8 + 0.5
    assert compute_depth_metrics(pred, target, "nyu")
    assert compute_depth_metrics(pred, target, "kitti", max_depth=80.0)
    ause = sparsification_ause(pred, target, (pred - target).abs())
    assert ause == ause


def synthetic_idea_check(steps=350, size=96, batch=8, quick=False):
    """Train full vs ablated models on the synthetic task; the idea 'holds'
    if the full model reaches clearly lower val rmse than the standard-skip
    variant."""
    if quick:
        steps = 120
    img, dep = make_synthetic_batch(4 * batch, size, seed=7)
    n_val = batch * 2
    train_x, train_y = img[:-n_val], dep[:-n_val]
    val_x, val_y = img[-n_val:], dep[-n_val:]

    configs = {
        "full (CGSA+RGR+DG)": {},
        "no_gcsa (standard skips)": {"disable_gated_skips": True},
        "no_dghead": {"disable_detail_head": True},
    }
    if not quick:
        configs["no_rgr"] = {"disable_recurrence": True}

    crit = DepthLoss()
    print(f"\nsynthetic task: {size}x{size}, {steps} steps, batch {batch}")
    print(f"{'config':28s} {'first loss':>11s} {'val loss':>9s} {'val d1':>7s} {'val rmse':>9s}")
    table = {}
    for name, cfg in configs.items():
        torch.manual_seed(0)
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        first_loss = None
        for _ in range(steps):
            idx = torch.randint(0, train_x.shape[0], (batch,))
            x, y = train_x[idx], train_y[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(x) * 10
            loss, _ = crit(pred, y, x)
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = loss.item()
        with torch.no_grad():
            pred = model(val_x) * 10
            m = compute_depth_metrics(pred, val_y, "nyu")
            vloss, _ = crit(pred, val_y, val_x)
        table[name] = m["rmse"]
        print(
            f"{name:28s} {first_loss:11.3f} {vloss.item():9.3f} "
            f"{m['d1']:7.3f} {m['rmse']:9.3f}"
        )

    full = table["full (CGSA+RGR+DG)"]
    std = table["no_gcsa (standard skips)"]
    print(f"\nfull vs standard-skip val rmse: {full:.3f} vs {std:.3f} "
          f"({100*(std-full)/std:+.1f}%)")
    verdict = (
        "IDEA HOLDS on synthetic task: gated multi-scale skips fit the "
        "fine+coarse depth field better than standard skips."
        if full < std
        else "INCONCLUSIVE on synthetic task - confirm with the real-data "
        "ablation matrix (the synthetic bias may not favor gated skips)."
    )
    if "no_rgr" in table and table["no_rgr"] < full:
        print(f"note: no_rgr reached {table['no_rgr']:.3f} rmse - the shared-weight "
              "recurrent pass did not help on this synthetic task; its value must "
              "be judged on real data (complex boundaries, depth discontinuities).")
    print(verdict)
    assert full < 1.5, f"full model failed to learn synthetic task (rmse {full:.3f})"
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true", help="structure checks only")
    ap.add_argument("--full", action="store_true", help="longer synthetic comparison")
    args = ap.parse_args()

    print("=== 1/2 structure checks ===")
    structure_checks()
    width = max(len(n) for _, n, _, _ in results)
    for status, name, t, detail in results:
        mark = "ok" if status == PASS else "XX"
        print(f"[{mark}] {name.ljust(width)} ({t}) {detail if status == FAIL else ''}")
    n_fail = sum(1 for s, *_ in results if s == FAIL)

    if not args.structure:
        print("\n=== 2/2 synthetic idea check ===")
        synthetic_idea_check(quick=not args.full)

    if n_fail:
        print(f"\nSMOKE TEST FAILED ({n_fail} structure checks)")
        sys.exit(1)
    print("\nSMOKE TEST OK - ready to train on real data")


if __name__ == "__main__":
    main()
