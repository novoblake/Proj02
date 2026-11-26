import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from litedepth.model import build_model
from litedepth.datasets import ImageDepthDataset
from litedepth.metrics import compute_depth_metrics, median_scale_align, sparsification_ause


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="nyu", choices=["nyu", "kitti"])
    ap.add_argument("--size", nargs=2, type=int, default=None)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--median-align", action="store_true")
    ap.add_argument("--save-vis", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt.get("cfg", {})).to(device).eval()

    ds = ImageDepthDataset(
        args.data_root, "val",
        size=tuple(args.size) if args.size else None,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=2)

    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    use_unc = bool(ckpt.get("cfg", {}).get("uncertainty", False))
    agg, n = {}, 0
    with torch.no_grad():
        for idx, (img, dep) in enumerate(loader):
            img, dep = img.to(device), dep.to(device)
            if use_unc:
                pred, logvar = model(img, return_uncertainty=True)
                pred = pred * args.max_depth
            else:
                pred = model(img) * args.max_depth
            if args.median_align:
                pred = median_scale_align(pred, dep, max_depth=args.max_depth)
            m = compute_depth_metrics(pred, dep, args.dataset, max_depth=args.max_depth)
            if use_unc:
                ause = sparsification_ause(pred, dep, logvar, max_depth=args.max_depth)
                if ause == ause:
                    m["ause"] = ause
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            if args.save_vis and idx < 20:
                p = pred[0, 0].cpu().numpy()
                p = (p / max(p.max(), 1e-3) * 255).astype(np.uint8)
                Image.fromarray(p).save(Path(args.save_vis) / f"{idx:03d}_pred.png")
                d = dep[0, 0].cpu().numpy()
                d = (d / max(d.max(), 1e-3) * 255).astype(np.uint8)
                Image.fromarray(d).save(Path(args.save_vis) / f"{idx:03d}_gt.png")

    crop_name = "garg" if args.dataset == "kitti" else "eigen"
    print(f"{args.dataset.upper()} eval ({n} samples) | crop={crop_name}{' + median-align' if args.median_align else ''}")
    header = "abs_rel  sq_rel   rmse    rmse_log log10   d1      d2      d3"
    if use_unc:
        header += "  ause"
    print(header)
    if agg:
        line = (
            f"{agg['abs_rel']/n:.4f}   {agg['sq_rel']/n:.4f}   "
            f"{agg['rmse']/n:.4f}  {agg['rmse_log']/n:.4f}   "
            f"{agg['log10']/n:.4f} {agg['d1']/n:.4f} {agg['d2']/n:.4f} {agg['d3']/n:.4f}"
        )
        if use_unc and "ause" in agg:
            line += f" {agg['ause']/n:.4f}"
        print(line)


if __name__ == "__main__":
    main()
