import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from litedepth.model import build_model
from litedepth.losses import DepthLoss
from litedepth.datasets import ImageDepthDataset
from litedepth.metrics import compute_depth_metrics


def poly_lr(base_lr, step, total, power=0.9):
    return base_lr * (1 - step / total) ** power


def evaluate(model, loader, device, dataset_type="nyu", max_depth=10.0):
    model.eval()
    agg = {}
    n = 0
    with torch.no_grad():
        for img, dep in loader:
            img, dep = img.to(device), dep.to(device)
            pred = model(img) * max_depth
            agg_now = compute_depth_metrics(pred, dep, dataset_type, max_depth=max_depth)
            for k, v in agg_now.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--val-root", default=None)
    ap.add_argument("--dataset", default="nyu", choices=["nyu", "kitti"])
    ap.add_argument("--size", nargs=2, type=int, default=[384, 384])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--encoder", default="tinynet")
    ap.add_argument("--recurrent", type=int, default=2)
    ap.add_argument("--disable-gated-skips", action="store_true")
    ap.add_argument("--disable-recurrence", action="store_true")
    ap.add_argument("--disable-detail-head", action="store_true")
    ap.add_argument("--ssi-weight", type=float, default=0.0, help="enable for cross-dataset/zero-shot style training")
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--out-dir", default="runs/litedepth_gd")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = {
        "encoder": args.encoder,
        "recurrent": args.recurrent,
        "disable_gated_skips": args.disable_gated_skips,
        "disable_recurrence": args.disable_recurrence,
        "disable_detail_head": args.disable_detail_head,
    }
    model = build_model(model_cfg).to(device)
    criterion = DepthLoss(ssi_weight=args.ssi_weight).to(device)

    train_ds = ImageDepthDataset(
        args.data_root, "train", size=tuple(args.size),
        depth_scale=args.depth_scale, max_depth=args.max_depth, augment=True,
    )
    n_train = int(0.995 * len(train_ds))
    train_part = torch.utils.data.Subset(train_ds, range(n_train))
    val_part = torch.utils.data.Subset(train_ds, range(n_train, len(train_ds)))
    if args.val_root:
        val_part = ImageDepthDataset(
            args.val_root, "val", size=tuple(args.size),
            depth_scale=args.depth_scale, max_depth=args.max_depth, augment=False,
        )
    train_loader = DataLoader(
        train_part, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_part, batch_size=1, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | device: {device}")
    print(f"train {len(train_part)} / val {len(val_part)} | epochs {args.epochs}")

    step = 0
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, count = 0.0, 0
        for img, dep in train_loader:
            lr = poly_lr(args.lr, step, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            img, dep = img.to(device, non_blocking=True), dep.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(img) * args.max_depth
                loss, _ = criterion(pred, dep, img)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses += loss.item()
            count += 1
            step += 1
        metrics = evaluate(model, val_loader, device, args.dataset, args.max_depth)
        line = f"epoch {epoch+1}/{args.epochs} loss {losses/max(count,1):.4f} lr {lr:.2e} ({time.time()-t0:.0f}s)"
        if metrics:
            line += f" | d1 {metrics['d1']:.3f} abs_rel {metrics['abs_rel']:.3f} rmse {metrics['rmse']:.3f}"
            best_key = "rmse"
            if metrics[best_key] < best:
                best = metrics[best_key]
                torch.save({"model": model.state_dict(), "cfg": model_cfg, "epoch": epoch}, out_dir / "best.pth")
        print(line)
        torch.save({"model": model.state_dict(), "cfg": model_cfg, "epoch": epoch}, out_dir / "last.pth")
    if (out_dir / "best.pth").exists():
        print(f"best val rmse: {best:.4f} -> {out_dir/'best.pth'}")


if __name__ == "__main__":
    main()
