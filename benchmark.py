import argparse
import json
import time
from pathlib import Path

import torch

from litedepth.model import build_model


@torch.no_grad()
def latency_ms(model, shape, device, iters=50, warmup=10):
    x = torch.randn(1, *shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def count_macs(model, shape, device):
    macs = [0]
    hooks = []

    def hook(module, inp, out):
        if isinstance(module, torch.nn.Conv2d):
            kh, kw = module.kernel_size
            macs[0] += out.numel() * (module.in_channels // module.groups) * kh * kw

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    model(torch.randn(1, *shape, device=device))
    for h in hooks:
        h.remove()
    return macs[0] / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=2, type=int, default=[384, 480])
    ap.add_argument("--encoder", default="tinynet", choices=["tinynet", "mobilenetv3"])
    ap.add_argument("--recurrent", type=int, default=2)
    ap.add_argument("--disable-gated-skips", action="store_true")
    ap.add_argument("--disable-recurrence", action="store_true")
    ap.add_argument("--disable-detail-head", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--benchmark", action="store_true", help="run latency/MACs benchmark")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        {
            "encoder": args.encoder,
            "recurrent": args.recurrent,
            "disable_gated_skips": args.disable_gated_skips,
            "disable_recurrence": args.disable_recurrence,
            "disable_detail_head": args.disable_detail_head,
        }
    ).to(device).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(sd["model" if "model" in sd else ""])

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shape = (3, args.size[0], args.size[1])

    x = torch.randn(1, *shape, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"encoder={args.encoder} recurrent={args.recurrent}")
    print(f"input: {shape} -> output: {tuple(out.shape)} (sigmoid, meters)")
    print(f"params: {n_params/1e6:.2f}M (trainable {n_train/1e6:.2f}M)")

    if args.benchmark:
        macs = count_macs(model, shape, device)
        lat = latency_ms(model, shape, device)
        fps = 1000 / lat
        print(f"MACs @ {args.size[0]}x{args.size[1]}: {macs/1000:.2f}G (≈{2*macs/1000:.2f}GFLOPs)")
        print(f"latency: {lat:.1f} ms/frame on {device.type} | throughput: {fps:.1f} FPS")

    x = torch.randn(1, *shape, device=device)
    out = model(x)
    loss = torch.nn.functional.l1_loss(out * 10, torch.rand_like(out) * 10 + 1)
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(grads), "some parameters received no gradient"
    print(f"backward OK ({sum(grads)} tensors with grads)")


if __name__ == "__main__":
    main()
