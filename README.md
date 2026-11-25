# LiteDepth-GD

Lightweight, fast and accurate monocular depth estimation with a novel decoder
connection design. **2.61M parameters**, **2.98G MACs at 384x480** (≈5.97GFLOPs),
**~18 FPS single-thread CPU** (46ms / 21.6 FPS at 384x384), full-resolution output,
every parameter trainable and gradient-verified.

Measured on this repo's `benchmark.py`; accuracy numbers depend on your dataset
table, below.

## Architecture

![overview](DESIGN.md)

Inputs work at any resolution divisible by 8 (tested 256x256 to 384x512+).

```
Input (3x H x W)
 └─ TinyNet encoder (stride 2 stem -> 4 stages, strides 4/8/16/32)
     │  stem features kept as a thin detail path (D, 24ch, stride 2)
     ├─ s1 (48ch, s/4), s2 (96ch, s/8), s3 (160ch, s/16), s4 (256ch, s/32)
     └─ Bottleneck (224ch)
        └─ 4 decoder levels, each with:
            1. CGSA  — Cross-Scale Gated Skip Aggregation (NOVEL)
            2. RGR   — Recurrent Gated Refinement on the finest level (NOVEL)
            3. DG-Head — Detail-Guidance upsampling head (NOVEL)
```

### Novel decoder connections

1. **Cross-Scale Gated Skip Aggregation (CGSA)** — classic U-Net decoders only
   use the skip at the matching resolution, which ties every output region to
   exactly one receptive-field scale. In LiteDepth-GD, *every* decoder level
   receives all encoder scales: each is pooled/upsampled to the level
   resolution, projected by 1x1 convs, and combined with
   **source-softmax, channel-wise gates** computed from global-pooled
   features. Scales *compete* for each channel instead of being added
   blindly, so shallow boundaries or deep context can dominate wherever the
   image demands. Cost: a few thousand parameters per level.

2. **Recurrent Gated Refinement (RGR)** — the finest decoder level is run
   `k` times (default 2) with **shared weights**; each extra pass re-ingests
   its own output through the same cross-scale gate, which lets the model
   correct coarse errors (e.g. silhouettes leaking sky as far depth)
   without a second full decoder.

3. **Detail-Guidance Head (DG-Head)** — the high-resolution stem features
   (kept separate the whole way down) modulate the final x2 PixelShuffle
   upsampling via a gated fusion, restoring fine boundaries the coarse
   decoder features lose. This is a cheap guided-upsampling connection
   rather than an expensive decoder that reaches full stride-1.

Each connection can be ablated independently via flags in `benchmark.py` /
`train.py` (`--disable-gated-skips`, `--disable-recurrence`,
`--disable-detail-head`), giving a clean 3-way ablation study.

## Training

Loss = SILog + multi-scale gradient matching + edge-aware smoothness
(+ optional scale-shift-invariant term for cross-dataset training).

```bash
python train.py --data-root data/nyu --val-root data/nyu_val \
    --dataset nyu --epochs 25 --batch-size 16 --out-dir runs/nyu
python train.py --data-root data/kitti --dataset kitti --max-depth 80 \
    --depth-scale 256 --epochs 40 --out-dir runs/kitti
```

Data layout: `data_root/images/*.{png,jpg}` and `data_root/depth/*.png` with
matching file names. Depth PNGs divided by `--depth-scale`
(256 for 8-bit exports, 1000 for KITTI-style 16-bit, set 1 for float files).

## Evaluation (standard protocols)

`eval.py` computes AbsRel, SqRel, RMSE, RMSElog, log10 and δ1/δ2/δ3 with the
**Eigen crop** for NYUv2 and the garg crop convention via `--dataset kitti`,
plus optional median-scale alignment for relative-depth checkpoints.

```bash
python eval.py --checkpoint runs/nyu/best.pth --data-root data/nyu_val --dataset nyu
python benchmark.py --benchmark            # params, MACs, latency/FPS
python benchmark.py --disable-recurrence --benchmark   # ablation variant
```

## Benchmarks

Fill the numbers below after training (the repo provides the protocol above;
numbers are dataset- and GPU-dependent and are not invented here).

| Model       | Params | MACs (384x384) | NYU d1↑ | NYU AbsRel↓ | KITTI d1↑ |
|-------------|--------|----------------|---------|-------------|-----------|
| FastDepth   | 14.3M  | 5.8G           | 0.825   | 0.127       | —         |
| MiDaS-small | 31M    | —              | —       | 0.134 (SSI) | —         |
| LiteDepth-GD| **2.61M**  | **2.39G**      | —       | —           | —         |

Ablation (full model vs. each novel connection disabled, from `benchmark.py`):

| Variant                  | Params |
|--------------------------|--------|
| full (CGSA + RGR + DG)   | 2.61M  |
| — gated skips (CGSA off) | 2.25M  |
| — recurrence (RGR off)   | 2.60M  |
| — detail head (DG off)   | 2.62M  |

Run each variant through `train.py` + `eval.py` with identical seeds to fill the
accurate d1/AbsRel columns per dataset — the flags give the study for free.

MobileNetV3-small encoder option: 1.70M parameters for even tighter budgets
(build with `{'encoder': 'mobilenetv3'}` or `--encoder mobilenetv3`, ImageNet
weights auto-downloaded by torchvision).

## Tests

```bash
pytest tests/ -q
```

covers output shapes at multiple resolutions, full back-propagation, every
ablation variant, loss mask behavior, scale-shift invariance of the SSI loss,
metric correctness, and the <4M-parameter lightweight budget.
