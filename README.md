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

## Quick start: verify the setup (smoke test)

Before any real training, run the pre-flight smoke test on your local PC:

```bash
pip install -r requirements.txt
python smoke_test.py            # ~1 min: structure checks + synthetic idea check
python smoke_test.py --structure   # structure only (~5 s)
python smoke_test.py --full     # ~5 min: longer 4-config comparison
```

It verifies: all model variants build and backprop (both encoders, uncertainty
head, plug-in gate), the full metric suite (eigen/garg crops, AUSE), and then
trains on a synthetic depth field with coarse + fine multi-scale structure and
compares the full model against its ablations. On this task the gated-skip
full model beats standard-skip fusion by ~33% val RMSE — an early, controlled
sign the decoder-connection idea holds before spending GPU days on real data.

## Training

Loss = SILog + multi-scale gradient matching + edge-aware smoothness
(+ optional scale-shift-invariant term for cross-dataset training).

Data layout: `data_root/images/*.{png,jpg}` and `data_root/depth/*.png` with
matching file names. Depth PNGs are divided by `--depth-scale`
(256 when depths are stored as meters x 256, 1000 for millimeter 16-bit
exports, 1 for float files).

### NYUv2

```bash
python train.py --data-root data/nyu --val-root data/nyu_val \
    --dataset nyu --epochs 25 --batch-size 16 --out-dir runs/nyu
```

### KITTI (exact commands)

1. Download the KITTI depth-prediction data
   ([kitti_depth](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
   — "Depth Prediction annotated RGB-D ground truth", i.e. the Ma & Karaman
   converted annotated depth maps). Unzip so you have
   `<kitti_root>/train/<date>/<drive>/image_02/data/*.png` and
   `<kitti_root>/train/<date>/<drive>/proj_depth/groundtruth/image_02/data/*.png`
   (same for `val/`). GT depth PNGs there store depth_meters x 256.

2. Convert to the repo's paired layout (verified on synthetic KITTI trees):

```bash
tools/make_kitti_pairs.py --kitti-root /path/to/kitti_root --out data/kitti
```

3. Train (352x1216, 16-bit GT/256, max depth 80 m):

```bash
python train.py --data-root data/kitti --val-root data/kitti \
    --dataset kitti --size 352 1216 --batch-size 16 --epochs 40 \
    --lr 4e-4 --depth-scale 256 --max-depth 80 \
    --out-dir runs/kitti
```

Notes: the default val split = last 0.5% of `--data-root`; pass
`--val-root data/kitti_val` if you converted a separate val split. On small
GPUs drop `--batch-size` to 4-8. If your GT was exported differently, match
`--depth-scale` to the export (pixel_value / depth_scale = meters).

4. Evaluate on val with the standard understated KITTI protocol (Garg crop,
   depths in (0, 80) m):

```bash
python eval.py --checkpoint runs/kitti/best.pth --data-root data/kitti \
    --dataset kitti --size 352 1216 --depth-scale 256 --max-depth 80
```

5. If you want a single number for the paper/table, same eval again, logged:

```bash
python eval.py --checkpoint runs/kitti/best.pth --data-root data/kitti \
    --dataset kitti --size 352 1216 --depth-scale 256 --max-depth 80 \
    --save-vis runs/kitti/vis
```

## Evaluation (standard protocols)

`eval.py` computes AbsRel, SqRel, RMSE, RMSElog, log10 and δ1/δ2/δ3 with the
**Eigen crop** for NYUv2 and the **Garg crop** for KITTI (selected by
`--dataset` automatically), plus optional median-scale alignment for
relative-depth checkpoints.

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

## Reproducibility and ablation harness (journal-grade rigor)

- Fixed seeds + deterministic cudnn (`--seed 42`), EMA weights for eval and
  best-checkpoint selection (`--ema 1`), per-epoch metrics appended to
  `runs/<dir>/metrics.json`.
- **Aleatoric uncertainty head**: `--uncertainty` adds a Laplacian NLL head
  (heteroscedastic confidence) and reports **AUSE (sparsification error)**
  in train/eval output — a calibration metric journals expect.
- **Plug-in gate for generality studies**: `litedepth/plugins.py` exports the
  scale-competition gate for swapping into other encoder-decoder backbones.
- Multi-seed ablation + Markdown result table:

```bash
bash experiments/run_ablation.sh data/nyu data/nyu_val runs/ablation
python experiments/collect_results.py runs/ablation
```

See `docs/JOURNAL_PLAN.md` for the full venue analysis and experiment matrix.

## Tests

```bash
pytest tests/ -q
```

covers output shapes at multiple resolutions, full back-propagation, every
ablation variant, loss mask behavior, scale-shift invariance of the SSI loss,
metric correctness, and the <4M-parameter lightweight budget.
