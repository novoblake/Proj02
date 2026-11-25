# Design notes

## Motivation
Real-time monocular depth models must run on CPU/embedded devices. Efficiency
is dominated by (a) the encoder, (b) how skips are fused, (c) the output
upsampling. LiteDepth-GD targets all three:

- **Encoder**: TinyNet — a 5-stage MBConv hierarchy (24/48/96/160/256 ch,
  depths 1/2/3/3/2, expansion 2 in the two deep stages, strides 2/4/8/16/32)
  that ends at stride 32 and never wastes width where detail is irrelevant,
  ~1.9M params. Optionally swap in MobileNetV3-small (~1.7M total model).
- **Decoder**: 4 levels at strides 32 -> 4 (224/144/96/64 ch), each fusing
  ALL encoder scales via CGSA, ~0.7M params, plus the DG-Head.
- **Upsampling**: bilinear to stride 2, PixelShuffle x2, detail-gated fusion
  to stride 2 output. Only the final step ever runs at high resolution, and
  it is guided by the cheap stem features instead of decoder capacity.

## Why source-softmax gates
Additive skip fusion (concat + convex, or sum) has two failure modes at low
capacity: (1) the network cannot suppress a scale that is misleading for a
given channel (e.g. the coarsest scale inside sky regions), and (2) concat
fusion scales O(scales x channels x kernels). Gating glob-pooled per-scale
compatibilities with a sigmoid/softmax and multiplying the projected
features is O(scales x channels) per level regardless of kernel size, and
softmax gives anti-competition: putting weight on one scale automatically
dampens the others, which pure-sigmoid gating does not.

## Why a recurrent finest level
Errors in depth concentrate at object boundaries and depth discontinuities,
which are only visible to stride-4 features. Rather than a second decoder
pass, we re-run the finest level with shared weights, allowing entries
through the CGSA gate to be re-weighted given the first-pass prediction —
an "error-correcting" interface point with negligible parameter cost.

## Why a stem-guidance head
Boundary accuracy is limited by information at stride 4 in the decoder. The
stem (stride 2, 24 ch) is essentially free and contains exactly the
high-frequency shape content needed to pin edges; guided fusion of the
decoder feature with this detail path recovers edges that upsampled decoder
features cannot represent.

## Training details
- Input 384x384 (NYU) or 384x1216 (KITTI; set --size).
- SILog (lambda=0.5) is the primary loss; multi-scale gradient matching
  (scales 1, 2) sharpens structure; edge-aware smoothness prevents
  random-scale degenerate solutions at low parameter counts; optional SSI
  loss (`--ssi-weight 0.3`) for MDiT-/MiDaS-style relative-depth training.
- AdamW, poly LR decay, AMP on CUDA.
