# Journal target plan

## TL;DR recommendation

Target **TCSVT** (IEEE Trans. on Circuits and Systems for Video Technology) or
**RA-L** as the realistic primary journal; **TIP** as stretch. **TPAMI** only
if the plug-in gate experiments generalize across backbones AND training
scales to cross-dataset pretraining — that is a 2-3x scope increase.

## Why this tiering

Novelty is currently **moderate/incremental**: each mechanism (softmax scale
gating, weight-shared recurrent refinement, detail-guided upsampling) is a
recombination of known components. A journal paper is won or lost on
(a) demonstrated benefit vs. same-size and larger baselines, (b) thoroughness
of ablation, (c) evidence the contribution generalizes beyond one
architecture. TCSVT/RA-L accept "thorough empirical systems papers"
frequently; TPAMI/TIP expect a new formulation or very strong results.

## Required experiment matrix (what turns this into a paper)

| Experiment | What it demonstrates | Where |
|---|---|---|
| NYUv2 training, 3 seeds × 5 configs (`experiments/run_ablation.sh`) | Core ablation, mean±std, d1/absrel/rmse + AUSE | `--dataset nyu` |
| KITTI depth, 3 seeds × same configs | Cross-dataset transfer of the conclusion | `--dataset kitti --size 352 1216` |
| Zero-shot: NYU-trained → KITTI (and vice versa), scale-shift-invariant training + median align | Generalization claim | `--ssi-weight 0.3`, `eval.py --median-align` |
| Plug-in gate into 2 other UNets (e.g. ResUNet-mobile, MobileSeg | Generality of CrossScaleGate | `litedepth/plugins.py` |
| Uncertainty: AUSE vs baselines (mono unc head vs MC-dropout) | Bonus novelty axis, calibration rigor | `--uncertainty` |
| Efficiency table: params/GMACs/latency vs FastDepth, MiDaS-small, PETIT etc. | Efficiency claim | `benchmark.py` |

Compute for the core matrix (3 seeds × 5 configs × 2 datasets): roughly
30 runs; a single RTX 4090 / A100 finishes the NYU half in ~1-2 days.

## Venue matrix

| Venue | Fit | Blocker to clear |
|---|---|---|
| TCSVT | primary | clean ablation table + efficiency table + uncertainty |
| RA-L (robotics) | primary | real-time claim + KITTI numbers |
| TITS | good if driving narrative | KITTI + zero-shot + speed on embedded HW (Jetson numbers help a lot) |
| TIP | stretch | needs stronger methodological novelty (e.g. new loss derivation or theoretical motivation of scale competition) |
| TPAMI | not now | 2-3x scope + cross-backbone generalization + large-scale pretraining |

## Rigor checklist before submission

1. ≥3 seeds per config, report mean±std (harness provided).
2. Same budget comparisons: match params/MACs when claiming wins, not just
   absolute numbers.
3. Report the full metric suite (abs_rel, sq_rel, rmse, rmse_log, log10,
   δ1-δ3) with official crops (already implemented).
4. AUSE for uncertainty; calibration curve figure.
5. Qualitative figures: boundaries (DG head), sky/plant regions (CGSA gate
   heatmaps — gate weights are logged at 1x1 softmax, easy to visualize).
6. Deterministic config reproducibility: seed fixed, cudnn deterministic
   flags on (already default in train.py).
