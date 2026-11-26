"""Plug-in exports for generality studies.

ScaleCompetitionGate is the decoder-connection module of LiteDepth-GD
(source-softmax, per-channel gated fusion over ALL pyramid scales). Use it
to replace additive/concat skip fusion inside any other encoder-decoder:

    from litedepth.plugins import ScaleCompetitionGate

    gate = ScaleCompetitionGate(src_chs=[64, 128, 256, 512], ch=128)
    out = gate([f1, f2, f3, f4], upsampled_prev, out_size)
    # returns cat([upsampled_prev, fused], dim=1) for the level's conv block

Experiments swapping this single module into other backbones (ResUNet,
MobileUNet, SegFormer-lite) isolate its contribution from the rest of the
architecture — that plug-in generality is what elevates the claim beyond a
single model.
"""

from .decoder import CrossScaleGate as ScaleCompetitionGate

__all__ = ["ScaleCompetitionGate"]
