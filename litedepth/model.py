import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import build_encoder_any
from .decoder import LiteDepthDecoder


class LiteDepthGD(nn.Module):
    """LiteDepth-GD: lightweight monocular depth estimator.

    Novel decoder connections:
      - CrossScaleGate: every decoder level fuses ALL encoder scales through
        source-softmax per-channel gates (scales compete for channels).
      - Recurrent Gated Refinement: the finest decoder level is run k times
        with shared weights, re-gating skips each pass.
      - DetailGuidanceHead: high-resolution stem features modulate the
        final x2 PixelShuffle upsampling to restore fine boundaries.
    """

    def __init__(
        self,
        encoder="tinynet",
        recurrent=2,
        disable_gated_skips=False,
        disable_recurrence=False,
        disable_detail_head=False,
    ):
        super().__init__()
        enc, chs, detail_ch = build_encoder_any(encoder)
        self.encoder = enc
        self.decoder = LiteDepthDecoder(
            chs,
            detail_ch,
            recurrent=recurrent,
            disable_gated_skips=disable_gated_skips,
            disable_recurrence=disable_recurrence,
            disable_detail_head=disable_detail_head,
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)


def build_model(cfg=None, **kwargs):
    if cfg is None:
        return LiteDepthGD(**kwargs)
    return LiteDepthGD(
        encoder=cfg.get("encoder", "tinynet"),
        recurrent=cfg.get("recurrent", 2),
        disable_gated_skips=cfg.get("disable_gated_skips", False),
        disable_recurrence=cfg.get("disable_recurrence", False),
        disable_detail_head=cfg.get("disable_detail_head", False),
    )
