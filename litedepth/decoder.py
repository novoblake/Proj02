import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvBNAct, act


def _fit(x, size):
    if x.shape[-2:] == size:
        return x
    if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class CrossScaleGate(nn.Module):
    """Source-softmax channel gating over ALL encoder scales, not just the
    matching one. Each source is pooled to a small grid, projected, and the
    per-source per-channel gates are softmax-normalized so scales compete
    for each channel instead of being added blindly."""

    def __init__(self, src_chs, ch, bins=4):
        super().__init__()
        self.bins = bins
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                )
                for c in src_chs
            ]
        )
        self.n = len(src_chs)
        gate_feat = 8
        self.gate_in = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bins),
                    nn.Conv2d(c, gate_feat, 1),
                )
                for c in src_chs
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(self.n * gate_feat * bins * bins, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, self.n * ch),
        )

    def forward(self, sources, prev_up, size):
        projs = [p(_fit(s, size)) for s, p in zip(sources, self.projs)]
        pooled = [p_.flatten(1) for p_ in [gi(s) for s, gi in zip(sources, self.gate_in)]]
        g = self.gate(torch.cat(pooled, dim=1))
        g = g.view(-1, self.n, projs[0].shape[1], 1, 1)
        g = torch.softmax(g, dim=1)
        fused = sum(g[:, i] * projs[i] for i in range(self.n))
        return torch.cat([prev_up, fused], dim=1)


class DecoderLevel(nn.Module):
    def __init__(self, prev_ch, ch, src_chs, recurrent=1):
        super().__init__()
        self.gate = CrossScaleGate(src_chs, ch // 2)
        self.block = nn.Sequential(
            ConvBNAct(prev_ch + ch // 2, ch, 1, 1),
            ConvBNAct(ch, ch, 3, 1, groups=ch),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            act(),
        )
        self.recurrent = recurrent
        self.rec_proj = (
            nn.Conv2d(ch, prev_ch, 1, bias=False) if recurrent > 1 else None
        )

    def forward(self, prev, sources, size):
        x = self.block(self.gate(sources, prev, size))
        for _ in range(self.recurrent - 1):
            e = self.block(self.gate(sources, self.rec_proj(x), size))
            x = x + e
        return x


class DetailGuidanceHead(nn.Module):
    """Upsamples the decoder feature x2 and modulates it with the thin
    high-resolution stem features, recovering boundaries that coarse
    decoder features lose."""

    def __init__(self, dec_ch, detail_ch, out_ch=16, uncertainty=False):
        super().__init__()
        self.up = nn.PixelShuffle(2)
        self.fuse = nn.Sequential(
            ConvBNAct(dec_ch // 4 + detail_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, groups=out_ch),
        )
        self.head = nn.Conv2d(out_ch, 2 if uncertainty else 1, 1)
        self.detail_proj = nn.Conv2d(detail_ch, detail_ch, 1, bias=False)

    def forward(self, dec, detail):
        x = self.up(dec)
        d = self.detail_proj(detail)
        x = self.fuse(torch.cat([x, d], dim=1))
        out = self.head(x)
        if out.shape[1] == 2:
            return torch.sigmoid(out[:, :1]), out[:, 1:]
        return torch.sigmoid(out)


class LiteDepthDecoder(nn.Module):
    def __init__(
        self,
        enc_chs,
        detail_ch,
        chs=(224, 144, 96, 64),
        recurrent=2,
        disable_gated_skips=False,
        disable_recurrence=False,
        disable_detail_head=False,
        uncertainty=False,
    ):
        super().__init__()
        self.n_levels = len(chs)
        self.disable_gated_skips = disable_gated_skips
        self.bottleneck = ConvBNAct(enc_chs[-1], chs[0], 1, 1)
        if disable_gated_skips:
            self.levels = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvBNAct((chs[i - 1] if i > 0 else chs[0]) + enc_chs[i], chs[i], 1, 1),
                        ConvBNAct(chs[i], chs[i], 3, 1, groups=chs[i]),
                        nn.Conv2d(chs[i], chs[i], 1, bias=False),
                        nn.BatchNorm2d(chs[i]),
                        act(),
                    )
                    for i in range(self.n_levels)
                ]
            )
            self.recurrent = 1
        else:
            self.levels = nn.ModuleList(
                [
                    DecoderLevel(
                        chs[i - 1] if i > 0 else chs[0],
                        chs[i],
                        enc_chs,
                        recurrent=1 if (disable_recurrence or i < self.n_levels - 1) else recurrent,
                    )
                    for i in range(self.n_levels)
                ]
            )
            self.recurrent = 1 if disable_recurrence else recurrent
        head_kwargs = dict(dec_ch=chs[-1], detail_ch=detail_ch, uncertainty=uncertainty)
        self.head = (
            DetailGuidanceHead(**head_kwargs)
            if not disable_detail_head
            else SimpleHead(**head_kwargs)
        )

    def forward(self, feats):
        _, h, w = feats["s1"].shape[1:]
        srcs = [feats["s1"], feats["s2"], feats["s3"], feats["s4"]]
        x = self.bottleneck(feats["s4"])
        for i, lvl in enumerate(self.levels):
            size = (h >> (3 - i), w >> (3 - i))
            if x.shape[-2:] != size:
                up = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            else:
                up = x
            if self.disable_gated_skips:
                x = lvl(torch.cat([up, _fit(srcs[i], size)], dim=1))
            else:
                x = lvl(up, srcs, size)
        out = self.head(x, feats["detail"])
        if isinstance(out, tuple):
            depth, logvar = out
            depth = F.interpolate(depth, scale_factor=2, mode="bilinear", align_corners=False)
            logvar = F.interpolate(logvar, scale_factor=2, mode="bilinear", align_corners=False)
            return depth, logvar
        return F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)


class SimpleHead(nn.Module):
    def __init__(self, dec_ch, detail_ch, uncertainty=False):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(dec_ch + detail_ch, 32, 3, 1),
            nn.Conv2d(32, 2 if uncertainty else 1, 1),
        )

    def forward(self, dec, detail):
        d = F.interpolate(detail, size=dec.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv(torch.cat([dec, d], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[1] == 2:
            return torch.sigmoid(x[:, :1]), x[:, 1:]
        return torch.sigmoid(x)
