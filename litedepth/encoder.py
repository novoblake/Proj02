import torch
import torch.nn as nn


def act():
    return nn.SiLU(inplace=True)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, use_act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            act() if use_act else nn.Identity(),
        )


class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1),
            act(),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class MBConv(nn.Module):
    def __init__(self, cin, cout, s=1, k=3, exp=3, use_se=True):
        super().__init__()
        mid = cin * exp
        self.use_residual = s == 1 and cin == cout
        layers = []
        if exp != 1:
            layers += [ConvBNAct(cin, mid, 1)]
        layers += [
            ConvBNAct(mid, mid, k, s, groups=mid),
            SE(mid) if use_se else nn.Identity(),
            nn.Conv2d(mid, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class TinyNet(nn.Module):
    widths = (24, 48, 96, 160, 256)
    depths = (1, 2, 3, 3, 2)

    def __init__(self, in_ch=3):
        super().__init__()
        w = self.widths
        self.stem = ConvBNAct(in_ch, w[0], 3, 2)
        self.stage1 = nn.Sequential(
            ConvBNAct(w[0], w[1], 3, 2),
            *[MBConv(w[1], w[1], 1, 3) for _ in range(self.depths[1])],
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(w[1], w[2], 3, 2),
            *[MBConv(w[2], w[2], 1, 3 if i % 2 == 0 else 5) for i in range(self.depths[2])],
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(w[2], w[3], 3, 2),
            *[MBConv(w[3], w[3], 1, 3 if i % 2 == 0 else 5, exp=2) for i in range(self.depths[3])],
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(w[3], w[4], 3, 2),
            *[MBConv(w[4], w[4], 1, 3 if i % 2 == 0 else 5, exp=2) for i in range(self.depths[4])],
        )

    def forward(self, x):
        d = self.stem(x)
        c1 = self.stage1(d)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return {"detail": d, "s1": c1, "s2": c2, "s3": c3, "s4": c4}


def build_encoder(name="tinynet"):
    if name != "tinynet":
        raise ValueError(f"unknown encoder '{name}'")
    enc = TinyNet()
    return enc, list(enc.widths[1:]), enc.widths[0]


class MobileNetV3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.detail_ch = 16
        self.channels = [16, 24, 48, 576]

    def forward(self, x):
        stems = self.features[0:1](x)
        c1 = self.features[1:2](stems)
        c2 = self.features[2:4](c1)
        c3 = self.features[4:9](c2)
        c4 = self.features[9:](c3)
        return {"detail": stems, "s1": c1, "s2": c2, "s3": c3, "s4": c4}


def build_encoder_any(name="tinynet"):
    if name == "mobilenetv3":
        enc = MobileNetV3Encoder()
        return enc, enc.channels, enc.detail_ch
    return build_encoder(name)
