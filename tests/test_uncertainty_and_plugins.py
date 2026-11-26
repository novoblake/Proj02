import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litedepth.model import build_model
from litedepth.losses import laplacian_nll
from litedepth.metrics import sparsification_ause
from litedepth.plugins import ScaleCompetitionGate


def test_uncertainty_model_two_outputs():
    model = build_model({"uncertainty": True})
    depth, logvar = model(torch.randn(1, 3, 256, 256), return_uncertainty=True)
    assert depth.shape == (1, 1, 256, 256)
    assert logvar.shape == (1, 1, 256, 256)
    assert depth.min() >= 0 and depth.max() <= 1
    base = model(torch.randn(1, 3, 256, 256))
    assert base.shape == (1, 1, 256, 256)


def test_uncertainty_train_step():
    model = build_model({"uncertainty": True})
    from litedepth.losses import DepthLoss

    crit = DepthLoss()
    img = torch.randn(1, 3, 64, 64)
    dep = torch.rand(1, 1, 64, 64) * 8 + 1
    dep[:, :, :8] = 0
    pred, logvar = model(img, return_uncertainty=True)
    loss, parts = crit(pred * 10, dep, img, logvar=logvar)
    assert torch.isfinite(loss)
    assert "unc" in parts
    loss.backward()


def test_laplacian_nll_prefers_correct_confident_pred():
    torch.manual_seed(0)
    target = torch.full((1, 1, 16, 16), 5.0)
    pred = torch.full((1, 1, 16, 16), 5.0)
    mask = torch.ones_like(target).bool()
    confident = torch.full_like(pred, -2.0)
    vague = torch.full_like(pred, 2.0)
    assert laplacian_nll(pred, target, confident, mask) < laplacian_nll(
        pred, target, vague, mask
    )


def test_ause_zero_when_uncertainty_matches_error():
    torch.manual_seed(0)
    target = torch.rand(1, 1, 64, 64) * 9 + 0.2
    pred = target.clone()
    unc = (pred - target).abs()
    ause = sparsification_ause(pred, target, unc)
    assert ause == pytest.approx(0.0, abs=1e-4)


def test_plugin_gate_in_generic_toy_unet():
    torch.manual_seed(0)

    class ToyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = torch.nn.Conv2d(3, 16, 3, 2, 1)
            self.e2 = torch.nn.Conv2d(16, 32, 3, 2, 1)
            self.e3 = torch.nn.Conv2d(32, 64, 3, 2, 1)
            self.red = torch.nn.Conv2d(64, 16, 1)
            self.gate = ScaleCompetitionGate([16, 32, 64], 16)
            self.out = torch.nn.Conv2d(16 + 16, 1, 1)

        def forward(self, x):
            f1, f2, f3 = self.e1(x), self.e2(self.e1(x)), self.e3(self.e2(self.e1(x)))
            level = torch.nn.functional.interpolate(
                self.red(f3), size=f1.shape[-2:], mode="bilinear", align_corners=False
            )
            h = self.gate([f1, f2, f3], level, f1.shape[-2:])
            return self.out(h)

    net = ToyUNet()
    out = net(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 32, 32)
    out.sum().backward()


def test_default_model_has_single_channel_output():
    m = build_model({})
    out = m(torch.randn(1, 3, 128, 128))
    assert out.shape == (1, 1, 128, 128)
