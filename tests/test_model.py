import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litedepth.model import build_model
from litedepth.losses import DepthLoss, silog_loss, ssi_loss
from litedepth.metrics import compute_depth_metrics, eigen_crop


@pytest.mark.parametrize("size", [(256, 256), (256, 320), (384, 512)])
def test_forward_shapes(size):
    model = build_model()
    model.eval()
    x = torch.randn(2, 3, *size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, *size)
    assert out.min() >= 0 and out.max() <= 1


def test_backward_all_params():
    model = build_model()
    out = model(torch.randn(1, 3, 256, 256))
    loss = out.mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad: {missing[:8]}"


def test_ablation_variants():
    for cfg in [
        {"disable_gated_skips": True},
        {"disable_recurrence": True},
        {"disable_detail_head": True},
        {"encoder": "mobilenetv3"},
        {"recurrent": 1},
    ]:
        m = build_model(cfg)
        out = m(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 1, 256, 256)


def test_novel_modules_registered():
    m = build_model()
    from litedepth.decoder import CrossScaleGate

    gates = [mod for mod in m.decoder.modules() if isinstance(mod, CrossScaleGate)]
    assert len(gates) == 4
    assert m.decoder.recurrent == 2


def test_losses_finite_and_masked():
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64) * 9 + 1
    image = torch.rand(2, 3, 64, 64)
    target[:, :, :16] = 0
    crit = DepthLoss()
    loss, parts = crit(pred, target, image)
    assert torch.isfinite(loss)
    assert set(parts) == {"silog", "grad", "smooth"}
    empty_target = torch.zeros_like(target)
    loss2, _ = crit(pred, empty_target, image)
    assert loss2.item() == 0.0


def test_ssi_loss_alignment_invariance():
    torch.manual_seed(0)
    target = torch.rand(1, 1, 32, 32) * 8 + 0.5
    mask = torch.ones_like(target).bool()
    base = ssi_loss(target, target, mask)
    scaled = ssi_loss(2.0 * target + 0.5, target, mask)
    assert base.item() == pytest.approx(0.0, abs=1e-5)
    assert scaled.item() < 1.0


def test_metrics_orientation():
    pred = torch.full((1, 1, 480, 640), 5.0)
    target = torch.full((1, 1, 480, 640), 5.0)
    m = compute_depth_metrics(pred, target)
    assert m["d1"] == 1.0 and m["abs_rel"] == pytest.approx(0.0)
    crop = eigen_crop(pred.shape)
    assert crop[0].start > 0 and crop[1].start > 0


def test_param_budget():
    model = build_model()
    n = sum(p.numel() for p in model.parameters())
    assert n < 4e6, f"lightweight budget exceeded: {n/1e6:.2f}M"
