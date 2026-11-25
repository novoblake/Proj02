import torch
import torch.nn as nn
import torch.nn.functional as F


def silog_loss(pred, target, valid_mask, lam=0.5, eps=1e-6):
    p = torch.log(pred.clamp(min=eps))[valid_mask]
    t = torch.log(target[valid_mask])
    d = p - t
    return torch.sqrt((d**2).mean() - lam * d.mean() ** 2 + eps)


def ssi_loss(pred, target, valid_mask, eps=1e-6):
    n_valid = valid_mask.sum(dim=(1, 2, 3), keepdim=True).float()
    t = target * valid_mask.float()
    p = pred * valid_mask.float()
    t_mean = t.sum(dim=(1, 2, 3), keepdim=True) / n_valid.clamp(min=1)
    p_mean = p.sum(dim=(1, 2, 3), keepdim=True) / n_valid.clamp(min=1)
    t_c = (t - t_mean) * valid_mask.float()
    p_c = (p - p_mean) * valid_mask.float()
    scale = (t_c * p_c).sum(dim=(1, 2, 3), keepdim=True) / (
        (p_c**2).sum(dim=(1, 2, 3), keepdim=True).clamp(min=eps)
    )
    shift = t_mean - scale * p_mean
    aligned = scale * pred + shift
    return ((aligned - target)[valid_mask].abs()).mean()


def gradient_matching(pred, target, valid_mask, weights=(1.0, 0.5)):
    loss = 0.0
    w_total = 0.0
    for k, w in enumerate(weights):
        scale = 2**k
        if scale > 1:
            size = (target.shape[-2] // scale, target.shape[-1] // scale)
            p = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
            t = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
            m = F.interpolate(valid_mask.float(), size=size, mode="nearest") > 0.5
        else:
            p, t, m = pred, target, valid_mask
        dx_p, dy_p = p[..., 1:] - p[..., :-1], p[..., 1:, :] - p[..., :-1, :]
        dx_t, dy_t = t[..., 1:] - t[..., :-1], t[..., 1:, :] - t[..., :-1, :]
        dx_m = m[..., 1:] * m[..., :-1]
        dy_m = m[..., 1:, :] * m[..., :-1, :]
        if dx_m.any():
            loss = loss + w * (torch.abs(dx_p - dx_t)[dx_m]).mean()
        if dy_m.any():
            loss = loss + w * (torch.abs(dy_p - dy_t)[dy_m]).mean()
        w_total += w
    return loss / max(w_total, 1e-6)


def edge_aware_smoothness(pred, image, eps=1e-6):
    dx_d = torch.abs(pred[..., 1:] - pred[..., :-1]).mean()
    dy_d = torch.abs(pred[..., 1:, :] - pred[..., :-1, :]).mean()
    dx_i = torch.abs(image[..., 1:] - image[..., :-1]).mean(1, keepdim=True)
    dy_i = torch.abs(image[..., 1:, :] - image[..., :-1, :]).mean(1, keepdim=True)
    return dx_d * torch.exp(-dx_i.mean() / eps) + dy_d * torch.exp(-dy_i.mean() / eps)


class DepthLoss(nn.Module):
    def __init__(
        self,
        silog_weight=1.0,
        grad_weight=0.5,
        smooth_weight=0.05,
        ssi_weight=0.0,
        min_depth=1e-3,
    ):
        super().__init__()
        self.silog_weight = silog_weight
        self.grad_weight = grad_weight
        self.smooth_weight = smooth_weight
        self.ssi_weight = ssi_weight
        self.min_depth = min_depth

    def forward(self, pred, target, image):
        target = target.clamp(min=self.min_depth, max=10.0)
        valid_mask = target > self.min_depth
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0, {}
        loss = 0.0
        parts = {}
        if self.silog_weight > 0:
            parts["silog"] = silog_loss(pred, target, valid_mask)
            loss = loss + self.silog_weight * parts["silog"]
        if self.ssi_weight > 0:
            parts["ssi"] = ssi_loss(pred, target, valid_mask)
            loss = loss + self.ssi_weight * parts["ssi"]
        if self.grad_weight > 0:
            parts["grad"] = gradient_matching(pred, target, valid_mask)
            loss = loss + self.grad_weight * parts["grad"]
        if self.smooth_weight > 0:
            parts["smooth"] = edge_aware_smoothness(pred, image)
            loss = loss + self.smooth_weight * parts["smooth"]
        return loss, parts
