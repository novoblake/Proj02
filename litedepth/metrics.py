import torch


def eigen_crop(shape, dataset="nyu"):
    top, bottom, left, right = 45, 471, 41, 601
    _, _, h, w = shape
    return (
        slice(int(top / 480 * h), int(bottom / 480 * h)),
        slice(int(left / 640 * w), int(right / 640 * w)),
    )


def garg_crop(shape, dataset="kitti"):
    _, _, h, w = shape
    top = int(0.40810811 * h)
    bottom = int(0.99135135 * h)
    left = int(0.03594771 * w)
    right = int(0.96405229 * w)
    return (slice(top, bottom), slice(left, right))


CROPS = {"nyu": eigen_crop, "kitti": garg_crop}


def compute_depth_metrics(pred, target, dataset="nyu", min_depth=1e-3, max_depth=10.0):
    if pred.shape[-2:] != target.shape[-2:]:
        import torch.nn.functional as F

        pred = F.interpolate(
            pred, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    crop = CROPS[dataset](pred.shape, dataset)
    p = pred[:, :, crop[0], crop[1]]
    t = target[:, :, crop[0], crop[1]]
    mask = (t > min_depth) & (t < max_depth)
    if mask.sum() == 0:
        return {}
    p, t = p[mask], t[mask]
    abs_rel = ((p - t).abs() / t).mean().item()
    sq_rel = (((p - t) ** 2) / t).mean().item()
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()
    rmse_log = torch.sqrt(((torch.log(1 + p) - torch.log(1 + t)) ** 2).mean()).item()
    log10 = (torch.abs(torch.log10(p) - torch.log10(t))).mean().item()
    thresh = torch.max(p / t, t / p)
    deltas = [(thresh < 1.25**k).float().mean().item() for k in (1, 2, 3)]
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "log10": log10,
        "d1": deltas[0],
        "d2": deltas[1],
        "d3": deltas[2],
    }


def median_scale_align(pred, target, min_depth=1e-3, max_depth=10.0):
    mask = (target > min_depth) & (target < max_depth)
    if mask.sum() == 0:
        return pred
    s = torch.median(target[mask]) / torch.median(pred[mask])
    return pred * s
