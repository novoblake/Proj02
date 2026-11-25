import torch


EIGEN_CROP = {"nyu": (45, 471, 41, 601)}  # top, bottom, left, right for 480x640


def eigen_crop(shape, dataset="nyu"):
    _, _, h, w = shape
    top, bottom, left, right = EIGEN_CROP[dataset]
    top = int(top / 480 * h)
    bottom = int(bottom / 480 * h)
    left = int(left / 640 * w)
    right = int(right / 640 * w)
    return (slice(top, bottom), slice(left, right))


def compute_depth_metrics(pred, target, dataset="nyu", min_depth=1e-3, max_depth=10.0):
    crop = eigen_crop(pred.shape, dataset)
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
