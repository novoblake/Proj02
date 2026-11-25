import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_dir(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)


class ImageDepthDataset(Dataset):
    """Generic paired dataset. Expects `root/images` and `root/depth`
    folders with matching (sorted) filenames, e.g. NYUv2/labeled or a
    KITTI-style exported pairs folder. Depth maps in png: divide by
    `depth_scale` (256 for 8-bit exports, 1000 for KITTI-style 16-bit)."""

    def __init__(
        self,
        root,
        split="train",
        size=(384, 384),
        depth_scale=1000.0,
        min_depth=1e-3,
        max_depth=10.0,
        augment=True,
    ):
        root = Path(root)
        self.images = _list_dir(root / "images")
        self.depths = _list_dir(root / "depth")
        assert len(self.images) == len(self.depths) and len(self.images) > 0, (
            f"{root}: found {len(self.images)} images / {len(self.depths)} depth maps"
        )
        self.size = size
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.images)

    def _load(self, path):
        return Image.open(path)

    def __getitem__(self, idx):
        img = self._load(self.images[idx]).convert("RGB")
        dep = self._load(self.depths[idx])
        w0, h0 = img.size
        if self.size is not None and (h0, w0) != tuple(self.size):
            img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
            dep = dep.resize((self.size[1], self.size[0]), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        dep = np.asarray(dep, dtype=np.float32)
        if dep.ndim == 3:
            dep = dep[..., 0]
        dep = dep / self.depth_scale
        dep[dep < self.min_depth] = 0.0
        dep[dep > self.max_depth] = 0.0
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1].copy()
            dep = dep[:, ::-1].copy()
        img = torch.from_numpy(img).permute(2, 0, 1)
        dep = torch.from_numpy(dep).unsqueeze(0)
        return img, dep


def build_loaders(cfg):
    from torch.utils.data import DataLoader

    train = ImageDepthDataset(
        cfg["data_root"],
        "train",
        size=cfg.get("size", (384, 384)),
        depth_scale=cfg.get("depth_scale", 1000.0),
        max_depth=cfg.get("max_depth", 10.0),
        augment=True,
    )
    val = ImageDepthDataset(
        cfg["val_root"] or cfg["data_root"],
        "val",
        size=cfg.get("size", (384, 384)),
        depth_scale=cfg.get("depth_scale", 1000.0),
        max_depth=cfg.get("max_depth", 10.0),
        augment=False,
    )
    return (
        DataLoader(
            train,
            batch_size=cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        ),
        DataLoader(val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True),
    )
