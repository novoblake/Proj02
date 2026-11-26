import argparse
import shutil
from pathlib import Path


def find_gt(gt_dir):
    for cand in (gt_dir, gt_dir / "data"):
        if cand.is_dir() and any(cand.glob("*.png")):
            return {p.name: p for p in cand.glob("*.png")}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kitti-root",
        required=True,
        help="KITTI depth-prediction root containing train/<drive>/... and val/<drive>/...",
    )
    ap.add_argument("--out", required=True, help="output root, will contain out/images and out/depth")
    args = ap.parse_args()

    root = Path(args.kitti_root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            print(f"skip missing {split_root}")
            continue
        n = 0
        for rgb_dir in sorted(split_root.glob("*/image_02/data")) + sorted(split_root.glob("*/*/image_02/data")):
            drive_dir = rgb_dir.parent.parent
            gt_by_name = find_gt(drive_dir / "proj_depth" / "groundtruth" / "image_02")
            for img in sorted(rgb_dir.glob("*.png")):
                dep = gt_by_name.get(img.name)
                if dep is None:
                    continue
                name = f"{split}_{drive_dir.name}_{img.stem}.png"
                di, dd = out / "images" / name, out / "depth" / name
                if not di.exists():
                    shutil.copy2(img, di)
                if not dd.exists():
                    shutil.copy2(dep, dd)
                n += 1
        print(f"{split}: {n} pairs")
        total += n
    print(f"total: {total} pairs -> {out}/images, {out}/depth")


if __name__ == "__main__":
    main()
