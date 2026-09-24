"""Build MWPD + scale-matched HRP4K training data.

HRP4K potholes are small in 4K frames (median box ~2% of the edge, ~60 px), while MWPD boxes are
~17% of a 640 image. Plain 640 tiling left HRP4K boxes at ~9%. Here every crop is centred on a
random HRP4K box and sized so that box lands at a size drawn from the MWPD train distribution;
the crop is then resized to 640. Other boxes in the crop are kept if >30% visible.

Output (new folder, sources untouched):
    data/MWPD_HRP4K_scaled_v1/train   MWPD_reviewed_v1 train copy + HRP4K train crops
    data/MWPD_HRP4K_scaled_v1/valid   MWPD_reviewed_v1 valid copy (unchanged benchmark)
    data/HRP4K_scaled_valid_v1/valid  HRP4K valid crops, only for the cross-dataset check
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from tile_yolo_dataset import boxes_from_yolo, project_box

ROOT = Path(__file__).resolve().parent
# Raw HRP4K root (train/valid/test, each with images/ and labels/). Override with --hrp4k or HRP4K_ROOT.
HRP4K = Path(os.environ.get('HRP4K_ROOT', ROOT / 'data/HRP4K'))
MWPD = ROOT / 'data/MWPD_reviewed_v1'
OUT_SIZE = 640


def mwpd_box_sizes() -> np.ndarray:
    sizes = []
    for label in (MWPD / 'train/labels').glob('*.txt'):
        for row in label.read_text(encoding='utf-8').splitlines():
            if row.strip():
                _, _, _, w, h = map(float, row.split())
                sizes.append((w * h) ** 0.5)
    return np.asarray(sizes)


def square_crop(anchor: tuple[float, ...], side: int, width: int, height: int,
                rng: random.Random) -> tuple[int, int, int, int]:
    """Random square window of `side` px that fully contains `anchor` (x1, y1, x2, y2)."""
    side = min(side, width, height)
    x = rng.uniform(max(0, anchor[2] - side), min(anchor[0], width - side))
    y = rng.uniform(max(0, anchor[3] - side), min(anchor[1], height - side))
    return int(x), int(y), int(x) + side, int(y) + side


def write_crop(image: np.ndarray, crop: tuple[int, int, int, int], boxes: list, stem: str,
               out_dir: Path, sizes: list[float]) -> int:
    x, y, right, bottom = crop
    rows = []
    for box in boxes:
        row, _ = project_box(box, x, y, right, bottom, 0.30)
        if row:
            rows.append(row)
            _, _, _, w, h = map(float, row.split())
            sizes.append((w * h) ** 0.5)
    patch = image[y:bottom, x:right]
    interpolation = cv2.INTER_AREA if patch.shape[0] > OUT_SIZE else cv2.INTER_CUBIC
    patch = cv2.resize(patch, (OUT_SIZE, OUT_SIZE), interpolation=interpolation)
    cv2.imwrite(str(out_dir / 'images' / f'{stem}.jpg'), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out_dir / 'labels' / f'{stem}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
    return len(rows)


def crop_split(hrp4k: Path, split: str, out_dir: Path, targets: np.ndarray, crops_per_image: int,
               negative_ratio: float, rng: random.Random) -> dict:
    for sub in ('images', 'labels'):
        (out_dir / sub).mkdir(parents=True, exist_ok=False)
    src = hrp4k / split
    positives, negatives, sizes, crop_sides = 0, [], [], []
    for image_path in sorted((src / 'images').glob('*.jpg')):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        boxes = boxes_from_yolo(src / 'labels' / f'{image_path.stem}.txt', width, height)
        if not boxes:
            negatives.append(image_path)
            continue
        for index, anchor in enumerate(rng.sample(boxes, min(crops_per_image, len(boxes)))):
            box_side = ((anchor[2] - anchor[0]) * (anchor[3] - anchor[1])) ** 0.5
            target = float(np.clip(targets[rng.randrange(len(targets))], 0.05, 0.60))
            # Crop must contain the whole anchor box, otherwise the size target is meaningless.
            side = int(max(box_side / target, anchor[2] - anchor[0], anchor[3] - anchor[1], 160))
            crop = square_crop(anchor, side, width, height, rng)
            write_crop(image, crop, boxes, f'hrp4k_{split}_{image_path.stem}_{index}', out_dir, sizes)
            crop_sides.append(crop[2] - crop[0])
            positives += 1
    kept_negatives = rng.sample(negatives, min(len(negatives), int(positives * negative_ratio)))
    for image_path in kept_negatives:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        side = int(np.clip(rng.choice(crop_sides), 160, height))
        crop = square_crop((width, height, 0, 0), side, width, height, rng)  # any position
        write_crop(image, crop, [], f'hrp4k_{split}_{image_path.stem}_neg', out_dir, sizes)
    sizes = np.asarray(sizes)
    return dict(positive_crops=positives, negative_crops=len(kept_negatives), boxes=len(sizes),
                box_size_p10_p50_p90=np.percentile(sizes, [10, 50, 90]).round(3).tolist(),
                crop_side_px_p10_p50_p90=np.percentile(crop_sides, [10, 50, 90]).round(0).tolist())


def copy_split(src: Path, dst: Path) -> int:
    for sub in ('images', 'labels'):
        shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    return len(list((src / 'images').iterdir()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hrp4k', type=Path, default=HRP4K,
                        help='Raw HRP4K root (default: HRP4K_ROOT env var or data/HRP4K).')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/MWPD_HRP4K_scaled_v1')
    parser.add_argument('--valid-output', type=Path, default=ROOT / 'data/HRP4K_scaled_valid_v1')
    parser.add_argument('--crops-per-image', type=int, default=1)
    parser.add_argument('--negative-ratio', type=float, default=0.15,
                        help='Negative crops as a fraction of positive crops.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not (args.hrp4k / 'train/images').is_dir():
        parser.error(f'{args.hrp4k} has no train/images; pass --hrp4k or set HRP4K_ROOT.')
    for path in (args.output, args.valid_output):
        if path.exists():
            raise FileExistsError(f'{path} exists; never overwriting data folders.')

    targets = mwpd_box_sizes()
    summary = dict(seed=args.seed, crops_per_image=args.crops_per_image, negative_ratio=args.negative_ratio,
                   mwpd_train_box_size_p10_p50_p90=np.percentile(targets, [10, 50, 90]).round(3).tolist())
    summary['hrp4k_train'] = crop_split(args.hrp4k, 'train', args.output / 'train', targets, args.crops_per_image,
                                        args.negative_ratio, random.Random(args.seed))
    summary['mwpd_train_images'] = copy_split(MWPD / 'train', args.output / 'train')
    summary['mwpd_valid_images'] = copy_split(MWPD / 'valid', args.output / 'valid')
    summary['hrp4k_valid'] = crop_split(args.hrp4k, 'valid', args.valid_output / 'valid', targets, 1,
                                        args.negative_ratio, random.Random(args.seed + 1))

    for path, train in ((args.output, 'train/images'), (args.valid_output, 'valid/images')):
        (path / 'data.yaml').write_text(yaml.safe_dump(dict(
            path=path.as_posix(), train=train, val='valid/images', names={0: 'pothole'})), encoding='utf-8')
        (path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
