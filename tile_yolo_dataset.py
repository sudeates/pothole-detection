"""Tile YOLO train images while retaining clipped labels and sampled negatives.

The MWPD V1 validation split is copied unchanged so future models can be
compared on the original, unsliced validation images. No training is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
import yaml


ROOT = Path(__file__).resolve().parent
# Raw HRP4K train split. Override with --source-train or the HRP4K_ROOT env var.
DEFAULT_SOURCE = Path(os.environ.get('HRP4K_ROOT', ROOT / 'data/HRP4K')) / 'train'
DEFAULT_VALIDATION = ROOT / 'data/MWPD_reviewed_v1/valid'
DEFAULT_OUTPUT = ROOT / 'data/HRP4K_tiled_v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, round(tile * (1.0 - overlap)))
    positions = list(range(0, length - tile + 1, stride))
    final = length - tile
    if positions[-1] != final:
        positions.append(final)
    return positions


def boxes_from_yolo(label: Path, width: int, height: int) -> list[tuple[float, ...]]:
    boxes = []
    for line_number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != '0':
            raise ValueError(f'Invalid class/format: {label}:{line_number}')
        cx, cy, bw, bh = map(float, fields[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f'Invalid YOLO coordinates: {label}:{line_number}')
        x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
        y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
        if x1 < -0.001 * width or y1 < -0.001 * height or x2 > 1.001 * width or y2 > 1.001 * height:
            raise ValueError(f'Box extends outside image: {label}:{line_number}')
        boxes.append((max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)))
    return boxes


def project_box(box: tuple[float, ...], x: int, y: int, right: int,
                bottom: int, min_fraction: float) -> tuple[str | None, bool]:
    x1, y1, x2, y2 = box
    ix1, iy1 = max(x1, x), max(y1, y)
    ix2, iy2 = min(x2, right), min(y2, bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, False
    inside_fraction = ((ix2 - ix1) * (iy2 - iy1)) / ((x2 - x1) * (y2 - y1))
    if inside_fraction <= min_fraction:  # User rule: strictly more than threshold.
        return None, True
    width, height = right - x, bottom - y
    new_cx = ((ix1 + ix2) / 2 - x) / width
    new_cy = ((iy1 + iy2) / 2 - y) / height
    new_bw = (ix2 - ix1) / width
    new_bh = (iy2 - iy1) / height
    values = (new_cx, new_cy, new_bw, new_bh)
    if not (all(0 <= value <= 1 for value in values[:2]) and
            all(0 < value <= 1 for value in values[2:])):
        raise RuntimeError(f'Projected box invalid: {values}')
    return '0 ' + ' '.join(f'{value:.8f}' for value in values), inside_fraction < 1.0 - 1e-9


def retain_negative(source: str, x: int, y: int, seed: int, fraction: float) -> bool:
    key = f'{seed}:{source}:{x}:{y}'.encode('utf-8')
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
    return draw < fraction


def verify_file_copy(src: Path, dst: Path, manifest: list[dict], output: Path) -> None:
    source_hash = sha256(src)
    shutil.copy2(src, dst)
    if sha256(dst) != source_hash or sha256(src) != source_hash:
        raise RuntimeError(f'Copy hash mismatch: {src}')
    manifest.append(dict(source=str(src), destination=dst.relative_to(output).as_posix(),
                         sha256=source_hash, source_sha256=source_hash))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-train', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--source-valid', type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--min-box-fraction', type=float, default=0.30)
    parser.add_argument('--negative-fraction', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Use only the first N source train images for a demo.')
    args = parser.parse_args()
    if args.tile_size < 32 or not 0 <= args.overlap < 1:
        parser.error('tile-size must be >=32 and overlap must be in [0,1).')
    if not 0 <= args.min_box_fraction < 1 or not 0 <= args.negative_fraction <= 1:
        parser.error('fractions must be within [0,1] (box threshold strictly below 1).')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')

    source = args.source_train.resolve()
    validation = args.source_valid.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    source_images = sorted(p for p in (source / 'images').iterdir()
                           if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if args.limit is not None:
        source_images = source_images[:args.limit]
    if not source_images:
        raise RuntimeError('No training images found.')
    validation_images = sorted(p for p in (validation / 'images').iterdir() if p.is_file())
    if not validation_images or len(validation_images) != len(list((validation / 'labels').iterdir())):
        raise RuntimeError('Validation images/labels missing or unequal.')

    for split in ('train', 'valid'):
        for kind in ('images', 'labels'):
            (output / split / kind).mkdir(parents=True, exist_ok=False)
    (output / 'metadata').mkdir()
    (output / 'INCOMPLETE').write_text('Generation in progress.', encoding='utf-8')
    files = []
    counts = Counter()
    source_hashes = {}
    example = None

    for index, image_path in enumerate(source_images, 1):
        label_path = source / 'labels' / (image_path.stem + '.txt')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        source_hashes[str(image_path)] = sha256(image_path)
        source_hashes[str(label_path)] = sha256(label_path)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            boxes = boxes_from_yolo(label_path, width, height)
            counts['source_images'] += 1
            counts['source_boxes'] += len(boxes)
            for y in starts(height, args.tile_size, args.overlap):
                for x in starts(width, args.tile_size, args.overlap):
                    # PIL pads beyond a short source edge; output stays fixed-size.
                    right = x + args.tile_size
                    bottom = y + args.tile_size
                    counts['candidate_tiles'] += 1
                    labels = []
                    clipped = 0
                    for box in boxes:
                        projected, partial = project_box(box, x, y, right, bottom,
                                                         args.min_box_fraction)
                        if projected is None:
                            if partial:
                                counts['box_instances_dropped_at_boundary'] += 1
                            continue
                        labels.append(projected)
                        clipped += int(partial)
                        counts['box_instances_kept'] += 1
                    if not labels:
                        counts['empty_tile_candidates'] += 1
                        if not retain_negative(image_path.name, x, y, args.seed,
                                               args.negative_fraction):
                            continue
                        counts['empty_tiles_kept'] += 1
                    else:
                        counts['positive_tiles_kept'] += 1
                        counts['clipped_box_instances_kept'] += clipped
                    tile_name = f'{image_path.stem}__x{x}_y{y}.jpg'
                    tile_image = output / 'train/images' / tile_name
                    tile_label = output / 'train/labels' / (Path(tile_name).stem + '.txt')
                    cropped = image.crop((x, y, right, bottom)).convert('RGB')
                    cropped.save(tile_image, quality=95)
                    tile_label.write_text('\n'.join(labels) + ('\n' if labels else ''),
                                          encoding='utf-8')
                    files.extend([
                        dict(source=str(image_path), destination=tile_image.relative_to(output).as_posix(),
                             sha256=sha256(tile_image), source_sha256=source_hashes[str(image_path)],
                             crop_xyxy=[x, y, right, bottom]),
                        dict(source=str(label_path), destination=tile_label.relative_to(output).as_posix(),
                             sha256=sha256(tile_label), source_sha256=source_hashes[str(label_path)],
                             crop_xyxy=[x, y, right, bottom])])
                    if example is None and labels:
                        preview = cropped.copy()
                        draw = ImageDraw.Draw(preview)
                        for line in labels:
                            _, cx, cy, bw, bh = map(float, line.split())
                            draw.rectangle(((cx - bw / 2) * preview.width,
                                            (cy - bh / 2) * preview.height,
                                            (cx + bw / 2) * preview.width,
                                            (cy + bh / 2) * preview.height),
                                           outline='lime', width=3)
                        example = preview
        if index % 100 == 0 or index == len(source_images):
            print(f'{index}/{len(source_images)} source images processed', flush=True)

    for image_path in validation_images:
        label_path = validation / 'labels' / (image_path.stem + '.txt')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        verify_file_copy(image_path, output / 'valid/images' / image_path.name, files, output)
        verify_file_copy(label_path, output / 'valid/labels' / label_path.name, files, output)
    counts['validation_images_copied'] = len(validation_images)
    counts['train_tiles_written'] = counts['positive_tiles_kept'] + counts['empty_tiles_kept']
    counts['box_instances_dropped_total'] = counts['box_instances_dropped_at_boundary']
    counts['empty_tiles_dropped'] = counts['empty_tile_candidates'] - counts['empty_tiles_kept']

    if example is not None:
        example.save(output / 'metadata/example_positive_tile.jpg', quality=95)
    (output / 'metadata/files.json').write_text(json.dumps(files, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
    (output / 'metadata/source_hashes.json').write_text(json.dumps(source_hashes, indent=2),
                                                        encoding='utf-8')
    data = dict(path=output.as_posix(), train='train/images', val='valid/images',
                names={0: 'pothole'})
    (output / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                      encoding='utf-8')
    report = dict(created=datetime.now().isoformat(), source_train=str(source),
                  source_validation=str(validation), output=str(output), demo=args.limit is not None,
                  parameters=dict(tile_size=args.tile_size, overlap=args.overlap,
                                  min_box_fraction=args.min_box_fraction,
                                  negative_fraction=args.negative_fraction, seed=args.seed),
                  counts=dict(counts),
                  caveats=['Box counts are per tile; overlap can duplicate one source box.',
                           'A retained tile may contain small unlabelled box fragments at its edge.'])
    (output / 'metadata/summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f'Source changed during tiling: {path}')
    for row in files:
        if sha256(output / row['destination']) != row['sha256']:
            raise RuntimeError(f'Output differs from manifest: {row["destination"]}')
    (output / 'VERIFIED.json').write_text(json.dumps(dict(status='verified',
        source_unchanged=True, output_files_verified=len(files),
        validation_unchanged=True, train_images=counts['train_tiles_written'],
        validation_images=counts['validation_images_copied'], training_started=False),
        indent=2), encoding='utf-8')
    (output / 'INCOMPLETE').unlink()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
