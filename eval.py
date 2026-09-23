"""Evaluate one or more YOLO checkpoints on validation without touching test."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
import ultralytics.data.dataset as dataset_module
import yaml


ROOT = Path(__file__).resolve().parent
EDGES = (0.05, 0.10, 0.20, 0.35)
BUCKETS = ('<5%', '5-10%', '10-20%', '20-35%', '>=35%')
CONF = 0.25
MATCH_IOU = 0.50
VAL_OPTIONS = dict(imgsz=640, batch=1, workers=0, device=0, half=False,
                   conf=0.001, iou=0.7, max_det=300, augment=False,
                   rect=True, seed=42, deterministic=True, cache=False,
                   plots=False, save_txt=True, save_conf=True, verbose=False)


def local_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_yolo(path: Path, *, prediction: bool) -> tuple[np.ndarray, np.ndarray]:
    boxes, confidences = [], []
    if not path.is_file() and prediction:
        return np.empty((0, 4)), np.empty(0)
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        expected = 6 if prediction else 5
        if len(fields) != expected or fields[0] != '0':
            raise ValueError(f'Invalid one-class YOLO row: {path}:{line_number}')
        cx, cy, bw, bh = map(float, fields[1:5])
        minimum = 0 if prediction else 1e-12
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and minimum <= bw <= 1 and minimum <= bh <= 1):
            raise ValueError(f'Invalid YOLO coordinates: {path}:{line_number}')
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        if prediction:
            confidences.append(float(fields[5]))
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(confidences)


def bucket(box: np.ndarray) -> int:
    # sqrt(area) / image edge; for normalized YOLO boxes this is sqrt(width * height).
    size = np.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    return sum(size >= edge for edge in EDGES)


def iou_matrix(gt: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if not len(gt) or not len(predictions):
        return np.zeros((len(gt), len(predictions)))
    low = np.maximum(gt[:, None, :2], predictions[None, :, :2])
    high = np.minimum(gt[:, None, 2:], predictions[None, :, 2:])
    intersection = np.prod(np.maximum(high - low, 0), axis=2)
    gt_area = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0), axis=1)
    pred_area = np.prod(np.maximum(predictions[:, 2:] - predictions[:, :2], 0), axis=1)
    return intersection / np.maximum(gt_area[:, None] + pred_area[None, :] - intersection, 1e-12)


def match_at_fixed_conf(gt: np.ndarray, pred: np.ndarray, confidence: np.ndarray):
    """Greedy confidence-ordered one-to-one matching, as in the prior error audit."""
    eligible = np.flatnonzero(confidence >= CONF)
    overlaps = iou_matrix(gt, pred[eligible])
    used_gt = set()
    matches = []
    matched_pred = set()
    for local_index in np.argsort(-confidence[eligible]):
        candidates = [g for g in range(len(gt))
                      if g not in used_gt and overlaps[g, local_index] >= MATCH_IOU]
        if candidates:
            g = max(candidates, key=lambda item: overlaps[item, local_index])
            p = int(eligible[local_index])
            used_gt.add(g)
            matched_pred.add(p)
            matches.append((g, p, float(overlaps[g, local_index])))
    return eligible, matches, matched_pred


def validate_data(data_path: Path) -> tuple[Path, list[Path], dict[str, str]]:
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not {'path', 'val', 'names'} <= data.keys():
        raise ValueError('Data YAML must contain path, val and names.')
    if data['names'] != {0: 'pothole'}:
        raise ValueError('This diagnostic currently expects one pothole class.')
    if 'test' in Path(str(data['val'])).parts:
        raise ValueError('Refusing to evaluate the test directory.')
    dataset = local_path(data['path'])
    images_dir = dataset / data['val']
    labels_dir = images_dir.parent / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError('Validation images or labels folder missing.')
    images = sorted(p for p in images_dir.rglob('*')
                    if p.is_file() and p.suffix.lower().lstrip('.') in IMG_FORMATS)
    if not images:
        raise RuntimeError('No validation images.')
    hashes = {}
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix('.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        read_yolo(label, prediction=False)
        hashes[str(image)] = sha256(image)
        hashes[str(label)] = sha256(label)
    return images_dir, images, hashes


def diagnostics(images_dir: Path, images: list[Path], prediction_dir: Path) -> dict:
    labels_dir = images_dir.parent / 'labels'
    stats = [dict(gt=0, tp=0, fp=0) for _ in BUCKETS]
    tp_ious = []
    for image in images:
        relative = image.relative_to(images_dir).with_suffix('.txt')
        gt, _ = read_yolo(labels_dir / relative, prediction=False)
        pred, conf = read_yolo(prediction_dir / relative, prediction=True)
        eligible, matches, matched_pred = match_at_fixed_conf(gt, pred, conf)
        for box in gt:
            stats[bucket(box)]['gt'] += 1
        for g, _, overlap in matches:
            stats[bucket(gt[g])]['tp'] += 1
            tp_ious.append(overlap)
        for p in eligible:
            if int(p) not in matched_pred:
                stats[bucket(pred[p])]['fp'] += 1
    per_bucket = {}
    for name, item in zip(BUCKETS, stats):
        per_bucket[name] = dict(gt=item['gt'], tp=item['tp'],
                                fn=item['gt'] - item['tp'],
                                recall=item['tp'] / item['gt'] if item['gt'] else None,
                                fp=item['fp'])
    return dict(confidence=CONF, match_iou=MATCH_IOU, buckets=per_bucket,
                tp=sum(item['tp'] for item in stats), fp=sum(item['fp'] for item in stats),
                fn=sum(item['gt'] - item['tp'] for item in stats),
                tp_iou=dict(count=len(tp_ious),
                            median=float(np.median(tp_ious)) if tp_ious else None,
                            fraction_ge_075=float(np.mean(np.asarray(tp_ious) >= 0.75)) if tp_ious else None,
                            fraction_ge_090=float(np.mean(np.asarray(tp_ious) >= 0.90)) if tp_ious else None),
                bucket_definition='sqrt(normalized box width * normalized box height); FP assigned by predicted box size')


def markdown_report(report: dict) -> str:
    models = report['models']
    lines = ['# Validation karşılaştırması', '',
             f"Veri: `{report['data']}` · Bölüm: `val` · Görüntü: {report['images']}.", '',
             '| Model | P | R | mAP50 | mAP50–95 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in models:
        lines.append(f"| {row['name']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                     f"{row['map50']:.3f} | {row['map50_95']:.3f} |")
    lines.extend(['', f'Boyut tanısı: conf={CONF}, IoU={MATCH_IOU}; '
                  'kutu ölçeği √(normalize genişlik × normalize yükseklik).', '',
                  '| Model | Kova | GT | TP | FN | Recall | FP |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in models:
        for name, item in row['diagnostics']['buckets'].items():
            recall = f"{item['recall']:.3f}" if item['recall'] is not None else '—'
            lines.append(f"| {row['name']} | {name} | {item['gt']} | {item['tp']} | "
                         f"{item['fn']} | {recall} | {item['fp']} |")
    lines.extend(['', '| Model | TP IoU medyan | IoU ≥0.75 | IoU ≥0.90 |',
                  '| --- | ---: | ---: | ---: |'])
    for row in models:
        item = row['diagnostics']['tp_iou']
        values = ['—' if item[key] is None else f'{item[key]:.3f}'
                  for key in ('median', 'fraction_ge_075', 'fraction_ge_090')]
        lines.append(f"| {row['name']} | {' | '.join(values)} |")
    lines.extend(['', 'FP, tahmin kutusunun boyut kovasına atanır. '
                  'P/R/mAP değerleri Ultralytics `model.val()` çıktısıdır; '
                  'kova ve IoU tanısı ayrıca conf 0.25 ile hesaplanır.', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, action='append', required=True,
                        help='Repeat to compare checkpoints side by side.')
    parser.add_argument('--data', type=Path, default=Path('data.reviewed-v1.yaml'))
    parser.add_argument('--split', choices=('val',), default='val',
                        help='Only validation; use evaluate_reviewed_test.py for final test.')
    parser.add_argument('--nms-iou', type=float, default=0.7,
                        help='NMS IoU for model.val; default matches the original training recipe.')
    args = parser.parse_args()
    if not 0 < args.nms_iou < 1:
        parser.error('--nms-iou must be between zero and one.')
    data_path = local_path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    images_dir, images, input_hashes = validate_data(data_path)
    weights = [local_path(path) for path in args.weights]
    if len(set(weights)) != len(weights):
        parser.error('Duplicate --weights checkpoint.')
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = [path.parent.parent.name if path.parent.name == 'weights' else path.stem for path in weights]
    if len(set(labels)) != len(labels):
        labels = [f'{label}_{index + 1}' for index, label in enumerate(labels)]
    tag = labels[0] if len(labels) == 1 else 'compare'
    tag = re.sub(r'[^A-Za-z0-9_-]+', '-', tag)[:90]
    output = ROOT / 'reports' / f'eval_{tag}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT={output}', flush=True)
    # The validator otherwise writes .cache next to existing, verified labels.
    def memory_only_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_only_cache
    result_rows = []
    for name, weight in zip(labels, weights):
        weight_hash = sha256(weight)
        print(f'Evaluating {name}: {weight}', flush=True)
        model = YOLO(str(weight))
        val_options = dict(VAL_OPTIONS, iou=args.nms_iou)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            with redirect_stdout(log), redirect_stderr(log):
                metrics = model.val(data=str(data_path), split='val',
                                    project=str(output), name=name, exist_ok=False,
                                    **val_options)
        row = dict(name=name, weights=str(weight), weights_sha256=weight_hash,
                   precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                   map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                   inference_ms_per_image=float(metrics.speed['inference']),
                   diagnostics=diagnostics(images_dir, images, output / name / 'labels'))
        if sha256(weight) != weight_hash:
            raise RuntimeError(f'Weights changed during evaluation: {weight}')
        result_rows.append(row)
        print(f"{name}: P={row['precision']:.4f}, R={row['recall']:.4f}, "
              f"mAP50={row['map50']:.4f}, mAP50-95={row['map50_95']:.4f}", flush=True)
    if any(sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError('Validation data changed during evaluation.')
    report = dict(created=datetime.now().isoformat(), data=str(data_path),
                  data_sha256=sha256(data_path), split='val', images=len(images),
                  validation_unchanged=True, test_accessed=False,
                  val_options=dict(VAL_OPTIONS, iou=args.nms_iou), models=result_rows)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'summary.md').write_text(markdown_report(report), encoding='utf-8')
    print(f'Summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
