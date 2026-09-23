"""Compare P2, YOLOv8n and YOLOv8s on the same uncropped MWPD validation.

All models use 640-pixel SAHI windows, 20% overlap, full-image prediction and
IoU=0.50 NMS merge. Precision/recall are Ultralytics max-F1 values; AP uses
IoU 0.50:0.95 and a low prediction threshold (default 0.001).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.tools/sahi'))

import numpy as np
from PIL import Image
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.metrics import ap_per_class


VALID = ROOT / 'data/MWPD_reviewed_v1/valid'
DEFAULT_WEIGHTS = {
    'n': ROOT / 'runs/reviewed-v1-baseline_20260915_223719_077806/weights/best.pt',
    's': ROOT / 'runs/reviewed-v1-yolov8s_20260916_122325_277730/weights/best.pt',
    'p2': ROOT / 'runs/p2-hrp4k-v1/weights/best.pt',
}
FIELDS = ('model', 'status', 'images', 'ground_truth_boxes', 'predictions',
          'precision', 'recall', 'map50', 'map50_95', 'mean_ms_per_image',
          'weights', 'weights_sha256', 'error')
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ground_truth(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != '0':
            raise ValueError(f'Invalid one-class YOLO label: {path}:{line_number}')
        x, y, bw, bh = map(float, fields[1:])
        boxes.append(((x - bw / 2) * width, (y - bh / 2) * height,
                      (x + bw / 2) * width, (y + bh / 2) * height))
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def prediction_array(result) -> np.ndarray:
    rows = []
    for item in result.object_prediction_list:
        if item.category.id != 0:
            raise ValueError(f'Unexpected predicted class: {item.category.id}')
        rows.append([*item.bbox.to_xyxy(), item.score.value])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def box_iou(labels: np.ndarray, detections: np.ndarray) -> np.ndarray:
    if not len(labels) or not len(detections):
        return np.zeros((len(labels), len(detections)), dtype=np.float64)
    upper_left = np.maximum(labels[:, None, :2], detections[None, :, :2])
    lower_right = np.minimum(labels[:, None, 2:], detections[None, :, 2:])
    intersection = np.prod(np.maximum(lower_right - upper_left, 0), axis=2)
    label_area = np.prod(np.maximum(labels[:, 2:] - labels[:, :2], 0), axis=1)
    detection_area = np.prod(np.maximum(detections[:, 2:] - detections[:, :2], 0), axis=1)
    return intersection / np.maximum(label_area[:, None] + detection_area[None, :] - intersection, 1e-12)


def correct_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Use the one-to-one IoU matching order used by Ultralytics validation."""
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    iou = box_iou(labels, predictions[:, :4])
    for column, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if len(matches):
            if len(matches) > 1:
                matches = matches[np.argsort(iou[matches[:, 0], matches[:, 1]])[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), column] = True
    return correct


def verify_validation() -> list[Path]:
    images = sorted(path for path in (VALID / 'images').iterdir()
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(images) != 249:
        raise RuntimeError(f'Expected 249 original MWPD validation images; found {len(images)}.')
    manifest = json.loads((ROOT / 'data/HRP4K_tiled_v1/metadata/files.json').read_text(encoding='utf-8'))
    expected = {row['destination']: row['sha256'] for row in manifest
                if row['destination'].startswith('valid/')}
    if len(expected) != 498:
        raise RuntimeError('The tiled dataset validation manifest is incomplete.')
    for image in images:
        label = VALID / 'labels' / (image.stem + '.txt')
        for path, key in ((image, 'valid/images/' + image.name),
                          (label, 'valid/labels/' + label.name)):
            if not path.is_file() or sha256(path) != expected.get(key):
                raise RuntimeError(f'Original validation SHA-256 differs: {path}')
    return images


def evaluate_model(name: str, weights: Path, images: list[Path], args, per_image: list[dict]) -> dict:
    row = dict.fromkeys(FIELDS, '')
    row.update(model=name, weights=str(weights), status='missing' if not weights.is_file() else 'running')
    if not weights.is_file():
        print(f'{name}: checkpoint missing, skipped: {weights}', flush=True)
        return row
    row['weights_sha256'] = sha256(weights)
    detector = None
    try:
        detector = AutoDetectionModel.from_pretrained(
            model_type='ultralytics', model_path=str(weights),
            confidence_threshold=args.conf, device=args.device, image_size=640)
        if len(detector.model.names) != 1:
            raise ValueError(f'{name}: expected one-class checkpoint, got {detector.model.names}')
        # One warm-up prediction, excluded from timing and metrics.
        from sahi.predict import get_prediction
        get_prediction(str(images[0]), detector)
        tp_parts, conf_parts, pred_class_parts, target_class_parts = [], [], [], []
        total_ms = 0.0
        prediction_count = 0
        ground_truth_count = 0
        for index, image in enumerate(images, 1):
            with Image.open(image) as opened:
                width, height = opened.size
            labels = ground_truth(VALID / 'labels' / (image.stem + '.txt'), width, height)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = get_sliced_prediction(
                str(image), detector, slice_height=args.tile_size, slice_width=args.tile_size,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                perform_standard_pred=True, postprocess_type='NMS',
                postprocess_match_metric='IOU', postprocess_match_threshold=0.50,
                auto_slice_resolution=False, verbose=0, batch_size=1)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            predictions = prediction_array(result)
            tp_parts.append(correct_matrix(labels, predictions))
            conf_parts.append(predictions[:, 4])
            pred_class_parts.append(np.zeros(len(predictions), dtype=np.int64))
            target_class_parts.append(np.zeros(len(labels), dtype=np.int64))
            prediction_count += len(predictions)
            ground_truth_count += len(labels)
            total_ms += elapsed_ms
            per_image.append(dict(model=name, image=image.name, gt_boxes=len(labels),
                                  predictions=len(predictions), elapsed_ms=round(elapsed_ms, 3)))
            if index % 25 == 0 or index == len(images):
                print(f'{name}: {index}/{len(images)} validation images', flush=True)
        tp = np.concatenate(tp_parts, axis=0)
        conf = np.concatenate(conf_parts)
        pred_cls = np.concatenate(pred_class_parts)
        target_cls = np.concatenate(target_class_parts)
        if not len(target_cls):
            raise RuntimeError('Validation has no ground-truth objects.')
        metrics = ap_per_class(tp, conf, pred_cls, target_cls, names={0: 'pothole'})
        precision, recall, ap = metrics[2], metrics[3], metrics[5]
        row.update(status='partial' if args.limit else 'ok', images=len(images),
                   ground_truth_boxes=ground_truth_count, predictions=prediction_count,
                   precision=float(precision[0]), recall=float(recall[0]),
                   map50=float(ap[0, 0]), map50_95=float(ap[0].mean()),
                   mean_ms_per_image=total_ms / len(images))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower():
            row.update(status='oom', error=f'{type(exc).__name__}: {exc}')
            torch.cuda.empty_cache()
        else:
            raise
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p2-weights', type=Path, default=DEFAULT_WEIGHTS['p2'])
    parser.add_argument('--models', nargs='+', choices=('n', 's', 'p2'), default=('n', 's', 'p2'))
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.20)
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Low detection threshold for full precision-recall/AP curve.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, help='Smoke test on first N images; marked partial.')
    args = parser.parse_args()
    if args.tile_size < 32 or not 0 <= args.overlap < 1 or not 0 < args.conf < 1:
        parser.error('Invalid tile size, overlap or confidence threshold.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    images = verify_validation()
    if args.limit is not None:
        images = images[:args.limit]
    weights = dict(DEFAULT_WEIGHTS, p2=args.p2_weights.resolve())
    output = ROOT / 'reports' / ('p2_sahi_eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT={output}', flush=True)
    per_image = []
    rows = []
    for name in dict.fromkeys(args.models):
        row = evaluate_model(name, weights[name], images, args, per_image)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'per_image.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('model', 'image', 'gt_boxes', 'predictions', 'elapsed_ms'))
        writer.writeheader()
        writer.writerows(per_image)
    (output / 'protocol.json').write_text(json.dumps(dict(
        validation=str(VALID), validation_images=len(images), test_used=False,
        partial=args.limit is not None, models=list(dict.fromkeys(args.models)),
        inference=dict(tile_size=args.tile_size, overlap=args.overlap, conf=args.conf,
                       device=args.device, batch_size=1, full_image_prediction=True,
                       merge='NMS IoU=0.50', image_size=640),
        metrics='Ultralytics ap_per_class: P/R at max-F1; AP at IoU .50:.95',
        note='Models share inference and validation. P2 uses HRP4K crops while existing n/s use MWPD training; this is not a controlled architecture-only experiment.'
    ), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
