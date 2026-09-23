"""Compare three SAHI window sizes and four score thresholds on MWPD validation."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
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
from sahi.predict import get_prediction, get_sliced_prediction
from analyze_validation import match

WEIGHTS = ROOT / 'runs/reviewed-v3-baseline_20260916_222621_316060/weights/best.pt'
VALID = ROOT / 'data/MWPD_reviewed_v1/valid'
WINDOW_RATIOS = (0.50, 0.70, 0.90)
SCORE_THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
FIELDS = ('method', 'window_ratio', 'confidence', 'images', 'tp', 'fp', 'fn',
          'precision', 'recall', 'f1', 'mean_ms_per_image', 'status')


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_gt(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in path.read_text(encoding='utf-8').splitlines():
        cls, x, y, bw, bh = map(float, line.split())
        if cls != 0:
            raise ValueError(f'Unexpected class in {path}')
        boxes.append([(x - bw / 2) * width, (y - bh / 2) * height,
                      (x + bw / 2) * width, (y + bh / 2) * height])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def result_boxes(result) -> np.ndarray:
    return np.asarray([
        prediction.bbox.to_xyxy() + [prediction.score.value, prediction.category.id]
        for prediction in result.object_prediction_list
    ], dtype=np.float64).reshape(-1, 6)


def main() -> None:
    images = sorted((VALID / 'images').glob('*.jpg'))
    if len(images) != 249 or not WEIGHTS.is_file():
        raise RuntimeError('Expected 249 validation images and the existing V3 best.pt')
    source_files = [WEIGHTS] + [p for sub in ('images', 'labels')
                              for p in (VALID / sub).iterdir() if p.is_file()]
    source_hashes = {str(path): sha256(path) for path in source_files}
    output = ROOT / 'reports' / ('sahi_sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT={output}', flush=True)

    detector = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', model_path=str(WEIGHTS),
        confidence_threshold=min(SCORE_THRESHOLDS), device='cuda:0', image_size=640)
    get_prediction(str(images[0]), detector)  # warm-up, excluded from timing
    accum = defaultdict(lambda: dict(images=0, tp=0, fp=0, fn=0, total_ms=0.0, status='ok'))
    per_image = []

    for image_index, image in enumerate(images, 1):
        with Image.open(image) as opened:
            width, height = opened.size
        gt = load_gt(VALID / 'labels' / f'{image.stem}.txt', width, height)
        for method, ratio in [('standard', 0.0)] + [('sahi', value) for value in WINDOW_RATIOS]:
            key = (method, ratio)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                if method == 'standard':
                    result = get_prediction(str(image), detector)
                else:
                    side = max(128, round(max(width, height) * ratio))
                    result = get_sliced_prediction(
                        str(image), detector,
                        slice_height=side, slice_width=side,
                        overlap_height_ratio=0.20, overlap_width_ratio=0.20,
                        perform_standard_pred=True,
                        postprocess_type='NMS', postprocess_match_metric='IOU',
                        postprocess_match_threshold=0.50,
                        auto_slice_resolution=False, verbose=0, batch_size=1)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                predictions = result_boxes(result)
                for confidence in SCORE_THRESHOLDS:
                    hits, matched_predictions, selected_predictions = match(gt, predictions, confidence)
                    counts = accum[(method, ratio, confidence)]
                    counts['images'] += 1
                    counts['tp'] += len(hits)
                    counts['fp'] += len(selected_predictions) - len(matched_predictions)
                    counts['fn'] += len(gt) - len(hits)
                    counts['total_ms'] += elapsed_ms
                    per_image.append(dict(image=image.name, method=method, window_ratio=ratio,
                                          confidence=confidence, tp=len(hits),
                                          fp=len(selected_predictions) - len(matched_predictions),
                                          fn=len(gt) - len(hits), elapsed_ms=elapsed_ms))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                for confidence in SCORE_THRESHOLDS:
                    accum[(method, ratio, confidence)]['status'] = 'oom'
                per_image.append(dict(image=image.name, method=method, window_ratio=ratio,
                                      status='oom', error=str(exc)))
        if image_index % 25 == 0 or image_index == len(images):
            print(f'{image_index}/{len(images)}', flush=True)

    rows = []
    for (method, ratio, confidence), value in sorted(accum.items()):
        tp, fp, fn = value['tp'], value['fp'], value['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(dict(method=method, window_ratio=ratio, confidence=confidence,
                         images=value['images'], tp=tp, fp=fp, fn=fn,
                         precision=precision, recall=recall,
                         f1=2 * precision * recall / max(precision + recall, 1e-9),
                         mean_ms_per_image=value['total_ms'] / max(value['images'], 1),
                         status=value['status']))
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'per_image.json').write_text(json.dumps(per_image, ensure_ascii=False), encoding='utf-8')
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError('Input file changed during comparison')
    (output / 'protocol.json').write_text(json.dumps(dict(
        validation_images=len(images), ground_truth_objects=sum(
            len((VALID / 'labels' / f'{image.stem}.txt').read_text().splitlines()) for image in images),
        weights=str(WEIGHTS), weights_sha256=source_hashes[str(WEIGHTS)],
        model_input_size=640, window_ratios=WINDOW_RATIOS, confidences=SCORE_THRESHOLDS,
        overlap=0.20, merge='NMS IOU=0.50', standard_prediction_in_sahi=True,
        diagnostic_match_iou=0.50, timing='end-to-end ms/image including image read and SAHI merge',
        input_files_unchanged=True, test_used=False, training_started=False,
        note='Fixed-threshold diagnostic counts. Precision here is TP/(TP+unmatched predictions), not Ultralytics max-F1 precision or mAP.'
    ), indent=2), encoding='utf-8')
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
