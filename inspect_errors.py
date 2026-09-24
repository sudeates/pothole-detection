"""Draw missed (FN) and spurious (FP) boxes on validation images for manual review.

Usage:
    .\\venv\\Scripts\\python.exe .\\inspect_errors.py --weights runs\\<run>\\weights\\best.pt

Output: reports/errors_<run>_<time>/
    images/      one annotated image per validation image that has an FN or FP
    review.csv   one row per FN/FP; fill the `verdict` column by hand
    summary.md   FN counts per cause and size bucket

Colors: green = GT found (TP), red = GT missed (FN), orange = FP prediction,
thin gray = best low-confidence/misaligned prediction near a missed GT.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eval import BUCKETS, CONF, MATCH_IOU, ROOT, bucket, iou_matrix, local_path, match_at_fixed_conf, read_yolo

NEAR_IOU = 0.10
GREEN, RED, ORANGE, GRAY = (60, 200, 60), (40, 40, 230), (0, 150, 255), (170, 170, 170)


def fn_cause(gt_box: np.ndarray, pred: np.ndarray, confidence: np.ndarray) -> tuple[str, int | None]:
    """Why was this GT missed? Returns (cause, index of the most relevant prediction)."""
    if not len(pred):
        return 'hic_gormedi', None
    overlaps = iou_matrix(gt_box[None], pred)[0]
    overlapping = np.flatnonzero(overlaps >= MATCH_IOU)
    if len(overlapping):
        # Box is in the right place, only the score is below CONF (or it lost the match to another GT).
        best = int(overlapping[np.argmax(confidence[overlapping])])
        return ('dusuk_guven' if confidence[best] < CONF else 'eslesme_kaybi'), best
    confident = np.flatnonzero((confidence >= CONF) & (overlaps >= NEAR_IOU))
    if len(confident):
        # Model saw it but the box shape/extent disagrees with the label (boundary ambiguity).
        return 'kayik_kutu', int(confident[np.argmax(overlaps[confident])])
    return 'hic_gormedi', None


def to_pixels(box: np.ndarray, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x1, y1, x2, y2 = box * (width, height, width, height)
    return (int(x1), int(y1)), (int(x2), int(y2))


def label(image: np.ndarray, text: str, corner: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = corner[0], max(corner[1] - 4, 12)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--images', default='data/MWPD_reviewed_v1/valid/images')
    args = parser.parse_args()

    weights = local_path(args.weights)
    images_dir = local_path(args.images)
    labels_dir = images_dir.parent / 'labels'
    run_name = weights.parent.parent.name if weights.parent.name == 'weights' else weights.stem
    out_dir = ROOT / 'reports' / f'errors_{run_name}_{datetime.now():%Y%m%d_%H%M%S}'
    (out_dir / 'images').mkdir(parents=True)

    model = YOLO(str(weights))
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    rows, causes, bucket_causes = [], Counter(), Counter()
    total_gt = total_fp = 0

    for image_path in image_paths:
        gt, _ = read_yolo(labels_dir / f'{image_path.stem}.txt', prediction=False)
        result = model.predict(str(image_path), imgsz=640, conf=0.001, iou=0.7, max_det=300,
                               device=0, verbose=False)[0]
        pred = result.boxes.xyxyn.cpu().numpy().astype(np.float64).reshape(-1, 4)
        confidence = result.boxes.conf.cpu().numpy().astype(np.float64)
        eligible, matches, matched_pred = match_at_fixed_conf(gt, pred, confidence)
        matched_gt = {g for g, _, _ in matches}
        missed = [g for g in range(len(gt)) if g not in matched_gt]
        false_pos = [int(p) for p in eligible if int(p) not in matched_pred]
        total_gt += len(gt)
        total_fp += len(false_pos)
        if not missed and not false_pos:
            continue

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        for g, p, iou in matches:
            a, b = to_pixels(gt[g], width, height)
            cv2.rectangle(image, a, b, GREEN, 2)
            label(image, f'TP iou{iou:.2f}', a, GREEN)
        for g in missed:
            cause, near = fn_cause(gt[g], pred, confidence)
            size = BUCKETS[bucket(gt[g])]
            causes[cause] += 1
            bucket_causes[size, cause] += 1
            if near is not None:
                a, b = to_pixels(pred[near], width, height)
                cv2.rectangle(image, a, b, GRAY, 1)
                label(image, f'{confidence[near]:.2f}', (a[0], b[1] + 14), GRAY)
            a, b = to_pixels(gt[g], width, height)
            cv2.rectangle(image, a, b, RED, 3)
            label(image, f'FN {cause}', a, RED)
            near_iou = float(iou_matrix(gt[g][None], pred[near][None])[0, 0]) if near is not None else 0.0
            rows.append(dict(image=image_path.name, type='FN', cause=cause, size=size,
                             conf=f'{confidence[near]:.3f}' if near is not None else '',
                             iou=f'{near_iou:.2f}', verdict=''))
        for p in false_pos:
            a, b = to_pixels(pred[p], width, height)
            cv2.rectangle(image, a, b, ORANGE, 2)
            label(image, f'FP {confidence[p]:.2f}', a, ORANGE)
            rows.append(dict(image=image_path.name, type='FP', cause='', size=BUCKETS[bucket(pred[p])],
                             conf=f'{confidence[p]:.3f}', iou='', verdict=''))
        prefix = f'{len(missed):02d}fn_{len(false_pos):02d}fp'
        cv2.imwrite(str(out_dir / 'images' / f'{prefix}_{image_path.stem}.jpg'), image)

    with (out_dir / 'review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['image'])
        writer.writeheader()
        writer.writerows(rows)

    total_fn = sum(causes.values())
    lines = [f'# Hata incelemesi: {run_name}', '',
             f'conf={CONF}, eşleşme IoU={MATCH_IOU}. GT={total_gt}, FN={total_fn}, FP={total_fp}.', '',
             '| FN nedeni | adet | oran |', '|---|---|---|']
    for cause, count in causes.most_common():
        lines.append(f'| {cause} | {count} | {count / max(total_fn, 1):.0%} |')
    lines += ['', '| boyut | ' + ' | '.join(sorted(causes)) + ' |', '|---' * (len(causes) + 1) + '|']
    for size in BUCKETS:
        lines.append(f'| {size} | ' + ' | '.join(str(bucket_causes[size, c]) for c in sorted(causes)) + ' |')
    lines += ['', 'Nedenler: `hic_gormedi` = conf>=0.001 dahil hiçbir kutu IoU>=0.5 ile örtüşmüyor ve '
              f'conf>={CONF} kutu IoU>={NEAR_IOU} ile de yakın değil; `dusuk_guven` = doğru yerde kutu var '
              f'ama conf<{CONF}; `kayik_kutu` = model gördü ama kutu etiketle IoU<0.5 (sınır belirsizliği); '
              '`eslesme_kaybi` = örtüşen kutu başka GT ile eşleşti.', '',
             'review.csv `verdict` sütunu önerisi: `net` (bariz çukur, model kaçırdı), `belirsiz` '
             '(çatlak/yama/tartışmalı), `etiket_hatasi` (etiket yanlış veya eksik; FP için gerçekte çukur var).']
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:6 + len(causes)]))
    print(f'\nÇıktı: {out_dir}')


if __name__ == '__main__':
    main()
