"""Generic local YOLO training: train.py --config experiments/name.yaml [--start]."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import IMG_FORMATS
import yaml


ROOT = Path(__file__).resolve().parent


def local_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_split(dataset: Path, images_relative: str, classes: int) -> dict:
    images_dir = dataset / images_relative
    labels_dir = images_dir.parent / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Images or labels folder missing: {images_dir}, {labels_dir}')
    images = sorted(p for p in images_dir.rglob('*')
                    if p.is_file() and p.suffix.lower().lstrip('.') in IMG_FORMATS)
    if not images:
        raise RuntimeError(f'No images in {images_dir}')
    labels = []
    box_count = empty_count = 0
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix('.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing YOLO label: {label}')
        labels.append(label.resolve())
        rows = [row for row in label.read_text(encoding='utf-8').splitlines() if row.strip()]
        empty_count += not rows
        for line_number, row in enumerate(rows, 1):
            fields = row.split()
            if len(fields) != 5:
                raise ValueError(f'Expected five YOLO fields: {label}:{line_number}')
            cls = int(fields[0])
            coords = [float(value) for value in fields[1:]]
            if (not 0 <= cls < classes or any(not 0 <= value <= 1 for value in coords)
                    or coords[2] <= 0 or coords[3] <= 0):
                raise ValueError(f'Invalid YOLO annotation: {label}:{line_number}')
            box_count += 1
    actual_labels = {p.resolve() for p in labels_dir.rglob('*.txt') if p.is_file()}
    if actual_labels != set(labels):
        raise RuntimeError(f'Orphan labels in {labels_dir}')
    return dict(images=len(images), labels=len(labels), boxes=box_count, empty_images=empty_count)


def verify_existing_manifest(dataset: Path) -> dict | None:
    verified_path = dataset / 'VERIFIED.json'
    manifest_path = dataset / 'metadata/files.json'
    if (dataset / 'INCOMPLETE').exists():
        raise RuntimeError(f'Dataset is marked INCOMPLETE: {dataset}')
    if not verified_path.is_file() and not manifest_path.is_file():
        return None
    if not verified_path.is_file() or not manifest_path.is_file():
        raise RuntimeError('Dataset SHA-256 verification files are incomplete.')
    rows = json.loads(manifest_path.read_text(encoding='utf-8'))
    verified = json.loads(verified_path.read_text(encoding='utf-8'))
    if verified.get('status') not in ('copy_integrity_verified', 'verified'):
        raise RuntimeError('Dataset verification status is invalid.')
    if len(rows) != verified.get('files', verified.get('output_files_verified')):
        raise RuntimeError('Dataset verification count differs from manifest.')
    seen = set()
    for row in rows:
        target = (dataset / row['destination']).resolve()
        if not target.is_relative_to(dataset) or target in seen or not target.is_file():
            raise RuntimeError(f'Unexpected or missing manifest file: {target}')
        seen.add(target)
        if sha256(target) != row['sha256']:
            raise RuntimeError(f'Dataset SHA-256 differs: {target}')
    tracked_dirs = [dataset / name / kind for name in
                    ('train', 'valid', 'review_hold', 'train_review_hold')
                    for kind in ('images', 'labels') if (dataset / name / kind).is_dir()]
    actual = {p.resolve() for folder in tracked_dirs for p in folder.rglob('*') if p.is_file()}
    if actual != seen:
        raise RuntimeError('Dataset has added or missing image/label files.')
    return dict(files=len(rows), manifest_sha256=sha256(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Experiment recipe YAML.')
    parser.add_argument('--start', action='store_true', help='Start training after preflight.')
    args = parser.parse_args()
    config_path = local_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    recipe = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(recipe, dict):
        raise ValueError('Experiment YAML must be a mapping.')
    model_path = local_path(recipe.pop('model'))
    data_path = local_path(recipe.pop('data'))
    initial_path = local_path(recipe.pop('initial_weights')) if 'initial_weights' in recipe else None
    for path in (model_path, data_path, initial_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f'Required local file missing: {path}')
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not {'path', 'train', 'val', 'names'} <= data.keys():
        raise ValueError('data.yaml needs path, train, val and names.')
    dataset = local_path(data['path'])
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    names = data['names']
    classes = len(names)
    if not classes:
        raise ValueError('No classes in data.yaml.')
    splits = {split: check_split(dataset, data[split], classes) for split in ('train', 'val')}
    integrity = verify_existing_manifest(dataset)
    if isinstance(recipe.get('device'), int) and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for the selected device.')
    model = YOLO(str(model_path))
    if initial_path is not None:
        model.load(str(initial_path))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = f'{config_path.stem}_{stamp}'
    options = dict(recipe, data=str(data_path), project=str(ROOT / 'runs'),
                   name=run_name, exist_ok=False, resume=False)
    get_cfg(overrides=dict(model=str(model_path), **options))
    output = ROOT / 'runs' / run_name
    if output.exists():
        raise FileExistsError(output)
    report_dir = ROOT / 'reports' / f'preflight_{run_name}'
    report_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='preflight_passed', training_started=False,
                  config=str(config_path), model=str(model_path),
                  initial_weights=str(initial_path) if initial_path else None,
                  data=str(data_path), dataset=str(dataset), splits=splits,
                  integrity=integrity, run=str(output), options=options,
                  python=sys.version, torch=torch.__version__,
                  ultralytics=ultralytics.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  sha256={str(path): sha256(path) for path in
                          (config_path, model_path, data_path, initial_path) if path is not None})
    (report_dir / 'preflight.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Preflight passed: {report_dir}', flush=True)
    print(f'Train: {splits["train"]}; val: {splits["val"]}', flush=True)
    print(f'Run: {output}', flush=True)
    if not args.start:
        print('Training not started. Add --start to train.', flush=True)
        return
    # Keep existing verified data snapshots unchanged.
    import ultralytics.data.dataset as dataset_module
    def memory_only_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_only_cache
    (report_dir / 'STARTED.json').write_text(
        json.dumps(dict(started=datetime.now().isoformat(), run=str(output))), encoding='utf-8')
    try:
        model.train(**options)
        (report_dir / 'COMPLETED.json').write_text(
            json.dumps(dict(completed=datetime.now().isoformat(), run=str(output))), encoding='utf-8')
    except Exception as exc:
        (report_dir / 'FAILED.json').write_text(
            json.dumps(dict(error=f'{type(exc).__name__}: {exc}'), ensure_ascii=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
