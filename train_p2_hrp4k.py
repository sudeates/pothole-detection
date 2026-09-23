"""Preflight a local YOLOv8n-P2 run; add --start to train in the current terminal."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import torch
import ultralytics
from ultralytics import YOLO
import yaml


ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / 'experiments/p2_hrp4k_v1.yaml'
RUN_NAME = 'p2-hrp4k-v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(dataset: Path, original_validation: Path) -> dict:
    if (dataset / 'INCOMPLETE').exists():
        raise RuntimeError('Tiled dataset is marked INCOMPLETE.')
    verified_path = dataset / 'VERIFIED.json'
    verified = json.loads(verified_path.read_text(encoding='utf-8'))
    if verified.get('status') != 'verified' or not verified.get('source_unchanged'):
        raise RuntimeError('Tiled dataset verification status is invalid.')
    rows = json.loads((dataset / 'metadata/files.json').read_text(encoding='utf-8'))
    if len(rows) != verified['output_files_verified']:
        raise RuntimeError('SHA-256 manifest count differs from VERIFIED.json.')
    seen = set()
    valid_rows = 0
    for row in rows:
        relative = Path(row['destination'])
        target = (dataset / relative).resolve()
        if not target.is_relative_to(dataset) or target in seen or not target.is_file():
            raise RuntimeError(f'Unexpected/missing output file: {target}')
        seen.add(target)
        if sha256(target) != row['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {target}')
        if relative.parts[0] == 'valid':
            original = original_validation / relative.relative_to('valid')
            if not original.is_file() or sha256(original) != row['sha256']:
                raise RuntimeError(f'Original validation differs: {original}')
            valid_rows += 1
    actual = {p.resolve() for split in ('train', 'valid') for kind in ('images', 'labels')
              for p in (dataset / split / kind).iterdir() if p.is_file()}
    if actual != seen or valid_rows != 498:
        raise RuntimeError('Dataset has added/missing images or validation is incomplete.')
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Start training after preflight.')
    parser.add_argument('--batch', type=int, default=None, help='Override batch size, e.g. 1 after OOM.')
    parser.add_argument('--run-name', default=RUN_NAME, help='Unique runs/ directory name.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('--run-name may contain only letters, digits, hyphens and underscores.')
    recipe = yaml.safe_load(RECIPE.read_text(encoding='utf-8'))
    if args.batch is not None:
        if args.batch < 1:
            parser.error('--batch must be positive.')
        recipe['batch'] = args.batch
    architecture = ROOT / recipe.pop('model')
    initial = ROOT / recipe.pop('initial_weights')
    data_yaml = ROOT / recipe.pop('data')
    for path in (architecture, initial, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(f'Required local file is missing: {path}')
    if (recipe['imgsz'], recipe['patience'], recipe['close_mosaic'], recipe['workers']) != (640, 15, 5, 0):
        raise RuntimeError('P2 recipe no longer matches the approved training settings.')
    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    dataset = Path(data['path']).resolve()
    if (dataset != (ROOT / 'data/HRP4K_tiled_v1').resolve() or
            data.get('train') != 'train/images' or data.get('val') != 'valid/images' or
            data.get('names') != {0: 'pothole'} or 'test' in data):
        raise RuntimeError('Unexpected data.yaml: training/validation split is not approved.')
    verified = verify_dataset(dataset, ROOT / 'data/MWPD_reviewed_v1/valid')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable.')
    model = YOLO(str(architecture)).load(str(initial))
    strides = [int(value) for value in model.model.stride.tolist()]
    if strides != [4, 8, 16, 32] or model.model.yaml['nc'] != 1:
        raise RuntimeError(f'Expected one-class P2/P3/P4/P5 detector; got strides={strides}.')
    destination = ROOT / 'runs' / args.run_name
    if args.start and destination.exists():
        raise FileExistsError(f'Refusing to overwrite training run: {destination}')
    options = dict(recipe, data=str(data_yaml), project=str(ROOT / 'runs'),
                   name=args.run_name, exist_ok=False, resume=False)
    report = dict(created=datetime.now().isoformat(), training_started=False,
                  architecture=str(architecture), initial_weights=str(initial),
                  dataset=str(dataset), validation=str(dataset / 'valid/images'),
                  output=str(destination), strides=strides,
                  parameters=sum(p.numel() for p in model.model.parameters()),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  ultralytics=ultralytics.__version__, verified_files=verified['output_files_verified'],
                  sha256={str(path): sha256(path) for path in
                          (RECIPE, architecture, initial, data_yaml, dataset / 'metadata/files.json')},
                  options=options)
    report_dir = ROOT / 'reports' / ('p2_preflight_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Preflight passed: {report_dir}', flush=True)
    print(f'P2 strides: {strides}; parameters: {report["parameters"]:,}; batch: {recipe["batch"]}', flush=True)
    if not args.start:
        print('Training not started. Add --start in VS Code when ready.', flush=True)
        return
    # Keep the verified dataset unchanged; Ultralytics may otherwise write .cache files.
    import ultralytics.data.dataset as dataset_module
    def memory_only_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_only_cache
    (report_dir / 'STARTED.json').write_text(json.dumps(dict(started=datetime.now().isoformat(),
        output=str(destination))), encoding='utf-8')
    try:
        model.train(**options)
        (report_dir / 'COMPLETED.json').write_text(json.dumps(dict(completed=datetime.now().isoformat(),
            output=str(destination))), encoding='utf-8')
    except Exception as exc:
        (report_dir / 'FAILED.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'),
            ensure_ascii=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
