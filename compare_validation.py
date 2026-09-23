"""Mevcut ağırlıkları yalnızca validation üzerinde karşılaştırır; eğitim yapmaz."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import traceback

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs/baseline-3/weights/best.pt'
DATA = ROOT / 'data.local.yaml'
FIELDS = ['imgsz', 'split', 'batch', 'workers', 'status', 'precision', 'recall',
          'mAP50', 'mAP50_95', 'inference_ms_per_image', 'error', 'output_dir']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def worker(size, directory):
    row = dict(imgsz=size, split='val', batch=1, workers=0, status='error',
               output_dir=str(directory), error='')
    try:
        import torch
        from ultralytics import YOLO
        # Etiket önbelleğini veri dizinine yazma; mevcut veriyi koru.
        import ultralytics.data.dataset as dataset_module
        def memory_only_cache(prefix, path, cache, version):
            cache['version'] = version
        dataset_module.save_dataset_cache_file = memory_only_cache
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA kullanılamıyor.')
        model = YOLO(str(WEIGHTS))
        metrics = model.val(data=str(DATA), split='val', imgsz=size, batch=1,
                            workers=0, device=0, half=False, conf=0.001,
                            iou=0.7, max_det=300, augment=False, rect=True,
                            seed=42, deterministic=True, cache=False,
                            plots=True, project=str(directory.parent),
                            name=directory.name, exist_ok=True)
        row.update(status='ok', precision=float(metrics.box.mp),
                   recall=float(metrics.box.mr), mAP50=float(metrics.box.map50),
                   mAP50_95=float(metrics.box.map),
                   inference_ms_per_image=float(metrics.speed['inference']))
        write_json(directory / 'speed.json', metrics.speed)
    except Exception as exc:
        message = str(exc)
        row['status'] = 'oom' if isinstance(exc, MemoryError) or 'out of memory' in message.lower() else 'error'
        row['error'] = f'{type(exc).__name__}: {message}'
        traceback.print_exc()
    write_json(directory / 'result.json', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[640, 960, 1280], help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return
    import torch
    import ultralytics
    import yaml
    for path in (WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(f'Eksik dosya: {path}')
    config = yaml.safe_load(DATA.read_text(encoding='utf-8'))
    data_root = Path(config['path'])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    validation = data_root / config['val']
    labels = validation.parent / 'labels'
    if not validation.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f'Validation görüntü/etiket yolu eksik: {validation}, {labels}')
    from ultralytics.data.utils import IMG_FORMATS
    images = sorted(p for p in validation.rglob('*') if p.suffix.lower().lstrip('.') in IMG_FORMATS)
    if not images:
        raise RuntimeError('Validation görüntüsü bulunamadı.')
    missing = [str(labels / p.relative_to(validation).with_suffix('.txt')) for p in images
               if not (labels / p.relative_to(validation).with_suffix('.txt')).is_file()]
    output = ROOT / 'reports' / ('val_resolution_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'environment.json', dict(python=sys.version, executable=sys.executable,
        torch=torch.__version__, ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        weights=str(WEIGHTS), weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        data=str(DATA), data_yaml=config, validation_images=len(images), missing_labels=missing,
        half=False, conf=0.001, iou=0.7, max_det=300, rect=True,
        timing='Ultralytics inference ms/image; preprocessing, NMS and disk IO excluded'))
    print(f'OUTPUT_DIR={output}', flush=True)
    if missing:
        raise FileNotFoundError(f'{len(missing)} validation etiketi eksik; environment.json dosyasına bakın.')
    rows = []
    for size in (640, 960, 1280):
        directory = output / f'imgsz_{size}'
        directory.mkdir()
        print(f'Başlıyor: imgsz={size}', flush=True)
        with (directory / 'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--worker', str(size), '--output', str(directory)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
        result = directory / 'result.json'
        if result.exists():
            row = json.loads(result.read_text(encoding='utf-8'))
        else:
            log_text = (directory / 'run.log').read_text(encoding='utf-8', errors='replace')
            row = dict(imgsz=size, split='val', batch=1, workers=0,
                status='oom' if 'out of memory' in log_text.lower() else 'error',
                error=f'İşlem sonuç üretemedi, exit={process.returncode}; run.log dosyasına bakın.',
                output_dir=str(directory))
            write_json(result, row)
        rows.append(row)
        with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f'CSV: {output / "comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
