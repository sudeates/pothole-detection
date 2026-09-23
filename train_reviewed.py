"""Validate the reviewed-data experiment; training requires --start explicitly."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Start the prepared 100-epoch experiment.')
    parser.add_argument('--patience', type=int, default=15,
                        help='Stop after this many epochs without validation fitness improvement (default: 15).')
    parser.add_argument('--recipe', choices=['baseline','yolov8s','v2-baseline','v3-aug'], default='baseline',
                        help='Select the prepared experiment, keeping baseline behavior by default.')
    args = parser.parse_args()
    if args.patience < 1:
        parser.error('--patience must be at least 1.')
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    version = 'v2' if args.recipe == 'v2-baseline' else ('v3' if args.recipe == 'v3-aug' else 'v1')
    architecture = 'baseline' if version in ('v2','v3') else args.recipe
    recipe_path = ROOT / f'experiments/reviewed_{version}_{architecture}.yaml'
    recipe = yaml.safe_load(recipe_path.read_text(encoding='utf-8'))
    if version in ('v2','v3'):
        baseline = yaml.safe_load((ROOT/'experiments/reviewed_v1_baseline.yaml').read_text(encoding='utf-8'))
        expected_recipe = dict(baseline, data=('data.reviewed-v2.yaml' if version == 'v2' else 'data.reviewed-v3-aug.yaml'))
        if recipe != expected_recipe:
            raise RuntimeError('V2 must keep the V1 baseline training settings; only data may differ.')
    weights = ROOT / recipe.pop('model')
    data_path = ROOT / recipe.pop('data')
    if not weights.is_file() or not data_path.is_file():
        raise FileNotFoundError('Local model or dataset YAML missing; no download attempted.')
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    dataset = Path(data['path']).resolve()
    expected = (ROOT / ('data/MWPD_reviewed_v2' if version == 'v2' else 'data/MWPD_reviewed_v3_aug' if version == 'v3' else 'data/MWPD_reviewed_v1')).resolve()
    if (dataset != expected or 'test' in data or data.get('val') != 'valid/images'
            or data.get('train') != 'train/images' or data.get('names') != {0:'pothole'}):
        raise RuntimeError('Unexpected dataset/split settings.')
    if (dataset/'INCOMPLETE').exists() or not (dataset/'VERIFIED.json').is_file():
        raise RuntimeError('Dataset copy is not verified.')
    manifest = json.loads((dataset/'metadata/files.json').read_text(encoding='utf-8'))
    files = set()
    for row in manifest:
        path = (dataset/row['destination']).resolve()
        if not path.is_relative_to(dataset) or not path.is_file() or sha256(path) != row['sha256']:
            raise RuntimeError(f'Dataset differs from its verified manifest: {path}')
        files.add(path)
    splits = ('train','valid','review_hold','train_review_hold') if version in ('v2','v3') else ('train','valid','review_hold')
    actual = {p.resolve() for split in splits for sub in ('images','labels')
              for p in (dataset/split/sub).iterdir() if p.is_file()}
    if actual != files:
        raise RuntimeError('Dataset has added or missing files.')
    if version == 'v2':
        # Same validation content as V1; hold directories cannot enter training.
        baseline_manifest = json.loads((ROOT/'data/MWPD_reviewed_v1/metadata/files.json').read_text(encoding='utf-8'))
        def validation_hashes(rows):
            return {Path(r['destination']).as_posix():r['sha256'] for r in rows
                    if Path(r['destination']).parts[0]=='valid'}
        if validation_hashes(manifest) != validation_hashes(baseline_manifest):
            raise RuntimeError('V2 validation is not identical to V1.')
        counts = {split:len(list((dataset/split/'images').iterdir())) for split in splits}
        if counts != {'train':2103,'valid':249,'review_hold':11,'train_review_hold':46}:
            raise RuntimeError(f'Unexpected V2 split counts: {counts}')
    if version == 'v3':
        counts = {split:len(list((dataset/split/'images').iterdir())) for split in splits}
        if counts != {'train':4206,'valid':249,'review_hold':11,'train_review_hold':46}:
            raise RuntimeError(f'Unexpected V3 split counts: {counts}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    model = YOLO(str(weights))
    checkpoint_data = str(model.ckpt.get('train_args', {}).get('data', ''))
    if len(model.names) != 80 or Path(checkpoint_data).name != 'coco.yaml':
        raise RuntimeError('Expected local 80-class COCO initial weights.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    recipe['patience'] = args.patience
    options = dict(recipe, data=str(data_path), project=str(ROOT/'runs'),
                   name=f'reviewed-{version}-{architecture}_'+stamp, exist_ok=False, resume=False)
    get_cfg(overrides=dict(model=str(weights), **options))
    report = dict(status='preflight_passed',training_started=False,dataset_version=version,
        python=sys.version, executable=sys.executable,torch=torch.__version__,
        ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),
        initial_weights=str(weights),weights_sha256=sha256(weights),checkpoint_data=checkpoint_data,
        recipe_sha256=sha256(recipe_path),data_yaml_sha256=sha256(data_path),
        manifest_sha256=sha256(dataset/'metadata/files.json'),verified_files=len(files),
        train_images=len(list((dataset/'train/images').iterdir())),
        validation_images=len(list((dataset/'valid/images').iterdir())),
        gpu_training_memory_tested=False,options=options)
    report_dir = ROOT/'reports'/('reviewed_preflight_'+stamp)
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Preflight passed. Report: {report_dir}',flush=True)
    print(f'Early stopping patience: {options["patience"]} epochs.',flush=True)
    if not args.start:
        print('Training NOT started. --start is required.',flush=True)
        return
    # Keep snapshot labels unchanged; Ultralytics can rebuild cache in memory.
    import ultralytics.data.dataset as dataset_module
    def memory_only_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_only_cache
    (report_dir/'STARTED.json').write_text(json.dumps(dict(started=datetime.now().isoformat(),
        output=str(Path(options['project'])/options['name']))),encoding='utf-8')
    try:
        model.train(**options)
        (report_dir/'COMPLETED.json').write_text(json.dumps(dict(completed=datetime.now().isoformat(),
            output=str(Path(options['project'])/options['name']))),encoding='utf-8')
    except Exception as exc:
        (report_dir/'FAILED.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'),
            ensure_ascii=False,indent=2),encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
