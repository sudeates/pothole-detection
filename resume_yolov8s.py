"""Resume the interrupted YOLOv8s experiment in the visible terminal."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'runs/reviewed-v1-yolov8s_20260916_122325_277730/weights/last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Resume training; otherwise inspect only.')
    args = parser.parse_args()
    import torch
    import yaml
    from ultralytics import YOLO
    import ultralytics.data.dataset as dataset_module

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    model = YOLO(str(CHECKPOINT))
    checkpoint = model.ckpt
    epoch = checkpoint.get('epoch', -1)
    train_args = checkpoint.get('train_args', {})
    target = train_args.get('epochs', 100)
    if epoch < 0 or checkpoint.get('optimizer') is None or epoch + 1 >= target:
        raise RuntimeError('Checkpoint cannot resume an unfinished training run.')
    data_path = Path(train_args['data']).resolve()
    if data_path != (ROOT/'data.reviewed-v1.yaml').resolve():
        raise RuntimeError('Unexpected dataset in checkpoint.')
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if 'test' in data or Path(data['path']).resolve() != (ROOT/'data/MWPD_reviewed_v1').resolve():
        raise RuntimeError('Dataset configuration changed.')
    print(f'Checkpoint: {CHECKPOINT}', flush=True)
    print(f'Completed epochs: {epoch + 1}; target: {target}; next epoch: {epoch + 2}', flush=True)
    if not args.start:
        print('Inspection only. Use --start to resume.', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment.')
    def memory_only_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_only_cache
    model.train(resume=True, workers=0, device=0)


if __name__ == '__main__':
    main()
