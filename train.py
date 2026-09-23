"""Yerel MWPD verisini kopyalayarak kısa veya tam GPU eğitimi başlatır."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    import torch
    import yaml
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Bu Python ortamında CUDA kullanılamıyor.')
    source = (Path.home() / '.cache/kagglehub/datasets/jocelyndumlao/'
              'multi-weather-pothole-detection-mwpd/versions/1/'
              'Multi-Weather Pothole Detection (MWPD)/MWPD')
    destination = ROOT / 'data' / 'MWPD'
    # Yarım kalmış kopya ile eğitim başlatma; ham veriye dokunma.
    marker = destination / '.copy_complete'
    if not marker.exists():
        if destination.exists():
            raise RuntimeError(f'Tamamlanmamış veri kopyasını kontrol edin: {destination}')
        shutil.copytree(source, destination)
        marker.touch()
    config = {
        'path': destination.as_posix(),
        'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
        'names': {0: 'pothole'},
    }
    config_path = ROOT / 'data.local.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    model = YOLO('yolov8n.pt')
    model.train(data=str(config_path), epochs=args.epochs, batch=args.batch,
                imgsz=640, device=0, workers=0, seed=42, deterministic=True,
                project=str(ROOT / 'runs'),
                name='smoke' if args.epochs == 1 else 'baseline', exist_ok=False)

if __name__ == '__main__':
    main()
