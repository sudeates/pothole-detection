"""En iyi modeli ayrılmış test bölümünde değerlendir ve tahminleri kaydet."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    import torch
    import yaml
    from ultralytics import YOLO

    weights = ROOT / 'runs/baseline-3/weights/best.pt'
    config = ROOT / 'data.local.yaml'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA kullanılamıyor; GPU kontrolünü geçen venv ile çalıştırın.')
    data = yaml.safe_load(config.read_text(encoding='utf-8'))
    source = Path(data['path']) / data['test']
    if not source.is_dir():
        raise FileNotFoundError(source)
    model = YOLO(str(weights))
    metrics = model.val(data=str(config), split='test', imgsz=640,
                        batch=4, device=0, workers=0, plots=True,
                        project=str(ROOT / 'reports'), name='test', exist_ok=False)
    report_dir = Path(metrics.save_dir)
    summary = {str(k): float(v) for k, v in metrics.results_dict.items()}
    summary['weights'] = str(weights)
    summary['split'] = 'test'
    (report_dir / 'metrics.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    # Tahmin eşiği görselleştirme içindir; değerlendirme varsayılanlarını değiştirmez.
    prediction_dir = None
    for result in model.predict(source=str(source), imgsz=640, device=0,
                                conf=0.25, save=True, save_txt=True,
                                save_conf=True, stream=True,
                                project=str(report_dir), name='predictions',
                                exist_ok=False):
        prediction_dir = result.save_dir
    print('\nTEST SONUÇLARI:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Rapor klasörü: {report_dir}')
    print(f'Kutulu görüntüler: {prediction_dir}')

if __name__ == '__main__':
    main()
