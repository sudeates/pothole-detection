"""Final test evaluation with fixed settings; no tuning or training."""
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT/'runs/reviewed-v1-baseline_20260915_223719_077806/weights/best.pt'
DATA = ROOT/'data.local.yaml'


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS
    import ultralytics.data.dataset as dataset_module
    import yaml
    for path in (WEIGHTS,DATA):
        if not path.is_file(): raise FileNotFoundError(path)
    config=yaml.safe_load(DATA.read_text(encoding='utf-8'))
    source=Path(config['path'])/config['test']
    images=sorted(p for p in source.rglob('*') if p.suffix.lower().lstrip('.') in IMG_FORMATS)
    if not images: raise FileNotFoundError(f'No test images: {source}')
    labels=[source.parent/'labels'/p.relative_to(source).with_suffix('.txt') for p in images]
    missing=[str(p) for p in labels if not p.is_file()]
    if missing:raise FileNotFoundError(f'Missing test labels: {missing}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    output=ROOT/'reports'/('reviewed_final_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    model_hash=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    settings=dict(split='test',imgsz=640,batch=1,workers=0,device=0,half=False,
        conf=.001,iou=.7,max_det=300,augment=False,rect=True,seed=42,
        deterministic=True,cache=False,plots=True)
    metadata=dict(weights=str(WEIGHTS),weights_sha256=model_hash,data=str(DATA),
        settings=settings,images=len(images),annotation_rows=sum(len(p.read_text().splitlines()) for p in labels),
        python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0),test_used_for_tuning=False,
        timing='Ultralytics model inference ms/image, excluding preprocessing and postprocessing')
    (output/'environment.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    def memory_only_cache(prefix,path,cache,version):cache['version']=version
    dataset_module.save_dataset_cache_file=memory_only_cache
    print(f'OUTPUT_DIR={output}',flush=True)
    try:
        with (output/'run.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            metrics=YOLO(str(WEIGHTS)).val(data=str(DATA),**settings,
                project=str(output.parent),name=output.name,exist_ok=True)
        result=dict(status='ok',split='test',imgsz=640,batch=1,workers=0,
            precision=float(metrics.box.mp),recall=float(metrics.box.mr),
            mAP50=float(metrics.box.map50),mAP50_95=float(metrics.box.map),
            inference_ms_per_image=float(metrics.speed['inference']))
        (output/'speed.json').write_text(json.dumps(metrics.speed,indent=2),encoding='utf-8')
        if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=model_hash:
            raise RuntimeError('Model changed during evaluation.')
    except Exception as exc:
        (output/'FAILED.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'),ensure_ascii=False),encoding='utf-8')
        raise
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(result));writer.writeheader();writer.writerow(result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
