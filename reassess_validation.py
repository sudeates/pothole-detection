"""Bounded, validation-only model and inference ablation; never trains."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import gc
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import ultralytics.data.dataset as dataset_module
from PIL import Image, ImageDraw
from analyze_validation import match

ROOT = Path(__file__).resolve().parent
MODELS = {
    'original': 'baseline-3',
    'v1': 'reviewed-v1-baseline_20260915_223719_077806',
    'v2': 'reviewed-v2-baseline_20260916_183807_199935',
    'v3': 'reviewed-v3-baseline_20260916_222621_316060',
}

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    out = ROOT/'reports'/('reassessment_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    print(f'OUTPUT={out}', flush=True)
    config = ROOT/'data.reviewed-v1.yaml'
    data = yaml.safe_load(config.read_text(encoding='utf-8'))
    source = Path(data['path'])/'valid'
    inputs = list(source.rglob('*'))
    before = {str(p):sha(p) for p in inputs if p.is_file()}
    def memory_cache(prefix, path, cache, version):
        cache['version'] = version
    dataset_module.save_dataset_cache_file = memory_cache
    experiments = [(tag, tag, False, .7) for tag in MODELS]
    experiments += [('v3_tta', 'v3', True, .7), ('v3_nms05', 'v3', False, .5)]
    rows = []
    for name, tag, augment, iou in experiments:
        weights = ROOT/'runs'/MODELS[tag]/'weights/best.pt'
        weight_hash = sha(weights)
        print(f'RUN={name}', flush=True)
        model = YOLO(str(weights))
        try:
            result = model.val(data=str(config), split='val', imgsz=640, batch=1,
                workers=0, device=0, half=False, conf=.001, iou=iou, max_det=300,
                rect=True, augment=augment, plots=False, save_txt=True, save_conf=True,
                project=str(out), name=name, exist_ok=False, verbose=False)
            row = dict(name=name, status='ok', precision=float(result.box.mp),
                recall=float(result.box.mr), map50=float(result.box.map50),
                map5095=float(result.box.map), inference_ms=float(result.speed['inference']),
                weights=str(weights), sha256=weight_hash, augment=augment, nms_iou=iou)
            details=[]
            for p in sorted((source/'images').iterdir()):
                with Image.open(p) as im: w,h=im.size
                def boxes(label, prediction=False):
                    values=[]
                    if label.exists():
                        for line in label.read_text().splitlines():
                            fields=list(map(float,line.split()))
                            _,x,y,bw,bh=fields[:5]
                            box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
                            values.append(box+([fields[5],0] if prediction else []))
                    return np.asarray(values).reshape(-1,6 if prediction else 4)
                gt=boxes(source/'labels'/f'{p.stem}.txt')
                pred=boxes(out/name/'labels'/f'{p.stem}.txt',True)
                hits,ph,ids=match(gt,pred,.25)
                details.append(dict(image=str(p),gt=gt.tolist(),pred=pred.tolist(),
                    tp=len(hits),fp=len(ids)-len(ph),fn=len(gt)-len(hits)))
            row['fixed_conf025']={k:sum(d[k] for d in details) for k in ('tp','fp','fn')}
            (out/name/'diagnostics.json').write_text(json.dumps(details),encoding='utf-8')
            # Reproducible worst-error examples, rather than hand-picked successes.
            sheet=Image.new('RGB',(1200,900),'white')
            for idx,d in enumerate(sorted(details,key=lambda x:x['fn']+x['fp'],reverse=True)[:6]):
                im=Image.open(d['image']).convert('RGB'); w,h=im.size
                im.thumbnail((400,255)); sx,sy=im.width/w,im.height/h
                panel=Image.new('RGB',(400,450),'white'); panel.paste(im,(0,35)); draw=ImageDraw.Draw(panel)
                draw.text((4,4),f"{name}: TP {d['tp']} FP {d['fp']} FN {d['fn']}",fill='black')
                for b in d['gt']:
                    draw.rectangle((b[0]*sx,b[1]*sy+35,b[2]*sx,b[3]*sy+35),outline='lime',width=2)
                for b in d['pred']:
                    if b[4]>=.25:
                        draw.rectangle((b[0]*sx,b[1]*sy+35,b[2]*sx,b[3]*sy+35),outline='red',width=1)
                draw.text((4,310),Path(d['image']).name[:48],fill='black')
                sheet.paste(panel,((idx%3)*400,(idx//3)*450))
            sheet.save(out/name/'worst.jpg')
        except torch.cuda.OutOfMemoryError as exc:
            row=dict(name=name,status='oom',error=str(exc))
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        assert sha(weights)==weight_hash, 'Weights changed during evaluation'
        print(json.dumps(row),flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    assert all(sha(Path(p))==digest for p,digest in before.items())
    (out/'VERIFIED.json').write_text(json.dumps(dict(validation_unchanged=True,test_used=False,
        training_started=False,images=249,batch=1,workers=0,imgsz=640)),encoding='utf-8')

if __name__=='__main__':
    main()
