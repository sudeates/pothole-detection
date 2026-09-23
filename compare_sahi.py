"""SAHI vs standard prediction on MWPD validation, fixed conf .25; no training."""
from pathlib import Path
import sys, json, time, hashlib
from datetime import datetime
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.tools/sahi'))
import numpy as np
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from analyze_validation import match

def main():
    weights=ROOT/'runs/reviewed-v3-baseline_20260916_222621_316060/weights/best.pt'
    source=ROOT/'data/MWPD_reviewed_v1/valid'
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in [weights,*source.rglob('*.jpg'),*source.rglob('*.txt')]}
    out=ROOT/'reports'/('sahi_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    print(f'OUTPUT={out}',flush=True)
    model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(weights),
        confidence_threshold=.25,device='cuda:0',image_size=640)
    images=sorted((source/'images').glob('*.jpg'))
    assert len(images)==249
    get_prediction(str(images[0]),model) # excluded warmup
    records=[]
    for idx,p in enumerate(images):
        with Image.open(p) as im: w,h=im.size
        gt=[]
        for line in (source/'labels'/f'{p.stem}.txt').read_text().splitlines():
            _,x,y,bw,bh=map(float,line.split());gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        gt=np.asarray(gt).reshape(-1,4)
        for mode in ('standard','sahi'):
            torch.cuda.synchronize(); start=time.perf_counter()
            try:
                if mode=='standard': r=get_prediction(str(p),model)
                else:
                    tile=max(128,int(max(w,h)/2))
                    r=get_sliced_prediction(str(p),model,slice_height=tile,slice_width=tile,
                        overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,
                        postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,
                        auto_slice_resolution=False,verbose=0,batch_size=1)
                torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                pred=np.asarray([o.bbox.to_xyxy()+[o.score.value,o.category.id] for o in r.object_prediction_list]).reshape(-1,6)
                hits,ph,ids=match(gt,pred,.25)
                records.append(dict(image=p.name,mode=mode,tp=len(hits),fp=len(ids)-len(ph),fn=len(gt)-len(hits),ms=ms,
                    predictions=pred.tolist(),status='ok'))
            except torch.cuda.OutOfMemoryError:
                records.append(dict(image=p.name,mode=mode,status='oom'));torch.cuda.empty_cache()
        if (idx+1)%25==0: print(f'{idx+1}/{len(images)}',flush=True)
    (out/'predictions.json').write_text(json.dumps(records),encoding='utf-8')
    summary={}
    for mode in ('standard','sahi'):
        rows=[r for r in records if r['mode']==mode and r['status']=='ok']
        tp,fp,fn=[sum(r[k] for r in rows) for k in ('tp','fp','fn')]
        summary[mode]=dict(images=len(rows),tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
            end_to_end_ms=float(np.mean([r['ms'] for r in rows])))
    assert all(sha(Path(p))==digest for p,digest in before.items())
    (out/'summary.json').write_text(json.dumps(dict(results=summary,conf=.25,match_iou=.5,
        test_used=False,training=False,inputs_unchanged=True,timing='image decode + slicing + prediction + merging; warmup excluded',
        note='Fixed-threshold diagnostic counts, not mAP. Half-long-edge tiles, 20 percent overlap, full-image prediction included.'),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
