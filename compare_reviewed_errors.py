"""Compare reviewed YOLOv8n/s on validation only, without modifying inputs."""
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from ultralytics import YOLO
import ultralytics.data.dataset as dataset_module
from analyze_validation import overlaps, match
from audit_followups import page, embed

ROOT=Path(__file__).resolve().parent
MODELS={
    'n':ROOT/'runs/reviewed-v1-baseline_20260915_223719_077806/weights/best.pt',
    's':ROOT/'runs/reviewed-v1-yolov8s_20260916_122325_277730/weights/best.pt',
}


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    out=ROOT/'reports'/('reviewed_error_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir();print(f'OUTPUT_DIR={out}',flush=True)
    config=ROOT/'data.reviewed-v1.yaml'
    data=yaml.safe_load(config.read_text(encoding='utf-8'))
    source=Path(data['path'])/data['val']
    records={};metrics={};environment={}
    def memory_cache(prefix,path,cache,version):cache['version']=version
    dataset_module.save_dataset_cache_file=memory_cache
    for tag,weights in MODELS.items():
        model_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
        model=YOLO(str(weights))
        print(f'Validating YOLOv8{tag}',flush=True)
        result=model.val(data=str(config),split='val',imgsz=640,batch=1,workers=0,
            device=0,half=False,conf=.001,iou=.7,max_det=300,rect=True,augment=False,
            plots=False,project=str(out),name=f'val_{tag}',exist_ok=False,verbose=False)
        metrics[tag]=dict(precision=float(result.box.mp),recall=float(result.box.mr),
            map50=float(result.box.map50),map5095=float(result.box.map),
            inference_ms=float(result.speed['inference']))
        for idx,r in enumerate(model.predict(source=str(source),imgsz=640,batch=1,workers=0,
            device=0,half=False,conf=.001,iou=.7,max_det=300,rect=True,augment=False,
            stream=True,save=False,verbose=False)):
            name=Path(r.path).name
            if name not in records:
                h,w=r.orig_shape; gt=[]
                label=source.parent/'labels'/Path(name).with_suffix('.txt')
                for line in label.read_text().splitlines():
                    cls,x,y,bw,bh=map(float,line.split())
                    assert cls==0
                    gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
                records[name]=dict(path=r.path,width=w,height=h,gt=gt)
            records[name][tag]=r.boxes.data.cpu().numpy().tolist()
            if (idx+1)%80==0:print(f'YOLOv8{tag}: {idx+1} predictions',flush=True)
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=model_hash:raise RuntimeError('Model changed during evaluation')
        environment[tag]=dict(weights=str(weights),sha256=model_hash)
        del model,result;gc.collect();torch.cuda.empty_cache()
    dump(out/'predictions.json',records)
    dump(out/'metrics.json',metrics)
    dump(out/'environment.json',dict(models=environment,data=str(config),imgsz=640,batch=1,
        workers=0,conf=.001,diagnostic_conf=.25,iou_match=.5,nms_iou=.7,
        split='val',half=False,test_used=False))
    details=[];summaries={};matched_by_tag={}
    for tag in MODELS:
        causes=Counter();size={k:dict(total=0,matched=0) for k in ['small','medium','large']}
        thresholds=[];matches={}
        for t in [.05,.1,.15,.2,.25,.3,.35,.4,.5]:
            tp=fp=fn=0
            for name,rec in records.items():
                gt=np.asarray(rec['gt']).reshape(-1,4);pred=np.asarray(rec[tag]).reshape(-1,6)
                hits,ph,ids=match(gt,pred,t)
                tp+=len(hits);fp+=len(ids)-len(ph);fn+=len(gt)-len(hits)
                if t==.25:
                    matches[name]=hits
                    recovered,_,_=match(gt,pred,.1)
                    low,_,_=match(gt,pred,.001)
                    ious=overlaps(gt,pred[ids]); per_image=[]
                    for i,box in enumerate(gt):
                        dims=(box[2:]-box[:2])*640/max(rec['width'],rec['height'])
                        sq=float(np.sqrt(dims.prod()));bucket='small' if sq<32 else 'medium' if sq<96 else 'large'
                        size[bucket]['total']+=1;size[bucket]['matched']+=int(i in hits)
                        if i in hits:continue
                        top=float(ious[i].max(initial=0))
                        cause='low_confidence_010' if i in recovered else ('assignment_conflict' if top>=.5 else 'box_alignment' if top>=.1 else 'no_nearby_box_at_025')
                        causes[cause]+=1
                        per_image.append(dict(gt_index=i,cause=cause,max_iou_at_025=top,
                            recovered_at_0001=i in low,size=bucket,width640=float(dims[0]),height640=float(dims[1])))
                    details.append(dict(model=tag,image=name,tp=len(hits),fp=len(ids)-len(ph),fn=len(gt)-len(hits),misses=per_image))
            p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
            thresholds.append(dict(conf=t,tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-9)))
        matched_by_tag[tag]=matches
        summaries[tag]=dict(causes=dict(causes),size=size,thresholds=thresholds)
    differences=[]
    for name,rec in records.items():
        n=matched_by_tag['n'][name];s=matched_by_tag['s'][name]
        differences.append(dict(image=name,n_only=sorted(n-s),s_only=sorted(s-n),
            both=sorted(n&s),neither=sorted(set(range(len(rec['gt'])))-(n|s))))
    dump(out/'diagnostics.json',dict(summary=summaries,details=details,differences=differences))
    def draw(name,tag):
        rec=records[name];gt=np.asarray(rec['gt']).reshape(-1,4);pred=np.asarray(rec[tag]).reshape(-1,6)
        hits,ph,ids=match(gt,pred,.25)
        im=Image.open(rec['path']).convert('RGB');scale=600/im.width;im=im.resize((600,round(im.height*scale)))
        canvas=Image.new('RGB',(600,im.height+44),'white');canvas.paste(im,(0,44));d=ImageDraw.Draw(canvas)
        d.text((5,4),name[:85],fill='black');d.text((5,23),f'YOLOv8{tag} | TP={len(hits)} FP={len(ids)-len(ph)} FN={len(gt)-len(hits)}',fill='black')
        for i,box in enumerate(gt):
            b=(box*scale).tolist();b[1]+=44;b[3]+=44;d.rectangle(b,outline='lime' if i in hits else 'yellow',width=3)
            d.text((b[0],b[1]),f'G{i}',fill='white',stroke_width=1,stroke_fill='black')
        for i in ids:
            box=pred[i];b=(box[:4]*scale).tolist();b[1]+=44;b[3]+=44
            d.rectangle(b,outline='cyan' if i in ph else 'red',width=2)
            d.text((b[0],max(44,b[1]-13)),f'{box[4]:.2f}',fill='white',stroke_width=1,stroke_fill='black')
        return canvas
    rankings={
        's_regressions':sorted(differences,key=lambda d:(len(d['n_only'])-len(d['s_only']),len(d['n_only'])),reverse=True),
        'both_miss':sorted(differences,key=lambda d:len(d['neither']),reverse=True),
        's_false_positives':sorted([d for d in details if d['model']=='s'],key=lambda d:d['fp'],reverse=True),
    }
    for category,ranking in rankings.items():
        chosen=ranking[:6];cards=[]
        for start in (0,3):
            sheet=Image.new('RGB',(1200,3*644),'#dddddd')
            for j,item in enumerate(chosen[start:start+3]):
                name=item['image'];panels=[draw(name,t) for t in ('n','s')]
                for k,panel in enumerate(panels):
                    panel.thumbnail((600,644));sheet.paste(panel,(k*600,j*644))
                cards.append('<article><div class="pair">'+''.join('<figure>'+embed(p)+'</figure>' for p in panels)+'</div></article>')
            sheet.save(out/f'{category}_{start//3+1}.jpg',quality=94)
        page(out/f'{category}.html',category,'Sol: n; sağ: s. Yeşil eşleşen etiket, sarı kaçan etiket; turkuaz eşleşen tahmin, kırmızı eşleşmeyen tahmin. conf=0,25, IoU eşleşme=0,50. Seçilen zor örnekler tüm veri dağılımını temsil etmez.',''.join(cards))
    totals={key:sum(len(d[key]) for d in differences) for key in ['n_only','s_only','both','neither']}
    dump(out/'summary.json',dict(images=len(records),metrics=metrics,models=summaries,paired_counts=totals))
    print(json.dumps(dict(output=str(out),metrics=metrics,models=summaries,paired_counts=totals),indent=2),flush=True)


if __name__=='__main__':main()
