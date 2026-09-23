"""Read-only model/data analysis. Outputs go to a unique reports directory."""
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def overlaps(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.maximum(hi-lo, 0).prod(2)
    areas_a = np.maximum(a[:, 2:4]-a[:, :2], 0).prod(1)
    areas_b = np.maximum(b[:, 2:4]-b[:, :2], 0).prod(1)
    return inter / np.maximum(areas_a[:, None]+areas_b[None, :]-inter, 1e-9)


def match(gt, pred, threshold):
    ids = np.flatnonzero(pred[:, 4] >= threshold)
    used, hits = set(), []
    ious = overlaps(gt, pred[ids])
    for j in np.argsort(-pred[ids, 4]):
        candidates = [g for g in range(len(gt)) if g not in used and ious[g, j] >= .5]
        if candidates:
            g = max(candidates, key=lambda g: ious[g, j])
            used.add(g)
            hits.append(int(ids[j]))
    return used, set(hits), ids


def main():
    out = ROOT / 'reports' / ('error_analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    print(f'OUTPUT_DIR={out}', flush=True)
    with (ROOT / 'runs/baseline-3/results.csv').open() as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    key = 'metrics/mAP50-95(B)'
    best = max(rows, key=lambda r: r[key])
    windows = []
    for lo, hi in [(1,10),(39,48),(49,58),(59,68),(69,78)]:
        chunk = [r for r in rows if lo <= r['epoch'] <= hi]
        windows.append(dict(epochs=f'{lo}-{hi}', **{k:float(np.mean([r[k] for r in chunk])) for k in rows[0] if k not in ('epoch','time')}))
    dump(out / 'training_summary.json', dict(epochs=len(rows), best=best, last=rows[-1], windows=windows))
    fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
    for ax, keys, title in zip(axes.flat,
        [['metrics/mAP50(B)',key], ['metrics/precision(B)','metrics/recall(B)'],
         ['train/box_loss','val/box_loss'], ['train/cls_loss','val/cls_loss']],
        ['Validation mAP','Validation precision / recall','Box loss','Classification loss']):
        for k in keys:
            ax.plot([r['epoch'] for r in rows], [r[k] for r in rows], label=k)
        ax.axvline(best['epoch'], color='gray', linestyle=':', label=f'Best mAP: epoch {int(best["epoch"])}')
        ax.set(title=title, xlabel='Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)
    config = yaml.safe_load((ROOT / 'data.local.yaml').read_text())
    source = Path(config['path']) / config['val']
    model = YOLO(str(ROOT / 'runs/baseline-3/weights/best.pt'))
    records = []
    for idx, result in enumerate(model.predict(source=str(source), imgsz=640, batch=1,
            workers=0, device=0, conf=.001, iou=.7, max_det=300, half=False,
            rect=True, augment=False, stream=True, verbose=False, save=False)):
        h, w = result.orig_shape
        label = source.parent / 'labels' / (Path(result.path).stem + '.txt')
        gt = []
        for line in label.read_text().splitlines():
            cls, x, y, bw, bh = map(float, line.split())
            assert cls == 0 and bw > 0 and bh > 0, label
            gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        gt = np.asarray(gt).reshape(-1,4)
        pred = result.boxes.data.cpu().numpy()
        records.append(dict(path=result.path, width=w, height=h, gt=gt.tolist(), predictions=pred.tolist()))
        if (idx+1) % 50 == 0:
            print(f'Validation inference: {idx+1} images', flush=True)
    dump(out / 'predictions.json', records)
    thresholds = [.05,.1,.15,.2,.25,.3,.35,.4,.5]
    sweep, details, size_stats = [], [], {s:dict(total=0,matched=0) for s in ('small','medium','large')}
    for threshold in thresholds:
        tp=fp=fn=0
        for rec in records:
            gt = np.asarray(rec['gt']).reshape(-1,4)
            pred = np.asarray(rec['predictions']).reshape(-1,6)
            matched, hits, ids = match(gt, pred, threshold)
            tp += len(hits); fp += len(ids)-len(hits); fn += len(gt)-len(matched)
            if threshold == .25:
                scale = 640/max(rec['width'],rec['height'])
                sizes = np.sqrt((gt[:,2:]-gt[:,:2]).prod(1))*scale
                for g, size in enumerate(sizes):
                    bucket = 'small' if size < 32 else 'medium' if size < 96 else 'large'
                    size_stats[bucket]['total'] += 1
                    size_stats[bucket]['matched'] += int(g in matched)
                ious = overlaps(gt,pred[ids])
                missed = [g for g in range(len(gt)) if g not in matched]
                near = sum(.1 <= ious[g].max(initial=0) < .5 for g in missed)
                recover, _, _ = match(gt,pred,.1)
                details.append(dict(image=Path(rec['path']).name, tp=len(hits), fp=len(ids)-len(hits),
                    fn=len(missed), localization_candidates=int(near),
                    recovered_at_010=len(set(missed)&recover),
                    missed_gt_indices=missed, false_pred_indices=[int(i) for i in ids if i not in hits]))
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
        sweep.append(dict(conf=threshold,tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-9)))
    dump(out / 'diagnostics.json',dict(threshold_sweep=sweep,size_stats=size_stats, images=details))
    # Draw all errors for audit, plus small contact sheets for review.
    error_dir=out/'examples'; error_dir.mkdir()
    def draw(rec, detail):
        im=Image.open(rec['path']).convert('RGB'); scale=800/im.width
        im=im.resize((800,round(im.height*scale)))
        canvas=Image.new('RGB',(800,im.height+44),'white'); canvas.paste(im,(0,44)); d=ImageDraw.Draw(canvas)
        d.text((8,5),f'{detail["image"][:85]}',fill='black')
        d.text((8,23),f'TP={detail["tp"]} FP={detail["fp"]} FN={detail["fn"]} | GT green; missed yellow; prediction cyan; FP red',fill='black')
        for i,b in enumerate(rec['gt']):
            box=[b[0]*scale,b[1]*scale+44,b[2]*scale,b[3]*scale+44]
            d.rectangle(box,outline='yellow' if i in detail['missed_gt_indices'] else 'lime',width=3)
            d.text((box[0],box[1]),f'G{i}',fill='yellow' if i in detail['missed_gt_indices'] else 'lime')
        for i,b in enumerate(rec['predictions']):
            if b[4]<.25: continue
            box=[b[0]*scale,b[1]*scale+44,b[2]*scale,b[3]*scale+44]
            color='red' if i in detail['false_pred_indices'] else 'cyan'
            d.rectangle(box,outline=color,width=2); d.text((box[0],max(44,box[1]-12)),f'P{i} {b[4]:.2f}',fill=color)
        return canvas
    for rec, detail in zip(records,details):
        if detail['fp'] or detail['fn']:
            draw(rec,detail).save(error_dir/(Path(rec['path']).stem+'.jpg'),quality=90)
    for kind in ('fn','fp'):
        selected=sorted(range(len(details)),key=lambda i:details[i][kind],reverse=True)[:6]
        panels=[draw(records[i],details[i]) for i in selected]
        height=max(p.height for p in panels)
        sheet=Image.new('RGB',(1600,height*3),'#dddddd')
        for n,panel in enumerate(panels): sheet.paste(panel,((n%2)*800,(n//2)*height))
        sheet.save(out/f'top_{kind}.jpg',quality=92)
    print(json.dumps(dict(best_epoch=best['epoch'],best_map=best[key],windows=windows,
        threshold_sweep=sweep,size_stats=size_stats,images=len(records),
        error_images=sum(d['fp']>0 or d['fn']>0 for d in details),
        localization_candidates=sum(d['localization_candidates'] for d in details),
        recovered_at_010=sum(d['recovered_at_010'] for d in details)),indent=2),flush=True)


if __name__=='__main__': main()
