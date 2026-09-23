"""Read-only train/validation audit and threshold review using saved predictions."""
import base64
from collections import defaultdict
from datetime import datetime
import hashlib
import html
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from analyze_validation import match, overlaps

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'reports/error_analysis_20260915_214707'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def thumb(path, width=440):
    im=Image.open(path).convert('RGB'); im.thumbnail((width,440)); return im


def embed(im):
    stream=io.BytesIO(); im.save(stream,format='JPEG',quality=82)
    return '<img src="data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'


def page(path,title,intro,body):
    path.write_text('<!doctype html><html lang="tr"><meta charset="utf-8"><title>'+title+'</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#172333}'
        'article{background:white;padding:20px;margin:18px 0;border:1px solid #ccd5df;border-radius:8px}'
        '.pair{display:flex;gap:16px;flex-wrap:wrap}.pair figure{flex:1;min-width:280px}figure{margin:0}img{max-width:100%;height:auto}h2{font-size:20px}'
        'summary{cursor:pointer;padding:12px}code{overflow-wrap:anywhere}p{line-height:1.55}</style>'
        '<h1>'+title+'</h1><p>'+intro+'</p>'+body+'</html>',encoding='utf-8')


def main():
    out=ROOT/'reports'/('followup_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')); out.mkdir()
    print(f'OUTPUT_DIR={out}',flush=True)
    records=json.loads((SOURCE/'predictions.json').read_text(encoding='utf-8'))
    groups={}
    for split in ('train','valid'):
        groups[split]=defaultdict(list)
        for p in sorted((ROOT/'data/MWPD'/split/'images').glob('*')):
            if p.is_file(): groups[split][p.name.split('.rf.')[0]].append(p)
    common=sorted(set(groups['train'])&set(groups['valid']))
    sift=cv2.SIFT_create(nfeatures=1200)
    cache={}
    def features(path, flip=False):
        key=(str(path),flip)
        if key not in cache:
            im=np.asarray(thumb(path,640).convert('L'))
            if flip: im=np.fliplr(im).copy()
            kp,des=sift.detectAndCompute(im,None)
            cache[key]=(kp,des,im.shape)
        return cache[key]
    def compare(a,b):
        best=dict(inliers=0,matches=0,ratio=0,coverage=0,flip=False)
        kb,db,sb=features(b)
        if db is None: return best
        for flip in (False,True):
            ka,da,sa=features(a,flip)
            if da is None: continue
            pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
            good=[p[0] for p in pairs if len(p)==2 and p[0].distance < .72*p[1].distance]
            if len(good)<4: continue
            pa=np.float32([ka[m.queryIdx].pt for m in good]); pb=np.float32([kb[m.trainIdx].pt for m in good])
            matrix,mask=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=4)
            if mask is None: continue
            inside=mask.ravel().astype(bool); n=int(inside.sum())
            if n>=3:
                hull=cv2.convexHull(pb[inside]); coverage=float(cv2.contourArea(hull)/(sb[0]*sb[1]))
            else: coverage=0
            if n>best['inliers']: best=dict(inliers=n,matches=len(good),ratio=n/len(good),coverage=coverage,flip=flip)
        return best
    audits=[]; cards=[]
    for idx,name in enumerate(common):
        best=None
        for a in groups['train'][name]:
            for b in groups['valid'][name]:
                metric=compare(a,b)
                if best is None or metric['inliers']>best['inliers']:
                    best=dict(source=name,train=str(a),validation=str(b),**metric)
        best['geometric_support']=best['inliers']>=12 and best['ratio']>=.4 and best['coverage']>=.05
        audits.append(best)
        cards.append('<article><h2>'+html.escape(name)+'</h2><p>'+('Geometrik eşleşme desteği var' if best['geometric_support'] else 'Görsel kontrol gerekli')+
            f' · RANSAC uyumlu nokta: {best["inliers"]}/{best["matches"]} · alan kapsamı: %{best["coverage"]*100:.1f}</p>'+
            '<div class="pair"><figure><figcaption>TRAIN</figcaption>'+embed(thumb(best['train']))+'</figure>'+
            '<figure><figcaption>VALIDATION</figcaption>'+embed(thumb(best['validation']))+'</figure></div>'+
            '<details><summary>Dosya yolları</summary><code>'+html.escape(best['train'])+'<br>'+html.escape(best['validation'])+'</code></details></article>')
        if (idx+1)%15==0: print(f'Source families checked: {idx+1}/{len(common)}',flush=True)
    dump(out/'overlap_evidence.json',audits)
    page(out/'01_ortak_sahneler.html','Ortak sahneler',
        '89 ortak kaynak ailesinin her biri için en güçlü train–validation eşleşmesi. SIFT + RANSAC, aynalamayı da dener. '
        'Geometrik destek otomatik kanıttır; tüm çiftlerde manuel kesin doğrulama anlamına gelmez. Aynı ad taşımayan benzer sahneler taranmadı.', ''.join(cards))
    train_keep=[str(p) for name,paths in groups['train'].items() if name not in common for p in paths]
    excluded=[str(p) for name in common for p in groups['train'][name]]
    val=[str(p) for paths in groups['valid'].values() for p in paths]
    unseen=[str(p) for name,paths in groups['valid'].items() if name not in common for p in paths]
    for name,paths in [('proposed_train',train_keep),('excluded_train_family_overlap',excluded),('proposed_validation',val),('current_model_unseen_by_filename',unseen)]:
        (out/(name+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    plan=dict(shared_families=len(common),geometrically_supported=sum(a['geometric_support'] for a in audits),
        proposed_train_images=len(train_keep),excluded_train_images=len(excluded),validation_images=len(val),
        filename_unseen_validation_images=len(unseen),filename_unseen_validation_families=len(groups['valid'])-len(common))
    dump(out/'split_plan.json',plan)
    label_cards=[]; suspect=[]; threshold_cards=[]; changes=[]
    def annotate(rec, threshold=None):
        im=Image.open(rec['path']).convert('RGB'); scale=600/im.width; im=im.resize((600,round(im.height*scale)))
        d=ImageDraw.Draw(im)
        gt=np.array(rec['gt']).reshape(-1,4); pred=np.array(rec['predictions']).reshape(-1,6)
        matched,hits,ids=match(gt,pred,threshold) if threshold is not None else (set(),set(),[])
        for i,b in enumerate(gt):
            box=(b*scale).tolist(); d.rectangle(box,outline='lime' if threshold is None or i in matched else 'yellow',width=3)
            d.text((box[0],box[1]),f'G{i}',fill='white',stroke_width=1,stroke_fill='black')
        for i in ids:
            b=pred[i]; box=(b[:4]*scale).tolist(); color='cyan' if i in hits else 'red'
            d.rectangle(box,outline=color,width=2); d.text((box[0],max(0,box[1]-12)),f'{b[4]:.2f}',fill='white',stroke_width=1,stroke_fill='black')
        return im
    for rec in records:
        gt=np.array(rec['gt']).reshape(-1,4); pred=np.array(rec['predictions']).reshape(-1,6)
        area=(gt[:,2:]-gt[:,:2]).prod(1)/(rec['width']*rec['height'])
        ious=overlaps(gt,gt); np.fill_diagonal(ious,0)
        pairs=np.argwhere(np.triu(ious>=.5,1)); large=np.flatnonzero(area>=.3)
        if len(pairs) or len(large):
            item=dict(image=rec['path'],large_indices=large.tolist(),overlap_pairs=pairs.tolist(),max_area=float(area.max(initial=0)),max_iou=float(ious.max(initial=0)))
            suspect.append(item)
            label_cards.append((len(pairs)*2+len(large),'<article><h2>'+html.escape(Path(rec['path']).name)+'</h2>'+
                f'<p>Görüntünün ≥%30 alanını kaplayan kutular: {large.tolist()}; IoU ≥0,50 olan etiket çiftleri: {pairs.tolist()}. '
                'Bu eşikler yalnız inceleme adayı üretir; hata kararı değildir.</p>'+embed(annotate(rec))+'</article>'))
        a,ha,ia=match(gt,pred,.25); b,hb,ib=match(gt,pred,.35)
        lost=sorted(a-b); fp_a=len(ia)-len(ha); fp_b=len(ib)-len(hb)
        if len(ia)!=len(ib):
            change=dict(image=rec['path'],tp025=len(ha),fp025=fp_a,fn025=len(gt)-len(a),
                tp035=len(hb),fp035=fp_b,fn035=len(gt)-len(b),lost_gt=lost,
                removed_predictions=len(ia)-len(ib),fp_reduction=fp_a-fp_b)
            changes.append(change)
            card='<article><h2>'+html.escape(Path(rec['path']).name)+'</h2><p>'+f'Ek kaçan: {len(lost)}; azalan FP: {fp_a-fp_b}.</p><div class="pair">'
            for t,hits,ids,matched in [(.25,ha,ia,a),(.35,hb,ib,b)]:
                card+=f'<figure><figcaption>conf={t} · TP {len(hits)} / FP {len(ids)-len(hits)} / FN {len(gt)-len(matched)}</figcaption>'+embed(annotate(rec,t))+'</figure>'
            threshold_cards.append((len(lost)*10+fp_a-fp_b,card+'</div></article>'))
    dump(out/'label_review_candidates.json',suspect); dump(out/'threshold_changes.json',changes)
    page(out/'02_etiket_inceleme.html','Şüpheli etiket inceleme listesi',
        f'{len(suspect)} görüntü otomatik kurallarla işaretlendi. Yeşil kutular veri etiketleridir. '
        'Nesne başına kutu, grup kutusu, eksik etiket, dönüşüm hatası ve belirsiz su alanı açısından kontrol edin. Hiçbir etiket değiştirilmedi.',
        ''.join(c for _,c in sorted(label_cards,key=lambda x:-x[0])))
    page(out/'03_esik_karsilastirma.html','0,25 ve 0,35 karşılaştırması',
        f'{len(changes)} görüntüde tahmin sayısı değişti. Aynı kayıtlı 640 tahminleri kullanıldı; yeni çıkarım yapılmadı. '
        'Yeşil: eşleşen etiket; sarı: kaçan etiket; turkuaz: eşleşen tahmin; kırmızı: eşleşmeyen tahmin. '
        'Eşleşme IoU ≥0,50 ve güven sıralı bire bir eşleştirme. Önce ek çukur kaçıran örnekler gösterilir.',
        ''.join(c for _,c in sorted(threshold_cards,key=lambda x:-x[0])))
    # A compact contact sheet for manual review of the first 12 source families.
    sheet=Image.new('RGB',(880,12*245),'white'); d=ImageDraw.Draw(sheet)
    for i,a in enumerate(audits[:12]):
        d.text((5,i*245),a['source']+' | train (left) / validation (right)',fill='black')
        for j,key in enumerate(('train','validation')):
            im=thumb(a[key],220); im.thumbnail((430,220)); sheet.paste(im,(j*440,i*245+22))
    sheet.save(out/'overlap_contact.jpg')
    dump(out/'summary.json',dict(**plan,label_candidates=len(suspect),threshold_changed_images=len(changes),
        lost_tp=sum(c['tp025']-c['tp035'] for c in changes),fp_reduction=sum(c['fp_reduction'] for c in changes)))
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__': main()
