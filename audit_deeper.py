"""Read-only visual review preparation and cross-name duplicate candidate scan."""
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from audit_followups import thumb, embed, page, dump

ROOT=Path(__file__).resolve().parent
PREV=ROOT/'reports/followup_audit_20260915_215342'


def load_gray(path):
    im=Image.open(path).convert('L'); im.thumbnail((640,640))
    return np.asarray(im)


def main():
    out=ROOT/'reports'/('deep_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')); out.mkdir()
    print(f'OUTPUT_DIR={out}',flush=True)
    uncertain=[a for a in json.loads((PREV/'overlap_evidence.json').read_text()) if not a['geometric_support']]
    for start in range(0,len(uncertain),3):
        sheet=Image.new('RGB',(1000,3*365),'white'); d=ImageDraw.Draw(sheet)
        for row,a in enumerate(uncertain[start:start+3]):
            d.text((10,row*365+5),f'{start+row+1}: {a["source"]} | TRAIN left / VALIDATION right',fill='black')
            for j,k in enumerate(('train','validation')):
                im=Image.open(a[k]).convert('RGB'); im.thumbnail((490,330)); sheet.paste(im,(j*500,row*365+28))
        sheet.save(out/f'uncertain_{start//3+1:02}.jpg',quality=94)
    candidates=json.loads((PREV/'label_review_candidates.json').read_text())
    recs={r['path']:r for r in json.loads((ROOT/'reports/error_analysis_20260915_214707/predictions.json').read_text())}
    for start in range(0,len(candidates),6):
        sheet=Image.new('RGB',(1440,3*510),'white'); d=ImageDraw.Draw(sheet)
        for k,a in enumerate(candidates[start:start+6]):
            rec=recs[a['image']]; im=Image.open(a['image']).convert('RGB'); im.thumbnail((710,475))
            scale=im.width/rec['width']; dr=ImageDraw.Draw(im)
            for g,b in enumerate(rec['gt']):
                box=[v*scale for v in b]; dr.rectangle(box,outline='#00ff00',width=3)
                dr.text((box[0]+2,box[1]+2),f'G{g}',fill='white',stroke_width=2,stroke_fill='black')
            x=(k%2)*720; y=(k//2)*510
            d.text((x+5,y+5),f'{start+k+1}: {Path(a["image"]).name}',fill='black')
            sheet.paste(im,(x,y+25))
        sheet.save(out/f'labels_{start//6+1:02}.jpg',quality=94)
    dump(out/'review_index.json',dict(uncertain=uncertain,labels=candidates))
    train=sorted((ROOT/'data/MWPD/train/images').glob('*'))
    val=sorted((ROOT/'data/MWPD/valid/images').glob('*'))
    def family(p):return p.name.split('.rf.')[0]
    def phash(im):
        small=cv2.resize(im,(32,32)).astype(np.float32)
        coeff=cv2.dct(small)[:8,:8].flatten()[1:]
        return coeff>np.median(coeff)
    hashes=[]
    for i,p in enumerate(train):
        hashes.append(phash(load_gray(p)))
        if (i+1)%700==0:print(f'Image fingerprints: {i+1}/{len(train)}',flush=True)
    hashes=np.asarray(hashes); tf=np.array([family(p) for p in train])
    pairs={}
    for p in val:
        im=load_gray(p)
        variants=[phash(np.rot90(im,k)) for k in range(4)]+[phash(np.rot90(np.fliplr(im),k)) for k in range(4)]
        distances=np.min([np.count_nonzero(hashes != h,axis=1) for h in variants],axis=0)
        distances[tf==family(p)]=100
        chosen=set()
        for idx in np.argsort(distances):
            if distances[idx]>20 or len(chosen)>=3:break
            fam=family(train[idx])
            if fam in chosen:continue
            chosen.add(fam)
            pairs[(str(train[idx]),str(p))]=int(distances[idx])
    print(f'Cross-name pairs to verify: {len(pairs)}',flush=True)
    sift=cv2.SIFT_create(nfeatures=1600)
    @lru_cache(maxsize=128)
    def features(path,flip):
        im=load_gray(path)
        if flip:im=np.fliplr(im).copy()
        kp,desc=sift.detectAndCompute(im,None)
        return kp,desc,im.shape
    results=[]
    for n,((a,b),dist) in enumerate(pairs.items()):
        kb,db,sb=features(b,False); best=dict(inliers=0,matches=0,ratio=0.,coverage=0.,flip=False)
        if db is not None:
            for flip in (False,True):
                ka,da,sa=features(a,flip)
                if da is None:continue
                raw=cv2.BFMatcher().knnMatch(da,db,k=2)
                good=[m[0] for m in raw if len(m)==2 and m[0].distance<.72*m[1].distance]
                if len(good)<6:continue
                pa=np.float32([ka[m.queryIdx].pt for m in good]); pb=np.float32([kb[m.trainIdx].pt for m in good])
                mat,mask=cv2.findHomography(pa,pb,cv2.RANSAC,4)
                if mask is None:continue
                inside=mask.ravel().astype(bool); count=int(inside.sum())
                coverage=float(cv2.contourArea(cv2.convexHull(pb[inside]))/(sb[0]*sb[1])) if count>=3 else 0.
                if count>best['inliers']:best=dict(inliers=count,matches=len(good),ratio=count/len(good),coverage=coverage,flip=flip)
        supported=best['inliers']>=15 and best['ratio']>=.45 and best['coverage']>=.08
        results.append(dict(train=a,validation=b,train_family=family(Path(a)),val_family=family(Path(b)),
            hash_distance=dist,geometric_support=supported,**best))
        if (n+1)%60==0:print(f'Cross-name geometry: {n+1}/{len(pairs)}',flush=True)
    dump(out/'cross_name_scan.json',results)
    supported=[r for r in results if r['geometric_support']]
    cards=[]
    for r in sorted(supported,key=lambda r:-r['inliers']):
        cards.append('<article><h2>'+r['train_family']+' / '+r['val_family']+'</h2><p>'+str(r['inliers'])+
            ' uyumlu nokta</p><div class="pair"><figure>TRAIN'+embed(thumb(r['train']))+'</figure><figure>VALIDATION'+embed(thumb(r['validation']))+'</figure></div></article>')
    page(out/'cross_name_matches.html','Farklı adlarla ortak sahne adayları',
        'Algısal özetle aday seçimi ve geometrik eşleştirme. Sonuçlar görsel inceleme gerektirir; tüm olası benzerliklerin bulunması garanti değildir. Test okunmadı.', ''.join(cards))
    for start in range(0,len(supported),3):
        sheet=Image.new('RGB',(1000,1095),'white'); d=ImageDraw.Draw(sheet)
        for j,r in enumerate(supported[start:start+3]):
            d.text((5,j*365+5),f'{start+j}: {r["train_family"]} TRAIN / {r["val_family"]} VALIDATION',fill='black')
            for k,key in enumerate(('train','validation')):
                im=Image.open(r[key]).convert('RGB'); im.thumbnail((490,330)); sheet.paste(im,(k*500,j*365+25))
        sheet.save(out/f'cross_{start//3+1:02}.jpg',quality=94)
    dump(out/'scan_summary.json',dict(train_images=len(train),val_images=len(val),candidate_pairs=len(results),
        supported_pairs=len(supported),supported_family_pairs=len(set((r['train_family'],r['val_family']) for r in supported))))
    print((out/'scan_summary.json').read_text(),flush=True)


if __name__=='__main__':main()
