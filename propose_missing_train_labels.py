"""Register visually identified copies and propose missing boxes; no model inference."""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/full_train_label_review'
PAIRS=[(271,[273,274,275])]

def xyxy(rows,w,h):
    return np.array([[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h] for _,x,y,bw,bh in rows])

def iou(a,b):
    it=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return it/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it+1e-9)

def main():
    records=json.loads((OUT/'inventory.json').read_text())
    dest=OUT/'split_box_proposals';dest.mkdir(exist_ok=False)
    sift=cv2.SIFT_create(nfeatures=4000)
    proposals=[]; failures=[]
    for source_id,targets in PAIRS:
        sr=records[source_id-1]; sim=np.array(Image.open(sr['image']).convert('L'))
        sh,sw=sim.shape
        sboxes=xyxy(sr['boxes'],sw,sh)
        for target_id in targets:
            tr=records[target_id-1]; tim=np.array(Image.open(tr['image']).convert('L'));th,tw=tim.shape
            tk,td=sift.detectAndCompute(tim,None);best=None
            for flip in (False,True):
                work=np.fliplr(sim).copy() if flip else sim
                sk,sd=sift.detectAndCompute(work,None)
                matches=cv2.BFMatcher().knnMatch(sd,td,k=2)
                good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
                if len(good)<20:continue
                p=np.float32([sk[m.queryIdx].pt for m in good]);q=np.float32([tk[m.trainIdx].pt for m in good])
                mat,mask=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2)
                if mat is None:continue
                valid=mask.ravel().astype(bool); n=int(valid.sum())
                coverage=cv2.contourArea(cv2.convexHull(q[valid]))/(tw*th) if n>=3 else 0
                error=np.linalg.norm(p[valid]@mat[:,:2].T+mat[:,2]-q[valid],axis=1)
                if n<20 or coverage<.1 or np.median(error)>1.5:continue
                if best is None or n>best['inliers']:
                    best=dict(matrix=mat.tolist(),flip=flip,inliers=n,coverage=coverage,median_error_px=float(np.median(error)))
            if best is None:
                failures.append(dict(source_id=source_id,target_id=target_id,reason='Registration threshold not met'));continue
            existing=xyxy(tr['boxes'],tw,th);added=[]
            replace_all=source_id==271
            for j,b in enumerate(sboxes):
                if replace_all and j not in (1,2,3,5,7,8):continue
                p=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
                if best['flip']:p[:,0]=sw-1-p[:,0]
                mat=np.array(best['matrix']);q=p@mat[:,:2].T+mat[:,2]
                box=np.r_[q.min(axis=0),q.max(axis=0)]
                clipped=np.clip(box,[0,0,0,0],[tw,th,tw,th])
                area=(box[2]-box[0])*(box[3]-box[1]);ca=(clipped[2]-clipped[0])*(clipped[3]-clipped[1])
                if ca<.98*area or ca<16:continue
                if not replace_all and max((iou(box,old) for old in existing),default=0)>.2:continue
                added.append(dict(source_box_1based=j+1,xyxy=clipped.tolist()))
            if not added:
                failures.append(dict(source_id=source_id,target_id=target_id,reason='No fully visible unmatched reference box'));continue
            old=Path(tr['label']).read_text()
            if replace_all and len(added)!=6:
                failures.append(dict(source_id=source_id,target_id=target_id,reason='Incomplete full-scene replacement'));continue
            lines=[line for j,line in enumerate(old.strip().splitlines()) if not replace_all or j!=1]
            for a in added:
                x1,y1,x2,y2=a['xyxy']; lines.append(f'0 {(x1+x2)/2/tw:.9f} {(y1+y2)/2/th:.9f} {(x2-x1)/tw:.9f} {(y2-y1)/th:.9f}')
            target_label=dest/Path(tr['label']).name
            target_label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            im=Image.open(tr['image']).convert('RGB');d=ImageDraw.Draw(im)
            for b in existing:d.rectangle(tuple(b),outline='lime',width=2)
            for a in added:d.rectangle(tuple(a['xyxy']),outline='#ff8000',width=3)
            im.save(dest/f'{target_id}.jpg',quality=96)
            proposals.append(dict(source_id=source_id,target_id=target_id,image=tr['image'],label=tr['label'],
                source_label=sr['label'],source_sha256=hashlib.sha256(Path(tr['label']).read_bytes()).hexdigest(),
                proposed_label=str(target_label),registration=best,added=added,replaced_line_1based=2 if replace_all else None,status='needs_visual_verification'))
    for start in range(0,len(proposals),6):
        sheet=Image.new('RGB',(1200,1260),'white');d=ImageDraw.Draw(sheet)
        for j,p in enumerate(proposals[start:start+6]):
            im=Image.open(dest/f'{p["target_id"]}.jpg');im.thumbnail((590,390))
            x=(j%2)*600;y=(j//2)*420;sheet.paste(im,(x,y+25))
            d.text((x+5,y+4),f"ID {p['target_id']} +{len(p['added'])} | reference {p['source_id']}",fill='black')
        sheet.save(dest/f'sheet_{start//6+1:02}.jpg',quality=95)
    (dest/'proposals.json').write_text(json.dumps(proposals,indent=2),encoding='utf-8')
    (dest/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    print(json.dumps(dict(proposals=len(proposals),added_boxes=sum(len(p['added']) for p in proposals),failures=failures),indent=2))

if __name__=='__main__':main()
