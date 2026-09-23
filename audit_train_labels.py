"""Read-only geometry review of training labels; never changes dataset files."""
from pathlib import Path
from datetime import datetime
import json
import html
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    data = ROOT / 'data/MWPD_reviewed_v1/train'
    out = ROOT / 'reports' / ('train_label_review_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    records = []
    for image in sorted((data/'images').iterdir()):
        label = data/'labels'/f'{image.stem}.txt'
        boxes = []
        for line in label.read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
        pairs = []
        for i, a in enumerate(boxes):
            for j in range(i+1, len(boxes)):
                b = boxes[j]
                inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[2]-b[0])*(b[3]-b[1])
                iou=inter/(aa+ab-inter) if aa+ab>inter else 0
                if iou >= .5:
                    pairs.append({'indices':[i+1,j+1], 'iou':iou})
        area=max(((b[2]-b[0])*(b[3]-b[1]) for b in boxes), default=0)
        records.append(dict(image=str(image),label=str(label),boxes=boxes,pairs=pairs,max_area=area,
                            score=10*len(pairs)+max((p['iou'] for p in pairs),default=0)+area))
    ranked=sorted((r for r in records if r['pairs'] or r['max_area']>=.3),key=lambda r:r['score'],reverse=True)
    (out/'candidates.json').write_text(json.dumps(ranked,indent=2),encoding='utf-8')
    # Filename prefixes only diversify the first review batch; no source identity inferred.
    selected=[]; seen=set()
    for r in ranked:
        key=Path(r['image']).name.split('.rf.')[0]
        if key not in seen:
            selected.append(r); seen.add(key)
        if len(selected)==24: break
    cards=[]
    for idx,r in enumerate(selected,1):
        im=Image.open(r['image']).convert('RGB'); im.thumbnail((640,640))
        draw=ImageDraw.Draw(im)
        for k,b in enumerate(r['boxes'],1):
            xy=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height]
            draw.rectangle(xy,outline='#00ff50',width=2)
            draw.text((xy[0]+2,xy[1]+2),str(k),fill='black',stroke_width=2,stroke_fill='white')
        im.save(out/f'candidate_{idx:02}.jpg',quality=92)
        r['review_number']=idx
        cards.append(f'<article><h2>{idx}. {html.escape(Path(r["image"]).name)}</h2><p>Çakışan çift: {len(r["pairs"])}; en büyük kutu: %{r["max_area"]*100:.1f}</p><img src="candidate_{idx:02}.jpg"><p>Durum: görsel inceleme adayı; otomatik hata kararı değildir.</p></article>')
    for start in range(0,len(selected),6):
        sheet=Image.new('RGB',(1200,1260),'white')
        for pos,r in enumerate(selected[start:start+6]):
            im=Image.open(out/f'candidate_{start+pos+1:02}.jpg'); im.thumbnail((590,390))
            x=(pos%2)*600; y=(pos//2)*420
            sheet.paste(im,(x,y+25)); ImageDraw.Draw(sheet).text((x+8,y+5),f'#{start+pos+1} {Path(r["image"]).name[:45]}',fill='black')
        sheet.save(out/f'sheet_{start//6+1}.jpg',quality=90)
    (out/'selected.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Eğitim etiketi incelemesi</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto}article{border-bottom:1px solid #ccc;padding:20px}img{max-width:100%}</style><h1>Eğitim etiketi incelemesi</h1><p>Yalnız train; özgün etiketler değiştirilmedi. Kutu numaraları etiket satırlarının sırasıdır.</p>'+''.join(cards),encoding='utf-8')
    summary=dict(train_images=len(records),empty_labels=sum(not r['boxes'] for r in records),overlap_images=sum(bool(r['pairs']) for r in records),large_box_images=sum(r['max_area']>=.3 for r in records),candidate_images=len(ranked),review_batch=len(selected),output=str(out))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
