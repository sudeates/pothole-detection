"""Generate train-only background crop proposals, never auto-label them negative."""
from pathlib import Path
import random
import json
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/MWPD_reviewed_v1/train'
OUT=ROOT/'reports/train_label_review_20260916_144315/negative_crop_review'

def main():
    OUT.mkdir(exist_ok=False)
    images=sorted((SOURCE/'images').glob('*.jpg'))
    random.Random(42).shuffle(images)
    selected=[]; seen=set()
    for path in images:
        prefix=path.name.split('.rf.')[0]
        if prefix in seen: continue
        im=Image.open(path).convert('RGB')
        w,h=im.size
        if min(w,h)<400: continue
        boxes=[]
        for line in (SOURCE/'labels'/path.with_suffix('.txt').name).read_text().splitlines():
            _,x,y,bw,bh=map(float,line.split())
            boxes.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
        size=int(min(w,h)*.4)
        candidates=[]
        for fx in (0,.3,.6):
            for fy in (.3,.6,0):
                x=int(w*fx); y=int(h*fy)
                if x+size>w or y+size>h: continue
                if any(min(x+size+16,b[2])>max(x-16,b[0]) and min(y+size+16,b[3])>max(y-16,b[1]) for b in boxes): continue
                crop=im.crop((x,y,x+size,y+size))
                gray=crop.convert('L'); hist=gray.histogram()
                if sum(hist[:12])/size**2>.025 or sum(hist[244:])/size**2>.2: continue
                candidates.append((x,y,crop))
        if not candidates: continue
        x,y,crop=candidates[0]
        i=len(selected)+1
        crop.save(OUT/f'crop_{i:02}.png')
        selected.append(dict(id=i,source=str(path),source_split='train',crop_xyxy=[x,y,x+size,y+size],
                             file=f'crop_{i:02}.png',status='unreviewed; no label overlap does not prove absence of potholes'))
        seen.add(prefix)
        if len(selected)==24: break
    for start in range(0,len(selected),12):
        sheet=Image.new('RGB',(1200,940),'white'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected[start:start+12]):
            im=Image.open(OUT/r['file']); im.thumbnail((290,280))
            x=(j%4)*300; y=(j//4)*310
            sheet.paste(im,(x,y+25)); draw.text((x+5,y+5),f"#{r['id']} {Path(r['source']).name[:25]}",fill='black')
        sheet.save(OUT/f'sheet_{start//12+1}.jpg',quality=95)
    (OUT/'candidates.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print(f'Generated {len(selected)} unreviewed crops: {OUT}')

if __name__=='__main__': main()
