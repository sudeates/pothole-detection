"""Build a complete train-only contact review inventory, without editing labels."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/MWPD_labels_v2_candidate/train'
OUT=ROOT/'reports/full_train_label_review'

def main():
    OUT.mkdir(exist_ok=True)
    records=[]
    for i,p in enumerate(sorted((SOURCE/'images').glob('*')),1):
        label=SOURCE/'labels'/p.with_suffix('.txt').name
        boxes=[list(map(float,line.split())) for line in label.read_text().splitlines()]
        records.append(dict(id=i,image=str(p),label=str(label),boxes=boxes,
                            sheet=(i-1)//36+1,status='not_reviewed'))
    for start in range(0,len(records),36):
        sheet=Image.new('RGB',(1200,1320),'white')
        draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[start:start+36]):
            im=Image.open(r['image']).convert('RGB'); im.thumbnail((198,195))
            d=ImageDraw.Draw(im)
            for cls,x,y,w,h in r['boxes']:
                d.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline='#00ff40',width=1)
            x=(j%6)*200; y=(j//6)*220
            sheet.paste(im,(x,y+23))
            draw.text((x+3,y+3),f"{r['id']} {Path(r['image']).name.split('.rf.')[0][:20]}",fill='black')
        sheet.save(OUT/f'sheet_{start//36+1:02}.jpg',quality=95)
    (OUT/'inventory.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'{len(records)} training images; {(len(records)+35)//36} sheets; {OUT}')

if __name__=='__main__': main()
