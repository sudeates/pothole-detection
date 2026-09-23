from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/full_train_label_review'

def main():
    records=json.loads((OUT/'inventory.json').read_text())
    flags=json.loads((OUT/'triage_flags.json').read_text(encoding='utf-8'))
    (OUT/'details').mkdir(exist_ok=True)
    for gi,g in enumerate(flags,1):
        r=records[g['ids'][0]-1]
        im=Image.open(r['image']).convert('RGB')
        d=ImageDraw.Draw(im)
        for j,(_,x,y,w,h) in enumerate(r['boxes'],1):
            box=((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height)
            d.rectangle(box,outline='lime',width=2); d.text((box[0]+2,box[1]+2),str(j),fill='black',stroke_width=1,stroke_fill='white')
        im.save(OUT/'details'/f'group_{gi:02}.jpg',quality=96)
    for start in range(0,len(flags),9):
        sheet=Image.new('RGB',(1200,1260),'white'); d=ImageDraw.Draw(sheet)
        for j,g in enumerate(flags[start:start+9]):
            gi=start+j+1; im=Image.open(OUT/'details'/f'group_{gi:02}.jpg'); im.thumbnail((395,390))
            x=(j%3)*400; y=(j//3)*420
            sheet.paste(im,(x,y+25)); d.text((x+4,y+4),f'Group {gi} | image {g["ids"][0]}',fill='black')
        sheet.save(OUT/f'detail_sheet_{start//9+1:02}.jpg',quality=95)
    print('70 representative detail images generated; per-image triage retained.')

if __name__=='__main__':main()
