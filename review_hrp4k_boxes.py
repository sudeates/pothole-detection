"""Render existing pilot labels for inspection without changing data."""
from pathlib import Path
from PIL import Image, ImageDraw
P=Path(__file__).resolve().parent/'data/HRP4K_pilot_20260919_152449'
def main():
    images=sorted((P/'positive/images').glob('*.jpg'))
    for start in range(0,len(images),10):
        sheet=Image.new('RGB',(1800,1600),'white')
        for i,p in enumerate(images[start:start+10]):
            with Image.open(p) as src:
                im=src.convert('RGB'); w,h=im.size
            boxes=[]
            for line in (P/'positive/labels'/f'{p.stem}.txt').read_text().splitlines():
                _,x,y,bw,bh=map(float,line.split())
                boxes.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
            im.thumbnail((600,300)); d=ImageDraw.Draw(im)
            for b in boxes: d.rectangle(tuple(v*im.width/w for v in b),outline='lime',width=2)
            x=(i%2)*900;y=(i//2)*320
            sheet.paste(im,(x,y+20));ImageDraw.Draw(sheet).text((x,y),f'{p.name} boxes={len(boxes)}',fill='black')
            # Enlarged context around the smallest labelled object.
            b=min(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            side=max(160,(b[2]-b[0])*1.6,(b[3]-b[1])*1.6)
            with Image.open(p) as src:
                crop=src.crop((max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(w,int(cx+side/2)),min(h,int(cy+side/2))))
                crop.thumbnail((290,280));sheet.paste(crop,(x+605,y+25))
        sheet.save(P/'review'/f'boxes_{start//10+1:02}.jpg')
    print('Rendered 100 labelled images and smallest-object crops.')
if __name__=='__main__':main()
