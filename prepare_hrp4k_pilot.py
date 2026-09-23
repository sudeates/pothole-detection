"""Audit HRP4K train and prepare an isolated, unapproved 200-image pilot."""
from pathlib import Path
import json, random, shutil, hashlib, argparse
from datetime import datetime
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
SRC=Path(r'C:\Users\sdnra\Downloads\HRP4K\HRP4K')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    global SRC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SRC)
    SRC=parser.parse_args().source.resolve()
    out=ROOT/'data'/('HRP4K_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(); (out/'review').mkdir()
    records=[]; errors=[]; positives=[]; negatives=[]
    coco=json.loads((SRC/'train.json').read_text(encoding='utf-8'))
    counts={}
    for ann in coco['annotations']:
        counts[ann['image_id']]=counts.get(ann['image_id'],0)+1
    lookup={i['file_name']:i for i in coco['images']}
    missing_images=[name for name in lookup if not (SRC/'train/images'/name).is_file()]
    if missing_images:
        (out/'missing_images.json').write_text(json.dumps(missing_images,indent=2),encoding='utf-8')
        raise RuntimeError(f'{len(missing_images)} COCO train images missing; extraction incomplete. See {out}')
    for image in sorted((SRC/'train/images').iterdir()):
        label=SRC/'train/labels'/f'{image.stem}.txt'
        if not label.is_file():
            errors.append(f'Missing label {image.name}'); continue
        lines=[line.split() for line in label.read_text().splitlines() if line.strip()]
        try:
            for row in lines:
                assert len(row)==5 and float(row[0])==0
                x,y,w,h=map(float,row[1:])
                assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                assert x-w/2>=-0.001 and y-h/2>=-0.001 and x+w/2<=1.001 and y+h/2<=1.001
            if image.name not in lookup or counts.get(lookup[image.name]['id'],0)!=len(lines):
                errors.append(f'COCO/YOLO count mismatch {image.name}')
        except (ValueError,AssertionError): errors.append(f'Invalid label {image.name}')
        (positives if lines else negatives).append((image,label))
    report=dict(train_images=len(positives)+len(negatives),positive=len(positives),negative=len(negatives),
        errors=errors,source=str(SRC),test_read=False,validation_read=False,
        video_groups_available=False,warning='Numeric filenames: two-per-video sampling cannot be verified; random pilot only.')
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if errors: raise RuntimeError(f'Audit errors; see {out}')
    rng=random.Random(42)
    for kind,pool in [('positive',positives),('negative',negatives)]:
        selected=rng.sample(pool,100)
        for sub in ('images','labels'): (out/kind/sub).mkdir(parents=True)
        for index,(image,label) in enumerate(selected):
            for src,sub in [(image,'images'),(label,'labels')]:
                dest=out/kind/sub/src.name
                digest=sha(src); shutil.copy2(src,dest)
                assert sha(dest)==digest and sha(src)==digest
                records.append(dict(source=str(src),destination=dest.relative_to(out).as_posix(),sha256=digest))
            if index%25==0:
                sheet=Image.new('RGB',(1600,1100),'white')
            with Image.open(image) as im:
                im=im.convert('RGB'); im.thumbnail((320,190))
                x=(index%5)*320; y=((index%25)//5)*220
                sheet.paste(im,(x,y+25)); ImageDraw.Draw(sheet).text((x+3,y+4),f'{kind} {image.name}',fill='black')
            if index%25==24: sheet.save(out/'review'/f'{kind}_{index//25+1}.jpg')
    (out/'files.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (out/'README.md').write_text('Isolated review pilot, NOT training-ready. 100 positive and 100 source-labelled negative images.\nOnly official train used. No video identity available in numeric filenames; per-video cap not verified.\nContact sheets are preliminary review only: inspect uncertain and small objects at full resolution.\nNo training YAML generated. Source: https://doi.org/10.5281/zenodo.17522874 ; reported license CC BY 4.0.\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),**report)),flush=True)

if __name__=='__main__': main()
