"""Build a separate three-label correction candidate; does not launch training."""
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/MWPD_reviewed_v1'
TARGET = ROOT / 'data/MWPD_labels_v2_candidate'
REVIEW = ROOT / 'reports/train_label_review_20260916_144315'
CONFIG = ROOT / 'data.labels-v2-candidate.yaml'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def draw_labels(image_path, text):
    im = Image.open(image_path).convert('RGB')
    im.thumbnail((640,640))
    draw = ImageDraw.Draw(im)
    for i,line in enumerate(text.splitlines(),1):
        _,x,y,w,h = map(float,line.split())
        box = [(x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height]
        draw.rectangle(box, outline='lime', width=2)
        draw.text((box[0]+2,box[1]+2),str(i),fill='black',stroke_width=2,stroke_fill='white')
    return im

def main():
    if TARGET.exists() or CONFIG.exists():
        raise FileExistsError('Candidate already exists; refusing to overwrite.')
    manifest = json.loads((SOURCE/'metadata/files.json').read_text(encoding='utf-8'))
    changes = json.loads((REVIEW/'proposed_changes.json').read_text(encoding='utf-8-sig'))
    if len(changes) != 3:
        raise ValueError('Expected exactly three reviewed proposals.')
    patches = {}
    for change in changes:
        original = Path(change['source'])
        relative = original.relative_to(SOURCE)
        if relative.parts[:2] != ('train','labels') or sha(original).lower() != change['source_sha256'].lower():
            raise ValueError('Unexpected or modified source label.')
        lines = original.read_text().splitlines()
        proposed = (ROOT/change['proposal']).read_text().splitlines()
        if proposed != lines[:8]+lines[9:]:
            raise ValueError('Proposal contains changes beyond removing line 9.')
        patches[relative.as_posix()] = '\n'.join(proposed)+'\n'
    for row in manifest:
        p = SOURCE/row['destination']
        if sha(p) != row['sha256']:
            raise ValueError(f'Source manifest mismatch: {p}')
    required = sum((SOURCE/r['destination']).stat().st_size for r in manifest)
    if shutil.disk_usage(ROOT).free < required + 100_000_000:
        raise RuntimeError('Insufficient disk space.')
    TARGET.mkdir()
    (TARGET/'INCOMPLETE').write_text('Candidate not verified yet.')
    (TARGET/'metadata').mkdir()
    records=[]; changed=[]
    for row in manifest:
        rel = Path(row['destination']); src = SOURCE/rel; dst = TARGET/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        key = rel.as_posix()
        if key in patches:
            dst.write_text(patches[key],encoding='utf-8')
            changed.append(key)
        else:
            shutil.copy2(src,dst)
        actual = sha(dst)
        if key not in patches and actual != row['sha256']:
            raise RuntimeError(f'Copy verification failed: {rel}')
        records.append(dict(source=str(src),destination=key,sha256=actual,source_sha256=row['sha256']))
    if set(changed) != set(patches):
        raise RuntimeError('Not all patches were applied.')
    for row in manifest:
        if sha(SOURCE/row['destination']) != row['sha256']:
            raise RuntimeError('Source changed during build.')
    dump(TARGET/'metadata/files.json',records)
    dump(TARGET/'metadata/changes.json',dict(changed=changed,removed_line_1based=9,retained_line_1based=8,
        rationale='Near-duplicate annotation on same distant region; boundary remains uncertain.',
        status='candidate, not evaluated',parent=str(SOURCE),parent_metadata=str(SOURCE/'metadata')))
    cards=[]
    for i,key in enumerate(changed,1):
        image = SOURCE/'train/images'/Path(key).with_suffix('.jpg').name
        before=draw_labels(image,(SOURCE/key).read_text())
        after=draw_labels(image,(TARGET/key).read_text())
        sheet=Image.new('RGB',(1280,675),'white')
        sheet.paste(before,(0,35)); sheet.paste(after,(640,35))
        draw=ImageDraw.Draw(sheet); draw.text((10,10),'ONCE - 11 kutu',fill='black'); draw.text((650,10),'ADAY - 10 kutu (eski 9. satir kaldirildi)',fill='black')
        sheet.save(REVIEW/f'correction_{i}.jpg',quality=95)
        cards.append(f'<h2>{Path(key).name}</h2><img src="correction_{i}.jpg" style="max-width:100%">')
    (REVIEW/'corrections.html').write_text('<!doctype html><meta charset="utf-8"><title>Etiket düzeltme adayı</title><h1>Üç tekrar kutusunun kaldırılması</h1><p>Yalnız aday sürüm değişti. Uzak bölgedeki kesin sınır hâlâ belirsiz; başarım ölçülmedi.</p>'+''.join(cards),encoding='utf-8')
    CONFIG.write_text(f'path: {TARGET.as_posix()}\ntrain: train/images\nval: valid/images\nnames:\n  0: pothole\n',encoding='utf-8')
    summary=dict(status='verified copy; experimental label candidate',verified_files=len(records),
        changed_labels=len(changed),removed_boxes=3,train_images=2149,validation_images=249,
        validation_unchanged=True,source_unchanged=True,test_accessed=False,training_started=False)
    dump(TARGET/'VERIFIED.json',summary)
    (TARGET/'INCOMPLETE').unlink()
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
