"""Apply visually checked changes into an independent experimental dataset."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import shutil
import html
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/full_train_label_review'
SOURCE=ROOT/'data/MWPD_labels_v2_candidate'
TARGET=ROOT/'data/MWPD_reviewed_v2'
CONFIG=ROOT/'data.reviewed-v2.yaml'
ACCEPT_ADDITIONS={334,335,337,1608,1694,1695,1696,1773,1775,1790,1791,1792,1793,1794,1795,1799,1800,1801,1952,1953,1954}
HOLD_GROUPS={9,10,18,19,21,23,26,32,38,41,43,44,45,46,50,64}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def draw(path,label):
    im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im)
    for line in label.splitlines():
        _,x,y,w,h=map(float,line.split())
        d.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline='lime',width=2)
    return im

def main():
    if TARGET.exists() or CONFIG.exists():raise FileExistsError('Refusing to overwrite existing version.')
    inventory=json.loads((REPORT/'inventory.json').read_text())
    flags=json.loads((REPORT/'triage_flags.json').read_text(encoding='utf-8'))
    parent=json.loads((SOURCE/'metadata/files.json').read_text())
    additions=json.loads((REPORT/'missing_box_proposals/proposals.json').read_text())
    splits=json.loads((REPORT/'split_box_proposals/proposals.json').read_text())
    patches={p['target_id']:p for p in additions if p['target_id'] in ACCEPT_ADDITIONS}
    patches.update({p['target_id']:p for p in splits})
    if len(patches)!=24:raise ValueError('Expected 24 visually accepted corrections.')
    group_for={i:(g+1,row['reason']) for g,row in enumerate(flags) for i in row['ids']}
    held={i for i,(g,_) in group_for.items() if g in HOLD_GROUPS}
    if held & patches.keys():raise ValueError('Correction/hold overlap.')
    for row in parent:
        if sha(SOURCE/row['destination'])!=row['sha256']:raise ValueError('Parent manifest changed.')
    for i,p in patches.items():
        if sha(Path(p['label']))!=p['source_sha256']:raise ValueError('Correction source changed.')
        for line in Path(p['proposed_label']).read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            if not all(math.isfinite(v) for v in (cls,x,y,w,h)) or cls!=0 or w<=0 or h<=0 or min(x-w/2,y-h/2)<-1e-7 or max(x+w/2,y+h/2)>1+1e-7:
                raise ValueError('Invalid proposed YOLO box.')
    required=sum((SOURCE/r['destination']).stat().st_size for r in parent)
    if shutil.disk_usage(ROOT).free<required+100_000_000:raise RuntimeError('Insufficient disk space.')
    TARGET.mkdir();(TARGET/'metadata').mkdir();(TARGET/'INCOMPLETE').write_text('Verification pending.')
    name_to_id={Path(r['image']).stem:r['id'] for r in inventory}
    records=[];changes=[]
    for row in parent:
        rel=Path(row['destination']);source=SOURCE/rel
        i=name_to_id.get(rel.stem) if rel.parts[0]=='train' else None
        target_rel=Path('train_review_hold',*rel.parts[1:]) if i in held else rel
        dest=TARGET/target_rel;dest.parent.mkdir(exist_ok=True,parents=True)
        modified=i in patches and rel.parts[:2]==('train','labels')
        if modified:
            shutil.copyfile(patches[i]['proposed_label'],dest)
            old=source.read_text();new=dest.read_text()
            changes.append(dict(id=i,source=str(source),destination=target_rel.as_posix(),
                before=old,after=new,before_sha256=row['sha256'],after_sha256=sha(dest),
                reference_label=patches[i]['source_label'],registration=patches[i]['registration'],
                operation='replace grouped box with six boxes' if i in {273,274,275} else 'add missing boxes',
                review='image registration proposal followed by visual inspection; no model predictions'))
        else:shutil.copy2(source,dest)
        digest=sha(dest)
        if not modified and digest!=row['sha256']:raise RuntimeError('Copy hash mismatch.')
        records.append(dict(source=str(source),destination=target_rel.as_posix(),sha256=digest,source_sha256=row['sha256']))
    for row in parent:
        if sha(SOURCE/row['destination'])!=row['sha256']:raise RuntimeError('Parent altered during build.')
    decisions=[]
    for r in inventory:
        i=r['id'];group,reason=group_for.get(i,(None,'Önizleme taramasında belirgin sorun görülmedi; ayrıntılı doğruluk garantisi değildir.'))
        status='corrected' if i in patches else 'train_review_hold' if i in held else 'retained_with_note' if group else 'triage_no_obvious_issue'
        decisions.append(dict(id=i,image=r['image'],label=r['label'],status=status,group=group,reason=reason,
            review_level='contact_sheet_all_images; representative_detail_per_flag_group; accepted_corrections_visually_checked'))
    dump(REPORT/'decisions.json',decisions)
    dump(TARGET/'metadata/decisions.json',decisions)
    dump(TARGET/'metadata/files.json',records);dump(TARGET/'metadata/changes.json',changes)
    dump(TARGET/'metadata/provenance.json',dict(parent=str(SOURCE),original_v1=str(ROOT/'data/MWPD_reviewed_v1'),
        parent_metadata=str(SOURCE/'metadata'),original_exclusions=str(ROOT/'data/MWPD_reviewed_v1/metadata'),
        prior_duplicate_removals=3,negative_pilot_integrated=False))
    view=REPORT/'final_changes';view.mkdir(exist_ok=True)
    cards=[]
    for c in changes:
        r=inventory[c['id']-1]
        before=draw(r['image'],c['before']);after=draw(r['image'],c['after'])
        canvas=Image.new('RGB',(1280,675),'white');canvas.paste(before,(0,35));canvas.paste(after,(640,35))
        d=ImageDraw.Draw(canvas);d.text((10,10),f"ID {c['id']} | BEFORE",fill='black');d.text((650,10),'AFTER',fill='black')
        canvas.save(view/f'{c["id"]}.jpg',quality=94)
        cards.append(f'<article><h2>Görüntü {c["id"]}</h2><p>{html.escape(Path(r["image"]).name)}</p><img src="final_changes/{c["id"]}.jpg"></article>')
    (REPORT/'changes.html').write_text('<!doctype html><meta charset="utf-8"><title>Eğitim etiketi düzeltmeleri</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto}img{max-width:100%}article{padding:20px;border-bottom:1px solid #ccc}</style><h1>24 görüntüde uygulanan etiket düzeltmeleri</h1><p>Sol: önce. Sağ: yeni deneysel sürüm. Yalnız eğitim verisi değiştirildi.</p>'+''.join(cards),encoding='utf-8')
    (REPORT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Tüm eğitim taraması</title><h1>2.149 görüntü: ön tarama</h1><p>Küçük önizleme tam boyut etiket onayı değildir. Kararlar decisions.json dosyasında.</p>'+''.join(f'<h2>Sayfa {n}</h2><img loading="lazy" style="max-width:100%" src="sheet_{n:02}.jpg">' for n in range(1,61)),encoding='utf-8')
    CONFIG.write_text(f'path: {TARGET.as_posix()}\ntrain: train/images\nval: valid/images\nnames:\n  0: pothole\n',encoding='utf-8')
    net=sum(len(c['after'].splitlines())-len(c['before'].splitlines()) for c in changes)
    summary=dict(status='experimental dataset; not model-performance validated',train_images=2149-len(held),
        validation_images=249,train_review_hold=len(held),original_validation_review_hold=11,
        corrected_images=len(changes),missing_boxes_added=24,grouped_boxes_replaced=3,separate_boxes_for_groups=18,
        net_box_change_in_corrected_images=net,verified_files=len(records),source_unchanged=True,
        validation_unchanged=True,test_accessed=False,training_started=False,
        review_counts=dict(Counter(r['status'] for r in decisions)),negative_pilot_integrated=False)
    dump(TARGET/'VERIFIED.json',summary);dump(REPORT/'summary.json',summary)
    (TARGET/'INCOMPLETE').unlink()
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
