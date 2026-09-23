"""Create a separate, verified dataset snapshot without training or changing MWPD."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'reports/deep_audit_20260915_220606'
SOURCE = ROOT / 'data/MWPD'
TARGET = ROOT / 'data/MWPD_reviewed_v1'
CONFIG = ROOT / 'data.reviewed-v1.yaml'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_list(name):
    return [Path(line).resolve() for line in (AUDIT / name).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    if TARGET.exists() or CONFIG.exists():
        raise FileExistsError('Version already exists; existing data will not be overwritten.')
    train = read_list('proposed_train_v2.txt')
    validation = read_list('proposed_validation_v2.txt')
    excluded = read_list('excluded_train_v2.txt')
    decisions = json.loads((AUDIT / 'label_review_decisions.json').read_text(encoding='utf-8'))
    review = {Path(row['image']).resolve(): row for row in decisions if row['decision'] != 'no_clear_issue'}
    held = [p for p in validation if p in review]
    active_val = [p for p in validation if p not in review]
    if len(held) != 11 or len(train) != 2149 or len(validation) != 260:
        raise RuntimeError('Audit input counts changed; review the plan before creating a snapshot.')
    if set(train) & set(excluded) or len(set(train)) != len(train):
        raise RuntimeError('Duplicate/conflicting train entries.')
    actual_train = set((SOURCE / 'train/images').glob('*'))
    actual_val = set((SOURCE / 'valid/images').glob('*'))
    if set(train) | set(excluded) != actual_train or set(validation) != actual_val:
        raise RuntimeError('Audit inventory no longer matches train/validation.')
    for split, paths in [('train',train+excluded), ('valid',validation)]:
        parent = (SOURCE/split/'images').resolve()
        for path in paths:
            if path.parent != parent or not path.is_file():
                raise RuntimeError(f'Unexpected source path: {path}')
            if not (SOURCE/split/'labels'/path.with_suffix('.txt').name).is_file():
                raise FileNotFoundError(f'Missing label: {path}')
    guarded = [ROOT/'data.local.yaml', ROOT/'runs/baseline-3/weights/best.pt']
    guards = {str(p):digest(p) for p in guarded}
    required = sum(p.stat().st_size for p in train+validation)
    if shutil.disk_usage(ROOT).free < required * 2 + 100_000_000:
        raise RuntimeError('Insufficient free space for a separate copy.')
    TARGET.mkdir()
    marker = TARGET/'INCOMPLETE'
    marker.write_text('Do not use until verification completes.',encoding='utf-8')
    records=[]
    for split, paths in [('train',train),('valid',active_val),('review_hold',held)]:
        for sub in ('images','labels'):(TARGET/split/sub).mkdir(parents=True,exist_ok=True)
        for index,src in enumerate(paths,1):
            src_label=src.parent.parent/'labels'/src.with_suffix('.txt').name
            # Decode validation only is not enough: verify every copied source image.
            with Image.open(src) as image:
                image.verify()
            for line in src_label.read_text(encoding='utf-8').splitlines():
                fields=list(map(float,line.split()))
                if len(fields)!=5 or not all(math.isfinite(v) for v in fields):
                    raise ValueError(f'Invalid label: {src_label}')
                cls,x,y,w,h=fields
                if cls!=0 or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-5 or max(x+w/2,y+h/2)>1.00001:
                    raise ValueError(f'Invalid label coordinates: {src_label}')
            for original,folder in [(src,'images'),(src_label,'labels')]:
                destination=TARGET/split/folder/original.name
                before=digest(original)
                shutil.copy2(original,destination)
                if digest(destination)!=before or digest(original)!=before:
                    raise RuntimeError(f'Copy/source verification failed: {original}')
                records.append(dict(source=str(original),destination=str(destination.relative_to(TARGET)),sha256=before))
            if index%400==0:print(f'{split}: {index}/{len(paths)} verified',flush=True)
    metadata=TARGET/'metadata';metadata.mkdir()
    write_json(metadata/'files.json',records)
    write_json(metadata/'review_hold.json',[review[p] for p in held])
    for filename in ['exclusion_reasons.json','plan_v2_summary.json','label_review_decisions.json']:
        shutil.copy2(AUDIT/filename,metadata/filename)
    (metadata/'excluded_train_sources.txt').write_text('\n'.join(str(p) for p in excluded)+'\n',encoding='utf-8')
    config=dict(path=TARGET.as_posix(),train='train/images',val='valid/images',names={0:'pothole'})
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),
        train_images=len(train),validation_images=len(active_val),review_hold_images=len(held),
        excluded_train_images=len(excluded),verified_files=len(records),
        copied_bytes=sum((TARGET/r['destination']).stat().st_size for r in records),
        labels_modified=False,test_accessed=False,training_started=False,
        independent_validation_guaranteed=False,source_guards=guards)
    # Verify the snapshot from its saved manifest, including image-label pairing.
    for row in records:
        if digest(TARGET/row['destination'])!=row['sha256']:
            raise RuntimeError('Saved manifest verification failed.')
    for split, expected in [('train',len(train)),('valid',len(active_val)),('review_hold',len(held))]:
        images=list((TARGET/split/'images').iterdir());labels=list((TARGET/split/'labels').iterdir())
        if len(images)!=expected or {p.stem for p in images}!={p.stem for p in labels}:
            raise RuntimeError(f'Image-label pairing failed: {split}')
    if any(digest(Path(p))!=sha for p,sha in guards.items()):
        raise RuntimeError('Original model/config changed during snapshot creation.')
    write_json(metadata/'summary.json',summary)
    config_text=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
    # Publish the new config only after all content checks pass.
    with CONFIG.open('x',encoding='utf-8') as f:f.write(config_text)
    (TARGET/'data.yaml').write_text(config_text,encoding='utf-8')
    (TARGET/'README.md').write_text(f'''# MWPD reviewed v1

Orijinal MWPD dosyalarından ayrı fiziksel kopya. Eğitim başlatılmadı.

- Train: {len(train)} görüntü ve etiket.
- Validation: {len(active_val)} görüntü ve etiket.
- İnceleme bekleyen: {len(held)} validation görüntüsü, review_hold/ altında.
- Dışlanan train adayları: {len(excluded)}; orijinal konumlarında korundu.

Etiket koordinatları değiştirilmedi. İnceleme bekleyen görüntüler aktif validation
listesinde değildir; negatif görüntüye çevrilmedi. Dosya adı çakışmaları ve
video sekansı için ihtiyatlı dışlama kuralları kullanıldı. Bu sürümün tüm
sahne benzerliklerinden arındırıldığı garanti edilmez. Aynı validation sahnesinin
birden fazla türevi hâlâ bulunabilir. Train etiketlerinin tamamı görsel olarak
incelenmedi; biçim ve dosya bütünlüğü kontrol edildi.

Önceki model yeni validation sahnelerinin bir kısmını görmüş olabilir. Bu sürümle
elde edilecek metrikleri eski validation puanlarıyla doğrudan kıyaslamayın.
Bağımsız deney için veri denetiminin tamamlanması ve MWPD eğitimi görmemiş
başlangıç ağırlıkları gerekir.

Test bölümü bu sürüme kopyalanmadı ve YAML'a eklenmedi. Orijinal data.local.yaml
değişmedi. Yeni yapılandırma: data.reviewed-v1.yaml (proje kökünde).
metadata/files.json her kopyanın kaynak yolunu ve doğrulanmış SHA256 değerini taşır.
''',encoding='utf-8')
    marker.unlink()
    (TARGET/'VERIFIED.json').write_text(json.dumps(dict(status='copy_integrity_verified',files=len(records)),indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(f'CONFIG={CONFIG}',flush=True)


if __name__=='__main__':main()
