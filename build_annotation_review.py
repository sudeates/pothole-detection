"""Prepare a non-mutating review queue from existing training audit decisions."""
from pathlib import Path
import json
import html
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'reports'/'annotation_policy_review'
    out.mkdir(exist_ok=True)
    decisions=json.loads((ROOT/'data/MWPD_reviewed_v2/metadata/decisions.json').read_text(encoding='utf-8'))
    selected=[d for d in decisions if d['status'] in ('retained_with_note','train_review_hold')]
    groups={}
    for row in selected:
        groups.setdefault(row['group'],[]).append(row)
    cards=[]
    queue=[]
    for group,rows in sorted(groups.items(),key=lambda item: str(item[0])):
        for row in rows:
            split='train_review_hold' if row['status']=='train_review_hold' else 'train'
            path=ROOT/'data/MWPD_reviewed_v2'/split/'images'/Path(row['image']).name
            assert path.is_file(),path
            queue.append(dict(id=row['id'],group=group,image=str(path),status='pending_policy_review',
                current_split=split,reason=row['reason']))
        preview=ROOT/'reports/full_train_label_review/details'/f'group_{int(group):02d}.jpg'
        assert preview.is_file(),preview
        from base64 import b64encode
        image='data:image/jpeg;base64,'+b64encode(preview.read_bytes()).decode()
        links=' '.join(f'<a href="{html.escape(q["image"])}">{q["id"]}</a>' for q in queue if q['group']==group)
        cards.append(f'<article><h2>Grup {group}: {len(rows)} görüntü</h2><p>{html.escape(rows[0]["reason"])}</p><img loading="lazy" src="{image}"><p>Dosyalar: {links}</p></article>')
    (out/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf-8')
    doc='<!doctype html><meta charset="utf-8"><title>Çukur etiket incelemesi</title><style>body{font:18px sans-serif;max-width:1000px;margin:30px auto;background:#eee}article{background:white;padding:20px;margin:20px 0}img{max-width:100%}a{margin:6px}</style><h1>Etiket politikası inceleme kuyruğu</h1><p>Bu sayfa önceki eğitim taramasındaki belirsiz örnekleri birleştirir. Görseller önceki denetim önizlemeleridir. Yeni etiket doğrulaması veya düzeltmesi yapıldığı anlamına gelmez. Test/validation dahil değildir.</p>'+''.join(cards)
    (out/'index.html').write_text(doc,encoding='utf-8')
    print(json.dumps(dict(images=len(queue),groups=len(groups),output=str(out)),ensure_ascii=False))

if __name__=='__main__': main()
