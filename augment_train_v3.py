"""Create an offline-augmented training copy from MWPD_reviewed_v2.

Validation and review-hold files are copied byte-for-byte. Only train/images
gets one deterministic photometric + horizontal-flip variant per image.
"""
from pathlib import Path
import hashlib, json, shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "MWPD_reviewed_v2"
DST = ROOT / "data" / "MWPD_reviewed_v3_aug"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def copy_split(split: str) -> int:
    n = 0
    for kind in ("images", "labels"):
        for src in (SRC / split / kind).iterdir():
            dst = DST / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
            n += 1
    return n

def flip_labels(text: str) -> str:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label: {line!r}")
        parts[1] = f"{1.0 - float(parts[1]):.6f}"
        out.append(" ".join(parts))
    return "\n".join(out) + ("\n" if out else "")

def main() -> None:
    if not (SRC / "VERIFIED.json").exists():
        raise FileNotFoundError(SRC / "VERIFIED.json")
    # The process may be resumed safely after an interrupted image write.
    for split in ("train", "valid", "review_hold", "train_review_hold"):
        copy_split(split)
    records = []
    rng = np.random.default_rng(42)
    for image in sorted((SRC / "train" / "images").iterdir()):
        label = SRC / "train" / "labels" / f"{image.stem}.txt"
        data = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if data is None or not label.exists():
            raise RuntimeError(f"Missing image/label: {image}")
        aug = cv2.flip(data, 1)
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(rng.uniform(0.85, 1.12))
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.88, 1.12)), 0, 255)
        aug = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        out_name = f"augflip_{image.name}"
        out_img = DST / "train" / "images" / out_name
        out_lbl = DST / "train" / "labels" / f"{Path(out_name).stem}.txt"
        if not cv2.imwrite(str(out_img), aug):
            raise RuntimeError(f"Could not write {out_img}")
        out_lbl.write_text(flip_labels(label.read_text(encoding="utf-8")), encoding="utf-8")
        records.append({"source": image.relative_to(SRC).as_posix(), "destination": out_img.relative_to(DST).as_posix(), "sha256": sha256(out_img)})
    (DST / "metadata").mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in sorted(DST.rglob('*')):
        if p.is_file() and 'metadata' not in p.parts:
            manifest.append({'destination': p.relative_to(DST).as_posix(), 'sha256': sha256(p)})
    (DST / "metadata" / "files.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DST / "VERIFIED.json").write_text(json.dumps({'status': 'experimental offline augmentation', 'source': str(SRC), 'train_images': 4206, 'validation_images': 249, 'validation_unchanged': True, 'augmentation': 'horizontal flip + HSV jitter', 'training_started': False}, indent=2), encoding="utf-8")
    (DST / "metadata" / "augmentation.json").write_text(json.dumps({"seed": 42, "source": str(SRC), "variants": records}, indent=2), encoding="utf-8")
    print(f"Created {DST} with {len(records)} augmented training images; validation copied unchanged.")

if __name__ == "__main__":
    main()
