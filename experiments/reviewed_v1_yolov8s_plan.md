# YOLOv8s–640 kapasite karşılaştırması

Kullanıcı onayıyla başlatılan deney. Önceki reviewed-v1 YOLOv8n deneyinden
yalnız model değiştirilir: yerel COCO başlangıç ağırlığı weights/yolov8s.pt.
Ağırlık Ultralytics'in resmi assets v8.4.0 sürümünden indirildi; 80 sınıf
ve coco.yaml kayıtları doğrulandı. Venv ve paket sürümleri değiştirilmedi.

Veri: aynı 2149 train / 249 validation. Ayarlar: imgsz=640, epochs=100,
batch=4, workers=0, seed=42, AMP açık ve optimizer=auto.
Diğer ayarlar reviewed_v1_baseline.yaml ile aynıdır.

Çalıştırma:

```powershell
.\venv\Scripts\python.exe .\train_reviewed.py --recipe yolov8s --start
```

--start olmadan yalnız ön kontrol yapılır. Çıktı adı reviewed-v1-yolov8s
ön ekiyle ayrı ve benzersiz bir runs klasörüdür. Bellek hatasında otomatik
batch değişimi yapılmaz. Eğitim günlüğü reports/yolov8s_training_20260916_122232.log
dosyasına yazılır. Tamamlanma veya hata işareti ön kontrol rapor klasöründedir.

Karşılaştırma modeli:
runs/reviewed-v1-baseline_20260915_223719_077806/weights/best.pt.
Eğitim sonrası iki model aynı validation üzerinde 640, batch=1, workers=0
ile değerlendirilmeli. Recall, precision, mAP50–95 ve çıkarım süresi birlikte
raporlanmalı. Test bölümü bu kapasite deneyi için kullanılmaz.
Veri sürümünün önceki denetimde belirtilen kalan benzerlik/etiket sınırlamaları
iki model için de geçerlidir; daha büyük modelin daha başarılı olması garanti değildir.
