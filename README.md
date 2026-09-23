# Yerel pothole eğitimi

GPU destekli mevcut venv ile bu klasörde çalıştırın:

```powershell
python train.py
```

Varsayılan çalışma bir epoch sürer. İlk çalışmada YOLOv8n ağırlıkları indirilebilir.
Gereken paketler: CUDA destekli torch, eşleşen torchvision, ultralytics, pyyaml.
Betik Kaggle önbelleğini proje içindeki data/MWPD klasörüne kopyalar;
orijinal veriyi değiştirmez. Kopya ek disk alanı kullanır.
Colab yollarını yerel yollarla değiştiren data.local.yaml dosyasını üretir.

Kısa deneme başarılı olduktan sonra:

```powershell
python train.py --epochs 100
```

Bellek yetersizse --batch 2 veya --batch 1 kullanın.
Sonuçlar runs klasöründedir. Yeni çalışmalar mevcut deneyin üzerine yazılmaz.
Hazır veri bölümleri korunur; etiketlerin tam doğrulaması ve bölümler arası
benzer görüntü kontrolü henüz yapılmamıştır.

## Validation çözünürlük karşılaştırması

```powershell
.\venv\Scripts\python.exe .\compare_validation.py
```

`runs/baseline-3/weights/best.pt` ve `data.local.yaml` kullanılarak yalnızca
`val` bölümü 640, 960 ve 1280 boyutlarında değerlendirilir. Eğitim başlatılmaz.
Her deney ayrı Python sürecinde, GPU 0 üzerinde `batch=1`, `workers=0`,
FP32, `conf=0.001`, `iou=0.7`, `max_det=300` ve `rect=True` ile çalışır.
`imgsz` hedef boyuttur; dikdörtgen görüntüler en-boy oranı korunarak hazırlanır.

Her çalıştırma benzersiz `reports/val_resolution_<zaman>/` klasörü oluşturur.
`comparison.csv` metrikleri 0–1 ölçeğinde ve model çıkarım süresini ms/görüntü
olarak içerir. Süre Ultralytics'in `metrics.speed['inference']` ölçümüdür;
ön işleme, son işleme ve disk okumasını içermez. Tek geçiş süreleri GPU yükü
ve sıcaklığına göre değişebilir.

Her çözünürlük klasöründe günlük, sonuç JSON'u ve validation grafikleri bulunur.
Bellek yetersizliği `oom`, diğer hatalar `error` olarak CSV'ye yazılır ve sonraki
çözünürlüğe geçilir. `environment.json` ortamı, ağırlık SHA256 değerini ve eksik
etiket kontrolünü kaydeder. Veri etiket önbelleği diske yazılmaz; mevcut model,
veri ve venv değiştirilmez. Test bölümü bu karşılaştırmada değerlendirilmez.

## İncelenmiş veri sürümü

`data/MWPD_reviewed_v1/`, orijinal MWPD'den ayrı kopyalardan oluşur.
Yapılandırması `data.reviewed-v1.yaml` dosyasıdır. Bu sürümde 2149 train,
249 aktif validation ve `review_hold/` altında 11 inceleme bekleyen görüntü vardır.
581 train görüntüsü aday dışlama listesinde kalır; orijinal dosyaları korunur.
Etiket koordinatları değiştirilmemiştir. Test bölümü yeni yapılandırmaya eklenmez.

`prepare_reviewed_dataset.py` sürümü oluşturur ve SHA256 ile doğrular;
mevcut sürümün üzerine yazmayı reddeder. Kopya bütünlüğü sonucu `VERIFIED.json`,
kaynak/kopya kayıtları `metadata/files.json` içindedir. Kopya doğrulaması,
tüm sahne benzerliklerinin veya etiket hatalarının giderildiği anlamına gelmez.

Mevcut `train.py` ve `compare_validation.py` orijinal `data.local.yaml`
ile çalışır; yeni sürüme otomatik geçmez. Yeni eğitim henüz başlatılmadı.

Yeni sürüm için hazırlanan `train_reviewed.py`, varsayılan olarak yalnız veri,
model ve ayar kontrolü yapar. Eğitim ancak `--start` ile başlar.
Ayarlar `experiments/reviewed_v1_baseline.yaml`, deney açıklaması
`experiments/reviewed_v1_plan.md` içindedir.

YOLOv8s kapasite deneyi için `--recipe yolov8s` seçilir; eğitim için ayrıca
`--start` gerekir. Bu seçeneğin ayarları `experiments/reviewed_v1_yolov8s.yaml`
dosyasındadır. Varsayılan tarif hâlâ önceki YOLOv8n tarifidir.
