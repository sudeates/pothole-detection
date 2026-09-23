# Yeni veri sürümü için referans eğitim

## Amaç

İncelenmiş veri sürümü için YOLOv8n/640 referans modeli oluşturmak. Önceki model
başlangıç olarak kullanılmaz. Yerel yolov8n.pt kaydında 80 sınıf ve coco.yaml
bulundu; bu dosya başlangıç ağırlığı olarak seçildi.

## Ayarlar

| Ayar | Değer |
|---|---|
| Veri | data.reviewed-v1.yaml |
| Train / validation | 2149 / 249 görüntü |
| Model / imgsz | YOLOv8n / 640 |
| Planlanan epoch | 100 |
| Eğitim batch / workers | 4 / 0 |
| GPU | 0, RTX 3050 6 GB |
| AMP | Açık |
| Seed | 42 |
| Optimizer | auto, önceki eğitimdeki gibi |
| Mosaic kapanışı | Son 10 epoch |
| Çıktı | runs/reviewed-v1-baseline_<benzersiz zaman>/ |

Batch=4 önceki eğitimde kullanılmıştı. Yeni betiğin ön kontrolü GPU bellek
yeterliliğini eğitim yaparak test etmez; bellek hatası olursa otomatik olarak
batch değiştirilmez, hata raporu kaydedilir. Yeni batch ayrı deney olarak ele alınır.

## Çalıştırma

Ön kontrol, eğitim yapmaz:

```powershell
.\venv\Scripts\python.exe .\train_reviewed.py
```

Eğitim başlatılması ayrıca istendiğinde kullanılacak komut:

```powershell
.\venv\Scripts\python.exe .\train_reviewed.py --start
```

İkinci komut henüz çalıştırılmadı. Varsayılan çağrı yalnız ön kontrol yapar.
Betik veri kopyasının tüm 4818 dosyasını SHA256 manifestiyle karşılaştırır,
eklenmiş/eksik dosyaları denetler, model kayıtlarını ve CUDA'yı kontrol eder,
Ultralytics'in yerel ayar doğrulamasını çalıştırır. Rapor reports/ altında tutulur.
Yeni deneyler önceki çıktıların üzerine yazmaz. Venv oluşturma/paket kurma yoktur.

## Sonuç nasıl değerlendirilecek?

- Eğitim boyunca yalnız yeni validation kullanılır; test yapılandırmada yoktur.
- Checkpoint seçiminde validation mAP50–95 ve hata örnekleri incelenir.
- Son değerlendirme 640, batch=1, workers=0 ile aynı validation üzerinde yapılmalı.
- Önceki 260 görüntülük validation puanıyla yeni 249 görüntülük puan doğrudan
  gelişim yüzdesi olarak sunulmamalı: bölüm ve örnek sayısı değişti.
- Eski model yeni validation sahnelerinin bazılarını önceden görmüş olabilir;
  eski/yeni modelin aynı validation puanı bile bağımsız genelleme kanıtı değildir.
- Veri sürümü tüm benzerliklerden arındırılmış değildir; kalan kaynak/sekans
  incelemesi ve 11 bekleyen etiket kararı kayıtlıdır.

Bu deney yeni sürüm için başlangıç referansı sağlar. Model boyutu, çözünürlük
ve artırma ayarları daha sonra birer değişken olarak karşılaştırılabilir.
