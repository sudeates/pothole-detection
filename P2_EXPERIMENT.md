# P2 + HRP4K crop deneyi

Model: [experiments/yolov8n-p2-pothole.yaml](experiments/yolov8n-p2-pothole.yaml).
Kurulu Ultralytics 8.4.153 içindeki `yolov8-p2.yaml` düzeninden uyarlanmıştır:
P2/4, P3/8, P4/16 ve P5/32. Tek sınıf `pothole`, nano ölçek, yaklaşık
2,93 milyon parametre. Yerel `yolov8n.pt` içindeki uyumlu ağırlıklar aktarılır;
yeni P2 başlığı sıfırdan öğrenilir. İlk kontrol 219/437 ağırlığın aktarıldığını
gösterdi. Ağ bağlantısı veya yeni ağırlık indirmesi gerekmez.

Eğitim verisi `data/HRP4K_tiled_v1/data.yaml`: 20.292 adet 640×640 train
parçası. Validation, kırpılmamış 249 MWPD görüntüsüdür. Test bölümü kullanılmaz.
Eğitim ayarları [experiments/p2_hrp4k_v1.yaml](experiments/p2_hrp4k_v1.yaml)
dosyasındadır: 100 epoch üst sınırı, 15 epoch early stopping, son 5 epoch
mosaic kapalı, `imgsz=640`, `batch=2`, `workers=0`, CUDA ve AMP. Ön kontrol
tüm veri dosyalarının SHA-256 özetlerini yeniden doğrular ve veri klasörüne
`.cache` yazılmasını engeller.

VS Code terminalinde, proje kökünden eğitimi başlat:

```powershell
& .\venv\Scripts\python.exe .\train_p2_hrp4k.py --start
```

`--start` olmadan yalnız ön kontrol çalışır. Beklenen ağırlık:
`runs/p2-hrp4k-v1/weights/best.pt`. 6 GB GPU'da bellek yetersizliği olursa
var olan deneme klasörünü koruyarak farklı adla batch 1 denemesi başlat:

```powershell
& .\venv\Scripts\python.exe .\train_p2_hrp4k.py --start --batch 1 --run-name p2-hrp4k-v1-b1
```

Eğitimden sonra aynı özgün validation üzerinde SAHI karşılaştırması:

```powershell
& .\venv\Scripts\python.exe .\eval_p2_sahi.py
```

Farklı run adıyla eğitildiyse `--p2-weights .\runs\p2-hrp4k-v1-b1\weights\best.pt`
ekle. Değerlendirme üç modelde de 640 pencere, %20 örtüşme, `batch_size=1`,
tam görüntü tahmini ve IoU=0,50 NMS birleştirmesi kullanır. `conf=0.001`,
precision ve recall için Ultralytics max-F1 noktası, mAP için IoU 0,50–0,95
eğrisi kullanılır. Sonuçlar `reports/p2_sahi_eval_*/summary.csv` dosyasına yazılır.

Mevcut n ve s modelleri MWPD V1 eğitiminden, P2 ise HRP4K crop eğitiminden
geliyor. Ortak validation ve inference doğrudan sonuç karşılaştırmasına izin
verir; farkı yalnız mimariye bağlayan kontrollü bir deney değildir.

P2 eğitiminden önce tamamlanan referans ölçümü:

| Model | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | %71,8 | %66,2 | %70,1 | %30,9 |
| YOLOv8s | %79,2 | %58,3 | %69,4 | %30,5 |

Bu ölçümün ham CSV dosyası `reports/p2_sahi_eval_20260923_164427_775779/summary.csv`.
