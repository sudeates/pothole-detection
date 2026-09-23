# HRP4K etiket korumalı parçalara ayırma

`tile_yolo_dataset.py`, HRP4K train görüntülerini varsayılan olarak 640×640
parçalara böler. Her eksende %20 örtüşme vardır. Kaynak görüntü daha küçükse
sabit boyuta dolgu yapılır. YOLO kutuları parça koordinatlarına çevrilir.
Bir kutunun özgün alanının **%30'undan fazlası** parçada kalıyorsa kesilmiş
kutuyla etiketlenir; %30 veya altıysa o parçada etiketlenmez. Örtüşme nedeniyle
aynı kaynak kutu birden fazla parçada bulunabilir.

Etiketsiz aday parçaların sabit seed ile yaklaşık %10'u tutulur; oran
olasılıksaldır, her görüntü için tam %10 değildir. Kısmen görünen fakat %30
eşiğinin altında kalan nesne, tutulan etiketsiz parçanın içinde bulunabilir.
Bu durum `metadata/summary.json` içinde belirtilir; kullanılmadan önce negatif
parçalar görsel olarak incelenmelidir.

Çıktıda `train/images`, `train/labels`, `valid/images`, `valid/labels` ve
`data.yaml` bulunur. Validation, MWPD V1'in orijinal 249 görüntüsünden aynen
kopyalanır; kırpılmaz. HRP4K'nin kendi valid/test bölümleri kullanılmaz.
`metadata/files.json` her çıktı için SHA-256 ve kaynak hash'i, ayrıca parça
koordinatını tutar. `metadata/source_hashes.json` kaynak eğitim dosyalarını,
`VERIFIED.json` tamamlanmış doğrulamayı kaydeder. İşlem yarıda kalırsa
`INCOMPLETE` dosyası kalır ve klasör eğitim için kullanılmamalıdır.

Tam veri üretimi:

```powershell
& .\venv\Scripts\python.exe .\tile_yolo_dataset.py
```

Parametre örneği:

```powershell
& .\venv\Scripts\python.exe .\tile_yolo_dataset.py --tile-size 640 --overlap 0.20 --min-box-fraction 0.30 --negative-fraction 0.10 --output .\data\HRP4K_tiled_custom
```

Mevcut hedef klasör asla üzerine yazılmaz. Betik eğitim başlatmaz. P2 mimarisi
ve karşılaştırma Görev 2 kapsamında, veri sonucu gözden geçirildikten sonra yapılır.
