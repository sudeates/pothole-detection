# Pothole detection

Windows, Python 3.11, CUDA destekli mevcut `venv` ve Ultralytics 8.4.x ile
çalışır. Sanal ortamı yeniden kurmayın; veri ve mevcut `runs/` kayıtlarını
koruyun. Ana veri `data/MWPD_reviewed_v1`, yapılandırması
`data.reviewed-v1.yaml` dosyasıdır. `test` bölümü model seçimi için kullanılmaz.

## Kurulum kontrolü

Proje kökünde PowerShell/VS Code terminali açın:

```powershell
& .\venv\Scripts\python.exe -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"
```

## Eğitim

Ön kontrol veri yollarını ve etiket sayılarını doğrular, varsa SHA-256 veri
manifestini denetler ve modeli yükler; eğitim başlatmaz:

```powershell
& .\venv\Scripts\python.exe .\train.py --config .\experiments\reviewed_v1_baseline.yaml
```

Eğitimi aynı komuta `--start` ekleyerek başlatın. Çıktı, config adı ve zaman
damgasıyla benzersiz bir `runs/` klasörüne yazılır. YOLOv8s için
`experiments/reviewed_v1_yolov8s.yaml` kullanılabilir.
P2 tarifi ve bellek denemesi [P2 deney notunda](experiments/p2_reviewed_v1_plan.md).

## Validation değerlendirmesi

```powershell
& .\venv\Scripts\python.exe .\eval.py --weights .\runs\reviewed-v1-baseline_20260915_223719_077806\weights\best.pt
```

Birden fazla modeli karşılaştırmak için `--weights` tekrar verilir. Varsayılan
veri `data.reviewed-v1.yaml`, bölüm `val` ve çözünürlük 640'tır. P/R/mAP
Ultralytics `model.val()` ile, boyut kovaları ve TP IoU tanısı `conf=0.25`,
eşleştirme `IoU=0.5` ile hesaplanır. Sonuçlar
`reports/eval_<run>_<zaman>/summary.json` ve `summary.md` dosyalarındadır.
Son modelin test değerlendirmesi yalnız `evaluate_reviewed_test.py` ile yapılır.
Önceki SAHI skoruyla `model.val()` skoru arasındaki fark [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)
dosyasında açıklanır.
