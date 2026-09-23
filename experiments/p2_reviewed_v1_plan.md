# Kontrollü P2 deneyi: MWPD reviewed-v1

`p2_reviewed_v1.yaml`, `reviewed_v1_baseline.yaml` ile aynı eğitim/veri
ayarlarını taşır. Tek fark `model` değerinin P2–P5 başlıklı nano mimariye
çevrilmesi ve yerel `yolov8n.pt` ağırlıklarının
`initial_weights` ile yüklenmesidir. İki tarifin YAML sözlükleri bu iki alan
dışında programatik olarak karşılaştırıldı. Batch 4 korunur.
Modelin dört çıkış adımı `[4, 8, 16, 32]`, sınıf sayısı 1 olarak doğrulandı.

Ön kontrol tamamlandı: 2.149 train görüntüsü / 5.001 kutu, 249 validation
görüntüsü / 545 kutu, SHA-256 manifesti geçerli. Model yüklenirken
**219/437** uyumlu ağırlık aktarıldı. Yeni P2 katmanları eğitimde öğrenilir.
Smoke günlüğündeki ikinci `437/437` satırı hazırlanmış modelin eğiticiye
kopyalanmasıdır; COCO'dan aktarılan katman sayısı değildir.
Ön kontrol raporu:
`reports/preflight_p2_reviewed_v1_20260923_230715_339257/preflight.json`.

6 GB RTX 3050 üzerinde batch 4 ile bellek denemesi tamamlandı. Ayrı, geçici
tarifle `epochs=1`, `fraction=0.05`, `close_mosaic=0` kullanıldı. 107 train
görüntüsü ve tam 249 validation görüntüsü işlendi; OOM olmadı. Ultralytics
eğitim günlüğündeki en yüksek `GPU_mem` yaklaşık **0,873 GB** idi. Smoke
çıktısı:
`runs/p2_reviewed_v1_smoke_batch4_20260923_230321_310791/`.
Tek epoch'un mAP değeri model kalitesi karşılaştırması için kullanılmaz.
Smoke sonrasında tam tarifin ön kontrolü ve veri SHA-256 doğrulaması yeniden geçti.

VS Code terminalinde tam eğitimi başlatma komutu:

```powershell
& .\venv\Scripts\python.exe .\train.py --config .\experiments\p2_reviewed_v1.yaml --start
```

Çıktı `runs/p2_reviewed_v1_<zaman>/weights/best.pt` olur. Eğitim henüz
başlatılmadı. Tam eğitimden sonra yalnız validation üzerinde, aynı `eval.py`
protokolüyle baseline'la karşılaştır:

```powershell
& .\venv\Scripts\python.exe .\eval.py --weights .\runs\reviewed-v1-baseline_20260915_223719_077806\weights\best.pt --weights .\runs\p2_reviewed_v1_<zaman>\weights\best.pt
```

Yeni referans: baseline mAP50 **0,716**, mAP50–95 **0,331**, 5–10% kovasında
recall **0,550**. P2 başarısı, aynı validation'da 5–10% recall artışı ve
toplam mAP50'de anlamlı artışla değerlendirilecektir. Test split kullanılmaz.
