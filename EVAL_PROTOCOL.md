# Validation referans protokolü

`eval.py` tek standart kullanır: Ultralytics 8.4.153 `model.val()`, MWPD
reviewed-v1 validation (249 görüntü, 545 kutu), 640 piksel, batch 1,
conf 0.001, NMS IoU 0.7, max_det 300, FP32 ve TTA kapalı. Kova tanısı aynı
değerlendirmenin kaydettiği tahminleri conf 0.25 ve eşleştirme IoU 0.5 ile
inceler. Kutu boyutu, normalize genişlik ve yüksekliğin geometrik ortalamasıdır.

| Ölçüm | Önceki referans | Yeni `model.val()` |
| --- | ---: | ---: |
| Precision | 0.718 | 0.726 |
| Recall | 0.662 | 0.660 |
| mAP50 | 0.701 | 0.716 |
| mAP50–95 | 0.309 | 0.331 |
| 5–10% recall | 0.550 | 0.550 (66/120) |

Önceki 0.701/0.309 sonucu arşivdeki `eval_p2_sahi.py` betiğinden gelir
(`1910663` commit'i). O betik SAHI `get_sliced_prediction` kullanır ve
tahminlere IoU 0.5 NMS birleştirmesi uygular. 640×640 görüntü tek pencere
olduğu için [SAHI 0.12.6](https://github.com/obss/sahi/blob/0.12.6/sahi/predict.py)
sürümünde `perform_standard_pred=True` seçeneği ikinci bir
tam-görüntü geçişi eklemez; yine de tahmin/son-işleme protokolü farklıdır.
Yeni betik doğrudan Ultralytics `model.val()` sonucunu kullanır. Yalnız
`model.val()` NMS eşik değerini 0.5'e çekmek de eski
sonucu üretmedi: P 0.758 / R 0.657 / mAP50 0.720 / mAP50–95 0.325.

Bu nedenle önceki SAHI skoru ile yeni standart ölçüm aynı seri olarak
karşılaştırılmamalıdır. Sonraki modeller karşılaştırılacaksa bu betik ve aynı
ayarlar tüm ağırlıklarda kullanılmalıdır. Eski sayıların birebir yeniden
üretilememesi kayıtlı bir protokol farkıdır; model kalitesinde değişim kanıtı
değildir. Test bölümü kullanılmadı.
