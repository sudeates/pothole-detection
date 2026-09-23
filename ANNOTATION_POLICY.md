# Pothole etiketleme kuralı — çalışma sürümü

Amaç: yol yüzeyinde görünür malzeme kaybı veya belirgin çöküntü bulunan çukurları
tek sınıf `pothole` olarak tespit etmek. Görüntüden fiziksel derinlik ölçülmez.

1. **Tek kutu / tek çukur:** Bağlantılı hasar alanını sıkıca kapsayan bir kutu çiz.
   İki hasar arasında görünür sağlam yol şeridi varsa iki ayrı kutu kullan.
   Yan yana bütün çukurları kapsayan ek grup kutusu ekleme.
2. **Kutunun sınırı:** Görünür çukur kenarlarını kapsa; gölgeyi veya çevredeki
   sağlam asfaltı sırf koyu olduğu için kutuya katma. Aynı nesne için iç/dış iki kutu kullanma.
3. **Su:** Yalnızca su veya yansıma görmek çukur kanıtı değildir. Çukur kenarı veya
   malzeme kaybı seçiliyorsa etiketle. Belirsiz su birikintisini negatif ilan etme;
   görüntüyü incelemeye ayır.
4. **Yama, çatlak, renk farkı:** Malzeme kaybı/çöküntü görünmüyorsa bu sınıfa girmez.
   Bir görüntü negatif sayılmadan önce bütün yol alanında başka çukur olmadığı kontrol edilir.
5. **Küçük/uzak nesne:** Boyutu küçük diye silme. Çukur olduğuna ve sınırına karar
   verilemiyorsa inceleme durumunda bırak. Modelin tahmin etmemesi silme gerekçesi değildir.
6. **Kısmi görünürlük:** Görünür kısmın çukur olduğu açıksa sınırı görüntü kenarında
   kes; görünmeyen alanı tahmin ederek kutuyu genişletme.
7. **Belirsizlik:** Otomatik boş etiket üretme. Karar verilene kadar ayrı inceleme
   havuzu kullan. Mevcut 46 hold görüntüsünü bu kuralla yeniden değerlendirmeden eğitime katma.

Her düzeltme eski/yeni etiket, kaynak hash'i, gerekçe ve inceleme durumuyla yeni
bir sürüme kaydedilir. Önce eğitim örnekleri üzerinde tutarlılık kontrol edilir.
Validation etiketleri model sonucunu yükseltmek amacıyla değiştirilmez. Bağımsız
etiket denetimi zorunlu değişiklik gerektirirse yeni benchmark sürümü oluşturulur
ve bütün modeller yeniden aynı sürümde ölçülür; eski sonuçlarla karıştırılmaz.

Bu belge bir proje çalışma kuralıdır; mevcut 263 belirsiz görüntünün onaylandığını
veya etiketlerin uzman tarafından doğrulandığını ifade etmez.
