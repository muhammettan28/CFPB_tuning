# Ref [14] encoder hiperparametre ek deneyleri

Hazırlık ve dondurma tarihi: 21 Ekim 2026. Kullanıcı bu makaledeki baseline hiperparametrelerini denememizi açıkça istedi. Ana `gpu_revision_v1` protokolü değiştirilmedi; ek koşullar `configs/ref14_encoder_v1.json` ve SHA-256 dosyasında ayrı donduruldu. Sonuçların tamamlandığı bu hazırlık kaydından çıkarılamaz. [Güncel ek durum](REF14_BASELINE_STATUS.md).

Kaynak: Güllü ve arkadaşları, *Customer Complaint Classification with Large Language Models*, ACDSA 2025, DOI 10.1109/ACDSA65407.2025.11166338. Yerel PDF `literature/Customer_Complaint_Classification_with_Large_Language_Models.pdf`. PDF sayfa 3–4 görüntüleriyle tablolar kontrol edildi; kaynak dosyanın SHA-256 değeri konfigürasyonda saklandı.

| Kaynak tablo | Model | LR | Epoch | Batch | Girdi token sınırı | Optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| III, PDF s. 3 | bert-base-uncased | 2e-5 | 3 | 16 | 128 | AdamW |
| IV, PDF s. 4 | roberta-base | 5e-5 | 3 | 16 | 100 | AdamW |

Her iki LR/epoch çifti zaten ana altışar adaylık validation aramalarının kazananı. Ek koşulda fiziksel batch=16, accumulation=1 uygulanacak; ana koşul batch=4, accumulation=4 ve token cap=512. Böylece nominal etkin batch her ikisinde 16 kalsa da kısa token sınırı ve microbatch düzeni birlikte değişiyor. Farkın tek bir hiperparametreden kaynaklandığı ileri sürülmeyecek.

Her model seed 42/43/44 ile baştan üç kez eğitilecek. Aynı sabit 10.000 train ve 2.000 validation üyesi/sırası kullanılacak; nihai checkpoint üçüncü epoch sonu. Yeni grid/arama, test skoruna göre baseline değiştirme veya kaynak makalenin skorlarına sayısal üstünlük iddiası yok. Altı refit'in tamamı, ana encoder koşullarıyla birlikte raporlanacak.

Kaynakta belirtilmeyen veya burada korunan ayrıntılar açıkça ayrılıyor: 500-karakter narrative sınırı, düzeltilmiş veri bölünmesi, aynı sabit HF commit'leri, FP16 autocast, evaluation batch=16, padding multiple=8, weight decay=0,01, warmup=0,06 ve linear scheduler ana çalışmadan korunuyor. Optimizer `adamw_torch`, loss ağırlıksız cross-entropy. Kaynak Tablo IV RoBERTa için `return tensors=tf` verirken burada iki model de PyTorch ile çalışacak. Kaynağın 80/20 bölünmesi ve kendi preprocessing adımları yeniden yaratılmıyor. Bu nedenle çalışma **kaynak hiperparametrelerinden alınmış ek duyarlılık deneyi**, birebir replikasyon değil.

Daha kısa cap'ler için gerçek tokenizer anahtarları yeniden denetlendi: model başına 39.428 train/validation/internal/legacy girdisi, denetlenen beş split çifti arasında aynı görünür token girdisi çakışması sıfır. Internal ve legacy anahtarları frozen bootstrap gruplarını bölmüyor. [Hazırlık sonucu](analysis/ref14_encoder/PREPARATION.json). Semantik/parafraz bağımsızlığı garantisi değildir. Cap nedeniyle kırpılan satırlar ilgili input_audit JSON'larında ve satır bazlı gzip anahtar dosyalarında kayıtlı.

GPU işleri ana çalışmanın 88 aşaması ve sonuç analizi başarıyla tamamlanıp koordinatör kilidi bırakılınca başlayacak. Ayrı koordinatör aynı kilidi alacak; altı ek eğitimden önce hiçbir ek final tahmin üretmeyecek. Sonra altı evaluation işi: internal test ve 12 legacy görünümü. Her reload'da 2.000 validation tahmini Trainer çıktısıyla birebir eşleşmeden final test okunmayacak. Başarısız veya yarım çıktılar silinmez; inceleme için korunur.

Son CPU analizinde ana/ek encoder tahminleri ortak 10.000 paired frozen-group bootstrap çekimiyle karşılaştırılacak. Accuracy, macro-F1, weighted-F1, seed sample SD ve ek-ana paired fark aralıkları yazılacak. Legacy görünümlerinde ana analizdeki benzersiz normalize girdi ağırlıkları korunacak. Aralıklar eğitilmiş checkpoint'lere koşullu; training/model-selection belirsizliğini kapsamaz. Test daha önce incelenmiş olduğu için değerlendirme retrospektiftir.

Diğer kaynak tabloları da okundu: CNN (Tablo I) GloVe, 5 hidden layer, window 5, 128 neuron, 15 epoch, ReLU, Adam, LR 0,001; LSTM (Tablo II) GloVe, 3 hidden layer, window 100, 128 neuron, 15 epoch, tanh, Adam, LR 0,001. Metin GloVe için 100d der. LSTM paragrafındaki 5-layer ile tablodaki 3 hidden layer ifadeleri toplam/hidden katman sayımına göre bağdaşabilir; kesin katman dizilişi verilmez. GloVe release/corpus ve tüm mimari ayrıntıları da belirtilmez. Bu iki mimari mevcut encoder hiperparametre ek koşularına dahil edilmedi. Tablo V GPT-2/Llama için 3 epoch, 512 token, batch 8, LR 5e-5, AdamW, CrossEntropyLoss ve weighted average verir; bu tablo 4-bit QLoRA yapılandırması tanımlamaz. Bu ayarlar mevcut QLoRA protokolüne sessizce uygulanmadı.

Çalıştırıcılar: `scripts/run_ref14_baselines.py` ve `scripts/analyze_ref14_baselines.py`. Varsayılan komut yalnız konfigürasyonu gösterir; GPU çalıştırması `--execute --wait-for-core` ile açık. Yerel detached süreç başlatıldıktan sonra durum `REF14_BASELINE_STATUS.md` ve `analysis/ref14_encoder/RUN_STATUS.json` içinde izlenir. Tamamlanınca RESULTS_REPORT, RESULT_VALIDATION, revizyon günlüğü, hakem yanıt taslağı ve yerel reviewer paketi güncellenir.

## 24 Ekim 2026 — RoBERTa değerlendirme uygulama düzeltmesi

Ana çalışma ve altı ek refit tamamlandı. Ek RoBERTa evaluation, tokenizer aynı çağrıda max_length=100 ve padding multiple=8 istediği için final tahminlerden önce durdu. Uygulama düzeltildi: önce 100 gerçek tokena kırp, sonra eğitimdeki DataCollatorWithPadding gibi gerekirse 104 genişliğe pad et. Cap, görünür tokenlar, batch, precision, checkpoint veya veri değişmedi. Bu, dondurulmuş deney tasarımını değiştiren yeni bir ayar değildir.

39.428 girdinin tümünde kırpılmış token hashleri önceki audit ile aynı; mevcut 128/512 cap'lerinde 2.000 validation örneğinin bütün tensörleri önceki çağrıyla birebir aynı. [CPU düzeltme doğrulaması](analysis/ref14_encoder/PADDING_FIX_CPU_VALIDATION_2026-10-24.json). 35 CPU testi geçti. Başarısız dizin metadata/koduyla ayrı korundu; bitmiş eğitim veya ana deney tekrarlanmadı. Kalan üç RoBERTa değerlendirmesi kaydedilmiş checkpoint'lerden sürdürülüyor; her birinin tam validation parity kontrolü yine zorunlu. [Yeniden başlama kaydı](analysis/ref14_encoder/PADDING_FIX_RESTART_2026-10-24.json).
