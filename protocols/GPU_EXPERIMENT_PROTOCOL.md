# GPU deneyleri için önceden hazırlanmış protokol

**21 Ekim 2026 — çalıştırma güncellemesi:** Kullanıcı GPU deneylerini yetkilendirdi. Beş ağırlık seti ve tokenizer audit'i tam; ayrı GPU ortamı kuruldu ve beş küçük hardware/API smoke testi geçti. Tam çalışma devam ediyor: [canlı durum](GPU_RUN_STATUS.md), [ortam](analysis/gpu_execution/GPU_ENVIRONMENT.json), [yürütme ayrıntıları](analysis/gpu_execution/EXECUTION_NOTES_2026-10-21.md). Aşağıdaki 20 Ekim hazırlık kayıtları korunuyor; bugünkü durum için bu ek ve canlı kayıt esas alınmalı.

20 Ekim 2026. **Hazırlık CPU'da yapıldı; bu dosya GPU deneylerinin çalıştığını söylemez.** GPU'nun boşaldığını kullanıcı bildirmeden `--execute-gpu` kullanılmayacak. Orijinal veri protokolü, ORJ_PAPER ve eski eğitim kodu değiştirilmedi.

**İndirme kontrolü güncellemesi — 20 Ekim 2026:** Yeni koşular için beş gerçek HF commit kimliği kaydedildi; beş tokenizer'ın tam CPU girdi denetimi geçti. Ağırlıkları tamam olan tek model DistilBERT; diğer dördünde toplam 33,38 GB eksik. [Kontrol raporu](analysis/gpu_preparation/MODEL_DOWNLOAD_VALIDATION.json), [tamamlama script'i](scripts/complete_model_downloads.sh). Aşağıdaki tarihsel “henüz yapılmadı” ifadeleri ilk hazırlık anını anlatır; güncel tokenizer durumu bu ek ve `input_audit_*.json` kayıtlarıdır. GPU/CUDA/eğitim aşamaları hâlâ çalıştırılmadı.

Makinece okunur plan: [gpu_revision_v1.json](configs/gpu_revision_v1.json), [SHA-256](configs/gpu_revision_v1.sha256). Üst protokol: [revision_v1/protocol.json](analysis/revision_v1/protocol.json), SHA-256 `89fd382208515e946ab2a941d07809cec8b9fc659d554048f008cf2dea225d0f`. Üst protokoldeki `gpu_allowed: false` CPU aşamasının tarihsel kaydı olarak korunur; sonraki açık GPU çalıştırma komutu bunu sessizce değiştirmez.

## 1. Veri üyeliği ve seçim sınırı

Bütün revize QLoRA ve encoder eğitimleri **aynı** `data/revision_v1/train_10000.csv` ve `validation_2000.csv` dosyalarını kullanır. Her dosyanın tamamı, dosyadaki sıra ve kimliklerle yüklenir. Optimizasyon seed'leri 42, 43, 44; yeniden örnekleme yok. Eğitim batch sırası seed'e göre değişebilir. Dosya hash'leri her gerçek çalıştırmadan önce doğrulanır; gerçek seçilmiş ID manifestleri ve sıra hash'leri çıktıya yazılır.

Eğitim ve validation araması internal/legacy CSV'lerini açmaz. CPU girdi denetimi bunları sızıntı kontrolü için okur; test metriklerine göre seçim yapmaz. Final evaluation ayrı bir komuttur. Test daha önce incelendiğinden yeni/görülmemiş test olarak tanımlanmaz.

## 2. Model kimliği, tokenizer ve ortam

[Model registry şablonu](configs/model_registry.template.json) bilerek boş commit/snapshot alanları içerir. **Gerçek checkpoint kimlikleri henüz çözülmedi.** Gelecekte mevcut yerel HF cache'indeki `snapshots/<40-karakter-commit>` yolları `configs/model_registry.local.json` içine kaydedilir. Yalnız model adı veya `main` yeterli değildir. Planlanan Llama varyantı eski loglarda görülen **Llama-3.1-8B-Instruct**'tır. Yeni pin, eski run'ın eksik commit kanıtını geriye dönük tamamlamaz.

Çalıştırıcı model indirmez; tokenizer ve ağırlıkları `local_files_only=True` ile okur. Her gerçek çalıştırmada snapshot dosyalarının hash'leri, kod hash'leri, Python/kütüphane sürümleri, `pip freeze`, GPU/driver/CUDA bilgileri kaydedilir. Encoder fit edilen ağırlıklarının ve QLoRA adapter'larının hash'leri de alınır; sonradan değişmiş modelle değerlendirme reddedilir.

Önce gerçek tokenizer ile **CPU girdi denetimi** çalıştırılmalıdır:

```bash
python REVISION_PAPER/scripts/run_revision_gpu.py audit --model qwen --registry REVISION_PAPER/configs/model_registry.local.json
```

Denetim train/validation/internal/legacy arasında modelin gördüğü token kimliklerini, labelsız olarak karşılaştırır. Yeni encoder tokenizer'ları da denetlenir. LLM'de hedef JSON/EOS ve completion mask sınırı kontrol edilir; eğitim hedefi 512 tokena sığmıyorsa çalıştırma durur. Uzun koşulların test girdileri de gerçek eğitim/validation girdi anahtarlarıyla karşılaştırılır. Çakışma varsa veri sessizce değiştirilmez; tarihli protokol değişikliği gerekir. Başarılı bir audit dosyası olmadan GPU aşaması başlayamaz.

Chat template bir kez render edilir; tokenization `add_special_tokens=False` ile yapılır. Böylece template'in eklediği BOS tekrar eklenmez. QLoRA, aynı prompt prefix'i içeren açık `input_ids` ve `completion_mask` ile eğitilir; yalnız JSON/EOS hedefi loss'a girer. Bu, **yeni koşunun açık politikasıdır**; eski bütün koşuların aynı şekilde çalıştığı iddiası değildir. Tokenizer audit'i semantik/parafraz sızıntısını garanti etmez.

API hedefi Transformers 4.57.6, TRL 0.23.1, PEFT 0.17.1. Çalıştırmada yalnız bir GPU görünür olmalı (`CUDA_VISIBLE_DEVICES`); birden fazla GPU görünürse efektif batch değişimini önlemek için işlem durur. Gerçek GPU ortamı henüz kurulup test edilmedi. Eski V100/CUDA paketlerinin yeni masaüstüyle uyumlu olduğu varsayılmayacak; GPU açılınca donanıma uygun PyTorch/CUDA kurulumu ayrı kaydedilecek. CPU hazırlığı CUDA uyumluluğu veya eğitim smoke testi değildir.

## 3. QLoRA tekrarları

Qwen3-8B ve Llama-3.1-8B-Instruct, ayrı ayrı seed 42/43/44. Eski açık konfigürasyon korunur: NF4/double quantization/FP16, LoRA rank 16/alpha 32/dropout 0,05/all-linear; iki epoch; LR 2e-4; batch 1 × accumulation 16; paged AdamW 8-bit; cosine; warmup 0,03; non-reentrant gradient checkpointing. İlk 500 karakter, toplam eğitim dizisi en fazla 512 token; packing yok. Her 100 adımda validation loss; bu ölçüte göre en iyi checkpoint saklanır.

Bu aşama yeni bir QLoRA hiperparametre araması olarak sunulmaz: tarihsel ayarlar sabit, seçim validation checkpoint'iyle sınırlı. Test skoruna bakılarak koşu/seed elenmez. Her üç seed'in sonucu saklanır.

## 4. Encoder validation araması ve tekrarlar

BERT-base, DistilBERT, RoBERTa-base için **model başına altı aday**: LR {2e-5, 3e-5, 5e-5} × epoch {2, 3}. Arama seed 42; aynı 10k train/2k validation. Batch 4 × accumulation 4; eval batch 16; AdamW, weight decay 0,01, linear scheduler, warmup 0,06, FP16; metin 500 karakter ve 512 token. Her aday yalnız belirlenen son epoch'ta validation'da ölçülür; ara epoch seçimi yapılmaz.

Seçim ölçütü **validation macro-F1**. En iyi skordan en fazla 1e-12 uzakta olan adaylarda en küçük candidate ID seçilir. Altı aday tamamlanmadan kazanan dondurulamaz. Kazanan JSON/SHA dosyası final refit'ten önce yazılır. Seçilen konfigürasyonla seed 42/43/44 için üç gerçek refit yapılır; sınıflandırma başlığı seed ayarlandıktan sonra oluşturulur. Arama seed 42 sonucu yerine üç refit de ayrı kaydedilir. Bu tekrarlar TF-IDF'nin deterministik lbfgs kontrollerinden farklıdır.

QLoRA sabit tarihsel konfigürasyon, encoder altı aday, TF-IDF 48 aday/bütçe kullanır. Arama maliyetlerinin eşit olduğu iddia edilmez; ortak veri bütçesi, validation ayrımı ve seçim ölçütleri açık raporlanır. Encoder tuning'in **uygulama kodu hazırdır; arama ve eğitim yapılmış değildir**.

## 5. Nihai evaluation kayıtları

Base LLM ve her üç FT seed'i için internal test + 12 legacy görünümü; encoder'larda seçilmiş üç seed için aynı setler. Greedy generation, max_new_tokens 32, thinking kapalı, batch 1; base/FT aynı tokenizer ve SDPA attention/NF4/FP16 kullanır. Adapter birleştirilmez. Aynı regex ayrıştırıcı kullanılır; geçersiz sonuç `__off_label__` olarak saklanır, düşürülmez.

Çıktıda record ID veya legacy `dosya:satır` görünüm kimliği, source_file/source_row, gerçek etiket ve tahmin bulunur. LLM'de ham cevap, prompt/token hash'i, giriş/çıkış token sayısı ve gerçek görülen karakter sayısı da yazılır. Legacy görünüm kimliği orijinal CFPB complaint ID değildir.

Yeni belirsizlik ve nihai karşılaştırma için mevcut labelsız gruplar/unique-input ağırlıkları ve 10.000 paired grup bootstrap kullanılacak. Run SD ayrı verilecek. Kod şu anda tahmin ve betimsel metrik üretmeye hazır; GPU sonuçları gelmeden nihai karşılaştırma/CI üretilmez.

## 6. Kontrollü uzun girdi deneyi

**Aynı eğitilmiş checkpoint'ler, aynı bütün 15.428 internal-test ID'si, yeniden eğitim yok.** İki base model ve FT'nin üçer seed'i için üç koşul:

| Koşul | Anlatı sınırı | Prompt dahil input token sınırı | Amaç |
| --- | ---: | ---: | --- |
| chars500_tokens512 | 500 | 512 | Ana protokol |
| chars500_tokens2048 | 500 | 2.048 | Token-cap kontrolü |
| chars1500_tokens2048 | 1.500 | 2.048 | Daha uzun görünen metin |

Ana uzunluk karşılaştırması **500/2048 → 1500/2048**; aynı 500-karakter girdide 512→2048 farkı ayrıca kontrol edilir. İçerik değişmeyen anlatılar korunup toplam sonuçta yer alır; asıl uzunluk etkisi için ayrıca tam anlatısı >500 karakter olan önceden tanımlı kohort raporlanır. Kohort tahminlere göre seçilmez. Legacy'nin kırpılmış dosyalarından uzun metin uydurulmaz.

Her kayıtta talep edilen ve gerçekte kalan karakter, cap öncesi token ve tutulan token sayıları denetlenir. Her 1.500 karakterin mutlaka görünür olduğu varsayılmaz. Sınır yetersiz çıkarsa otomatik büyütme/teste göre ayar yok; görünürlük kaybı raporlanır veya gerekçeli protokol eki hazırlanır.

Accuracy, macro-F1 ve weighted-F1; koşullar arasında aynı ID/grup örneklemelerini kullanan 10.000 paired bootstrap, üç FT seed'i için ayrı run SD. Bu tasarım **500-karakterle eğitilmiş modellerin inference girdi uzunluğuna duyarlılığıdır**; daha uzun bağlamla yeniden eğitimin etkisi değildir. Daha uzun eğitim gerekirse ayrı deney/protokol olur.

## 7. Kontrollü gecikme ölçümü

[100 kayıtlık sabit örneklem](analysis/gpu_preparation/latency_sample_manifest.csv): internal testten her sınıf için `sha256(latency_v1|record_id)` sırasındaki ilk 20 kayıt. Bu dengeli örneklem doğal iş yükü prevalansı değildir. [10 warmup kaydı](analysis/gpu_preparation/latency_warmup_manifest.csv) sabit validation dosyasının ilk 10 kaydıdır. Bütün model/seed/uzunluk koşullarında aynı ID'ler kullanılır.

Her varyant/koşul yüklemesinden sonra 10 warmup; ardından 100 kaydın üç tekrarı. Her tekrardaki sıra `sha256(order_v1|repetition|record_id)` ile belirlenir. Batch 1; greedy/max_new_tokens 32; aynı makine/quantization/attention ayarı; unmerged adapter. Ölçüm sırasında başka GPU işi çalıştırılmamalı. Başta ve sonda GPU kimliği, sıcaklık, kullanım, bellek, güç ve clock bilgileri kaydedilir. Varyantlar ayrı yüklenir; aynı iş içinde üç tekrar yapılır. Tasarım tam randomize çaprazlama değildir; termal/yük drift'i kalabilir ve sınır olarak belirtilecek.

İki zaman kaydı:

- **Generation:** girdiler GPU'ya taşındıktan sonra CUDA synchronize → perf_counter → generate → CUDA synchronize → perf_counter. Prefill ve autoregressive üretimi içerir.
- **End-to-end:** tokenize, H2D, generation, D2H/decode ve regex ayrıştırmayı içerir; sınırlar synchronize ile kapatılır. Model/tokenizer yükleme, warmup ve disk yazımı dışarıda.

İstek başına input/new token sayısı saklanır; farklı EOS uzunluklarının süreye etkisi görünür tutulur. Ortalama, medyan, p95 ve ölçülen toplam end-to-end süreden **seri batch-1 requests/s** raporlanır; bu servis kapasitesi/SLA değildir. Eşleştirilmiş farklar için önce aynı ID'nin üç tekrarının ortalaması alınır, aynı kimlikler birlikte örneklenir; 300 ölçüm bağımsız 300 şikâyet sayılmaz. Gecikmeden deployment kabulü veya regulatory readiness sonucu çıkarılmaz.

## 8. Çalıştırma sırası ve mevcut doğrulama sınırı

1. Registry kimlikleri ve uyumlu GPU ortamını hazırlamak; mevcut tarihsel eksikleri ayrıca kaydetmek.
2. Her model için CPU tokenizer audit'ini geçirmek.
3. İki QLoRA modelinin üç seed'ini ve encoder aramalarını çalıştırmak; encoder kazananlarını dondurup üçer refit yapmak.
4. Ayrı `evaluate` komutlarıyla final tahminleri üretmek.
5. Aynı checkpoint'lerde `long-input` ve kontrollü `latency` koşullarını çalıştırmak.
6. Yeni sonuçları mevcut istatistik hattına almak, nihai karşılaştırma ve hakem cevaplarını güncellemek.

[commands_dry_run.txt](analysis/gpu_preparation/commands_dry_run.txt) bu işleri tek tek listeler. **Komutların varsayılanı dry-run**; eğitim/LLM inference/latency başlamaz. `audit` gerçek registry mevcutsa yalnız CPU tokenizer denetimi yapar. Gerçek GPU çağrısında sonradan açıkça `--execute-gpu` eklenmesi gerekir. Output klasörü varsa üzerine yazılmaz; yarım koşu için kayıtlı devam/yeniden deneme politikası ayrıca belirlenir. Şu an otomatik resume uygulanmadı.

```bash
# Şimdi yapılabilen CPU dry-run örnekleri:
python REVISION_PAPER/scripts/run_revision_gpu.py qlora --model qwen --seed 42
python REVISION_PAPER/scripts/run_revision_gpu.py encoder-search --model bert
python REVISION_PAPER/scripts/run_revision_gpu.py encoder-refit --model bert --seed 43
python REVISION_PAPER/scripts/run_revision_gpu.py long-input --model llama --variant 42 --condition chars1500_tokens2048
python REVISION_PAPER/scripts/run_revision_gpu.py latency --model qwen --variant base --condition chars500_tokens512
```

CPU test kaydı: [CPU_PREPARATION_VALIDATION.json](analysis/gpu_preparation/CPU_PREPARATION_VALIDATION.json). Testler komut/üyelik/hash/selection guard'larını ve mevcut tokenizer'larla maske/serileştirmeyi denetler. Gerçek yeni snapshot audit'leri, CUDA smoke testi, eğitim, yeni tahmin ve latency ölçümleri **henüz yapılmadı**.

API dayanakları: [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [chat-template special-token uyarısı](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating), [TRL 0.23.1 SFTTrainer kaynak kodu](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/sft_trainer.py), [PyTorch CUDA synchronize](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.synchronize.html). Kurulu CPU Transformers 4.57.6 kaynak alanları ayrıca kontrol edilir; 4.57.1 doküman sayfası 4.57.6 GPU smoke testinin yerine geçmez.
