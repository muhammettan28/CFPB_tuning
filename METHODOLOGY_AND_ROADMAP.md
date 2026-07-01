# CFPB Fine-Tuning Çalışması: Metodoloji Kararları ve Yol Haritası

Bu belge, zero-shot CFPB çalışmasının devamı olarak yürütülecek fine-tuning
projesinde alınan kararları sabitlemek ve deneyler ilerledikçe test verisine
istemeden uyum sağlanmasını önlemek için hazırlanmıştır. Belge yaşayan bir
metodoloji kaydıdır; deney protokolü değişirse değişikliğin tarihi ve gerekçesi
buraya eklenmelidir.

Referans makale:

`ref_paper/Can Open Source LLMs Replace Paid API Services for Zero Shot Consumer Complaint Classification.pdf`

## 1. Araştırmanın amacı

İlk çalışmada on bir açık ağırlıklı LLM, CFPB tüketici şikâyetlerini beş ürün
kategorisine zero-shot olarak sınıflandırmıştır. Yeni çalışmanın temel sorusu:

> Göreve özgü fine-tuning, açık ağırlıklı LLM'lerin zero-shot performansını ne
> ölçüde ve hangi eğitim/çıkarım maliyeti karşılığında artırır?

İlk çalışmadaki ana referans, dengeli değerlendirmede `gemma4:12b` için
`weighted F1 = 0.774 ± 0.016` sonucudur. Yeni çalışma yalnızca en yüksek skoru
aramayacak; performans artışını model ailesi, model büyüklüğü, reasoning
özelliği, eğitim verisi miktarı, sınıf ve hesaplama maliyeti bakımından
inceleyecektir.

## 2. Veri kaynakları

Ana veri:

- `dataset/complaints.csv`
- Ham satır sayısı: 162.415
- Eksik anlatı nedeniyle çıkarılan satır: 10
- Kullanılabilir satır: 162.405
- Normalize edilmiş benzersiz anlatı: 124.471

Hedef sınıflar:

1. `credit_reporting`
2. `debt_collection`
3. `mortgages_and_loans`
4. `credit_card`
5. `retail_banking`

İlk zero-shot çalışmada kullanılan legacy değerlendirme dosyaları:

- Dizin: `dataset/legacy_subsets/`
- Üç seed: 1, 2 ve 3
- İki dağılım: balanced ve imbalanced
- İki anlatı sınırı: 250 ve 500 karakter
- Toplam 12 CSV ve 12.000 değerlendirme satırı
- 5.909 benzersiz `etiket + anlatı` çifti

Bu 12 dosya birbirinden bağımsız test setleri değildir. Seed'ler, balanced ve
imbalanced koşullar ile 250/500 karakter sürümleri arasında örtüşen kayıtlar
vardır. Bu nedenle dosyalar birleştirilerek 12.000 örnekli tek bir test seti
gibi raporlanmayacaktır.

## 3. Veri setlerinin rolleri

### Train

Model ağırlıklarının veya LoRA adaptörlerinin öğrenilmesinde kullanılır.

Dosya: `dataset/splits/train.csv`

### Validation

Aşağıdaki kararların tamamı yalnızca validation sonuçlarına göre alınır:

- Learning rate
- Epoch sayısı ve early stopping
- LoRA rank, alpha ve dropout
- Batch size ve gradient accumulation
- Sequence length
- Checkpoint seçimi
- Sampling veya class-weighting stratejisi

Dosya: `dataset/splits/validation.csv`

### Internal test

Yeni çalışmanın bağımsız, doğal sınıf dağılımlı ve birincil test setidir.
Model veya hiperparametre seçmek için kullanılamaz. Nihai deney yapılandırması
dondurulduktan sonra çalıştırılır.

Dosya: `dataset/splits/internal_test.csv`

### Legacy test

İlk makaledeki zero-shot sonuçlarla aynı değerlendirme protokolü üzerinde
doğrudan, eşleştirilmiş karşılaştırma yapmak için kullanılır. Bu set:

- Validation değildir.
- Model/checkpoint seçmek için kullanılamaz.
- Yeni makalenin tek test seti değildir.
- `held-out legacy paired benchmark` olarak konumlandırılır.

Manifest: `dataset/splits/legacy_test_manifest.csv`

Legacy testin daha önce zero-shot inference için kullanılmış olması ağırlık
güncellemesi yaratmadığından doğrudan training leakage değildir. Bununla
birlikte ilk makaledeki model seçimi ve araştırmacı bilgisi dolaylı uyum
eleştirisine açık olabilir. Bu nedenle yeni makalenin birincil genelleme
sonuçları internal test üzerinde, legacy sonuçları ise ikincil ve eşleştirilmiş
karşılaştırma olarak verilecektir.

## 4. Leakage önleme mantığı

Split üretimi `scripts/prepare_splits.py` ile, seed `42` kullanılarak
yapılmıştır.

Uygulanan sıra:

1. Eksik `product` veya `narrative` içeren kayıtlar çıkarıldı.
2. Metinler Unicode NFKC, küçük harf, baş/son boşluk temizliği ve ardışık
   boşluk birleştirmesiyle normalize edildi.
3. Legacy dosyalarla aynı etiket ve anlatıya sahip exact eşleşmeler ana
   havuzdan çıkarıldı.
4. Legacy 250-karakter örnekleri çoğu zaman orijinal anlatının kırpılmış
   sürümü olduğu için ilk 250 karakter eşleşmeleri çıkarıldı.
5. Prefix dışlama etiketten bağımsız da uygulandı. Böylece aynı legacy metni
   farklı etiketle kayıtlı olsa bile train/validation/internal test içinde
   kalmadı.
6. Dışlanan bir anlatının farklı etiketli tam kopyaları varsa normalize edilmiş
   anlatı grubunun tamamı çıkarıldı.
7. Kalan veride aynı normalize edilmiş anlatıya ait bütün satırlar tek grup
   olarak aynı split'e atandı.
8. Gruplar yaklaşık stratified şekilde `%80 train / %10 validation /
   %10 internal test` oranında ayrıldı.

Legacy ilişkisi nedeniyle ana veriden çıkarılan:

- Satır: 8.123
- Benzersiz normalize edilmiş anlatı: 4.888

Dışlama nedenleri:

| Neden | Satır |
|---|---:|
| Legacy exact eşleşme | 4.990 |
| Aynı etiketli 250-karakter prefix | 3.096 |
| Normalize edilmiş anlatı grubunun tamamını dışlama | 23 |
| Farklı etiketli 250-karakter prefix | 14 |

Çıkarılan kayıtların denetim dosyası:

`dataset/splits/legacy_excluded_from_training.csv`

## 5. Mevcut split özeti

| Split | Satır | Benzersiz anlatı | Oran |
|---|---:|---:|---:|
| Train | 123.427 | 95.365 | %80,001 |
| Validation | 15.427 | 12.063 | %9,999 |
| Internal test | 15.428 | 12.155 | %10,000 |

Sınıf dağılımı:

| Sınıf | Train | Validation | Internal test |
|---|---:|---:|---:|
| `credit_reporting` | 68.400 | 8.566 | 8.571 |
| `debt_collection` | 18.053 | 2.206 | 2.244 |
| `mortgages_and_loans` | 14.703 | 1.838 | 1.826 |
| `credit_card` | 11.925 | 1.524 | 1.494 |
| `retail_banking` | 10.346 | 1.293 | 1.293 |

Doğrulanan overlap sonuçları:

- Train–validation normalize edilmiş anlatı overlap: 0
- Train–internal test normalize edilmiş anlatı overlap: 0
- Validation–internal test normalize edilmiş anlatı overlap: 0
- Kalan havuzda legacy exact hash overlap: 0
- Train'de legacy 250-karakter prefix overlap: 0
- Validation'da legacy 250-karakter prefix overlap: 0
- Internal testte legacy 250-karakter prefix overlap: 0

Makine tarafından üretilen ayrıntılı rapor:

`dataset/splits/split_report.json`

## 6. Deney sırasında değiştirilemeyecek kurallar

1. Train dışında hiçbir set model ağırlıklarını güncellemek için kullanılamaz.
2. Hiperparametre ve checkpoint seçimi yalnızca validation ile yapılır.
3. Internal test ve legacy test sonuçlarına bakılarak eğitim protokolü
   değiştirilemez.
4. Internal test, final yapılandırma dondurulana kadar çalıştırılmamalıdır.
5. Legacy sonuçları incelendikten sonra modelde değişiklik yapılırsa bu yeni
   bir deney turu sayılır; önceki legacy sonucu artık kör değerlendirme olarak
   tanımlanamaz.
6. Eski 12 subset dosyası eğitim veya validation verisine eklenemez.
7. Split CSV'leri manuel olarak düzenlenmemelidir. Gerekirse
   `python scripts/prepare_splits.py` komutuyla yeniden üretilmelidir.
8. Seed veya split algoritması değiştirilirse eski splitler üzerine
   yazılmadan yeni sürüm ve gerekçe kaydedilmelidir.

## 7. Değerlendirme ve raporlama protokolü

### Birincil sonuçlar: internal test

Raporlanacak metrikler:

- Macro-F1 (ana başarı metriği)
- Weighted F1
- Accuracy
- Her sınıf için precision, recall ve F1
- Confusion matrix
- Format/off-label hata sayısı
- Ortalama ve tercihen p50/p95 inference latency
- Peak VRAM

Ana verinin dengesiz olması nedeniyle yalnızca accuracy veya weighted F1
üzerinden sonuç çıkarılmayacaktır. Birincil model karşılaştırması macro-F1 ile
yapılacaktır.

### İkincil sonuçlar: legacy benchmark

Legacy dosyalar şu eksenlerde ayrı tutulacaktır:

- Balanced ve imbalanced
- 250 ve 500 karakter
- Seed 1, 2 ve 3

Her koşul için üç seed üzerinden ortalama ve standart sapma raporlanacaktır.
250 ve 500 karakter dosyaları ile balanced/imbalanced dosyaları tek bir büyük
test setinde birleştirilmeyecektir.

Temel karşılaştırma:

`Delta F1 = fine-tuned F1 - zero-shot F1`

Zero-shot ve fine-tuned model aynı örneklerde değerlendirildiğinden sonuçlar
eşleştirilmiştir. Uygun istatistiksel testler:

- Accuracy/hata farkı için paired McNemar testi
- F1 ve sınıf bazlı metrikler için paired bootstrap güven aralığı
- Birden fazla model karşılaştırmasında çoklu test düzeltmesi

## 8. Hakemlere karşı metodolojik konumlandırma

Makalenin Methods ve Limitations bölümlerinde şu noktalar açıkça yazılmalıdır:

- Legacy örneklerin tamamı fine-tuning train ve validation havuzundan
  çıkarılmıştır.
- Exact eşleşmenin yanında kırpılmış 250-karakter prefix eşleşmeleri ve farklı
  etiketli metin kopyaları da dışlanmıştır.
- Aynı normalize edilmiş anlatının farklı splitlere düşmesi grup bazlı ayırma
  ile engellenmiştir.
- Model ve hiperparametre seçimi yalnızca validation üzerinde yapılmıştır.
- Internal test, yeni çalışmanın bağımsız birincil değerlendirmesidir.
- Legacy test, ilk makaleyle aynı örnek/protokol üzerindeki ikincil paired
  benchmarktır.
- Legacy test sonuçları model seçimi için kullanılmamıştır.
- Legacy dosyalar arasındaki örnek örtüşmesi nedeniyle 12 dosya bağımsız
  12.000 örnek olarak yorumlanmamıştır.

Bu tasarım, ilk makaleyle doğrudan karşılaştırılabilirliği korurken yeni
çalışmanın yalnızca önceden görülmüş benchmark seçimlerine dayanması
eleştirisini internal test ile azaltır.

## 9. Fine-tuning deney yol haritası

### Aşama 1 — Pilot altyapı

- TRUBA üzerinde tek modelle uçtan uca QLoRA eğitimi kur.
- İlk pilot modeli olarak `Qwen/Qwen3-8B` kullan. Bu model ilk makaledeki
  `qwen3:8b` Ollama modelinin aynı upstream ailesindendir.
- Ollama Q4 GGUF artefaktının fine-tuning için doğrudan kullanılamayacağını;
  Hugging Face `safetensors` checkpoint'inin ayrıca gerektiğini kaydet.
- İlk pilotu V100 16 GB üzerinde FP16 compute, 4-bit NF4 QLoRA ve non-thinking
  chat şablonuyla çalıştır.
- Eğitim logları, seed, paket sürümleri, GPU türü, wall time ve peak VRAM'i
  kaydet.
- Validation macro-F1 ile checkpoint seçimini doğrula.
- Eğitimden bağımsız inference ve değerlendirme scriptleri oluştur.

Pilot dosyaları:

- `scripts/train_qlora.py`
- `scripts/evaluate_adapter.py`
- `scripts/download_hf_model.py`
- `slurm_files/qwen3_8b_qlora_pilot.slurm`
- `slurm_files/README.md`
- `requirements-truba.txt`

Doğrulanan ilk TRUBA pilot ortamı:

- Compute node doğrulaması: `akya1` ve `akya8`
- GPU: Tesla V100-SXM2-16GB
- NVIDIA driver: 565.57.01
- Driver CUDA seviyesi: 12.7
- Sistem Python: 3.9.16
- Deney ortamı: Miniforge/Conda Python 3.10.20 (`.conda-env`)
- Sistem CUDA/PyTorch modülü: bulunmadı
- Sabitlenen PyTorch: `2.7.1+cu118`
- Sabitlenen bitsandbytes: `0.46.1`
- CUDA kullanılabilirliği: doğrulandı
- GPU compute capability: `(7, 0)`
- Conda ortamında `Python.h`: doğrulandı
- Standart venv kullanılmama nedeni: sistem Python geliştirme başlığı
  `Python.h` bulunmadığından bitsandbytes/Triton yardımcı modülü derlenemedi.

### Aşama 2 — Temel model karşılaştırması

Önerilen başlangıç modeli kümesi:

- `gemma4:12b`: en güçlü zero-shot başlangıç
- `qwen3:8b`: düşük VRAM ve güçlü format uyumu
- `qwen3:14b`: güçlü reasoning modeli
- `llama3.1:8b`: düşük zero-shot tabanından olası yüksek fine-tuning kazancı
- `phi4:14b`: güçlü latency–accuracy dengesi

TRUBA kapasitesi elverirse aynı protokol diğer modellerle genişletilir.

### Aşama 3 — Eğitim verisi öğrenme eğrisi

Her model veya seçilen temsilci modeller için:

- 100 örnek/sınıf
- 500 örnek/sınıf
- 1.000 örnek/sınıf
- 5.000 örnek/sınıf
- Tüm train verisi

Alt örnekler train içinden alınmalı, validation ve testlere dokunulmamalı ve
örnekleme seed'leri kaydedilmelidir. Amaç, zero-shot performansını aşmak için
kaç etiketli örnek gerektiğini ölçmektir.

### Aşama 4 — Sınıf dengesizliği

En az aşağıdaki eğitim koşulları karşılaştırılmalıdır:

- Doğal sınıf dağılımı
- Balanced sampling veya class-aware sampling

Generative LLM eğitiminde loss'a doğrudan class weight uygulamak kullanılan
altyapıya bağlıdır; uygulanırsa yöntem açıkça belgelenmelidir. Sonuçlar
özellikle azınlık sınıflarının recall ve F1 değerleriyle yorumlanmalıdır.

İlk sınırlı arama turu:

| Deney | Train sampling | LR | LoRA r/alpha | Epoch | Durum |
|---|---|---:|---:|---:|---|
| E0 | Natural | 2e-4 | 16/32 | 1 | Tamamlandı |
| E1 | Balanced | 2e-4 | 16/32 | 1 | Planlandı |
| E2 | Balanced | 1e-4 | 16/32 | 1 | Planlandı |
| E3 | Balanced | 2e-4 | 8/16 | 1 | Planlandı |

Balanced koşullarda 10.000 train kaydı, sınıf başına 2.000 örnek olarak
seçilecektir. Validation doğal dağılımda ve tüm deneylerde aynı 2.000 kayıt
olarak sabit kalacaktır. Her koşunun seçtiği train ve validation kayıtları
manifest CSV dosyalarına yazılacaktır.

Bu turun amacı aynı anda geniş bir grid search yapmak değil, sırasıyla sampling,
learning rate ve LoRA kapasitesinin etkisini ölçmektir. İki epoch deneyi yalnızca
bu turda validation macro-F1 açısından kazanan yapılandırmaya uygulanacaktır.

### Aşama 5 — Güçlü baseline'lar

Fine-tuned generative LLM sonuçları en az bir encoder ve bir klasik yöntemle
karşılaştırılmalıdır:

- DeBERTa-v3, RoBERTa veya benzeri encoder classifier
- Mümkünse FinBERT
- TF-IDF + Logistic Regression

Küçük bir encoder modelin LLM'leri geçmesi olasıdır ve bilimsel olarak önemli
bir sonuçtur; bu baseline'lar çıkarılmamalıdır.

### Aşama 6 — Nihai değerlendirme

1. Model ve hiperparametreleri validation ile dondur.
2. Deney konfigürasyonunu ve checkpoint checksum'larını kaydet.
3. Internal test üzerinde birincil değerlendirmeyi çalıştır.
4. Legacy 12 dosyada ikincil paired değerlendirmeyi çalıştır.
5. İstatistiksel testleri ve güven aralıklarını hesapla.
6. Zero-shot, fine-tuned ve baseline sonuçlarını performans–maliyet ekseninde
   karşılaştır.

## 10. Deney kayıtlarında zorunlu alanlar

Her eğitim koşusu için en az şu bilgiler saklanmalıdır:

- Deney kimliği ve tarih
- Base modelin tam adı/revizyonu
- Tokenizer revizyonu
- Train veri sürümü ve hash'i
- Eğitim alt kümesi ve sampling seed'i
- Global random seed
- LoRA/QLoRA parametreleri
- Quantization ayarları
- Learning rate, scheduler ve warmup
- Epoch, batch size, gradient accumulation
- Maksimum sequence length
- Optimizer
- Validation metriği ve seçilen checkpoint
- GPU modeli, GPU sayısı, wall time ve peak VRAM
- Yazılım ve CUDA sürümleri
- Checkpoint/adaptör checksum'ı

## 11. Açık araştırma soruları

- RQ1: Fine-tuning, open-weight LLM'lerin zero-shot performansını ne kadar
  artırır?
- RQ2: Artış model büyüklüğü, ailesi ve reasoning özelliğine göre değişir mi?
- RQ3: Performans kaç etiketli örnekten sonra doygunluğa ulaşır?
- RQ4: Fine-tuning azınlık sınıflarını ve macro-F1'ı ne ölçüde iyileştirir?
- RQ5: Performans artışı eğitim GPU-saati ve inference maliyetine değer mi?
- RQ6: Fine-tuned generative LLM'ler encoder tabanlı classifier'ları geçer mi?
- RQ7: Zero-shot model sıralaması fine-tuning sonrasında korunur mu?

## 12. Mevcut durum

- [x] İlk makalenin yöntemi, sonuçları ve sınırlılıkları incelendi.
- [x] Legacy subsetlerin kendi aralarındaki tekrarlar ölçüldü.
- [x] 250-karakter subsetlerin kırpılmış anlatılar içerdiği doğrulandı.
- [x] Legacy-related kayıtlar ana eğitim havuzundan çıkarıldı.
- [x] Grup bazlı, yaklaşık stratified train/validation/internal-test splitleri
      üretildi.
- [x] Exact ve prefix overlap kontrolleri sıfırlandı.
- [x] Split raporu ve legacy manifest oluşturuldu.
- [x] TRUBA fine-tuning ortamı ve paket sürümleri sabitlendi.
- [x] Qwen3-8B QLoRA smoke testi tamamlandı.
- [x] 10.000 örnekli pilot QLoRA eğitimi gerçekleştirildi.
- [ ] Evaluation ve istatistik scriptleri hazırlanacak.
- [ ] Model ve veri ölçeği deneyleri yürütülecek.
- [ ] Internal ve legacy final değerlendirmeleri protokole uygun çalıştırılacak.

### Qwen3-8B QLoRA smoke testi

Smoke test, eğitim ve inference zincirini doğrulamak amacıyla 64 train ve
32 validation örneği üzerinde çalıştırılmıştır.

- Epoch: 1
- Train süresi: 112,8 saniye
- Train loss: 0,2490
- En düşük gözlenen validation loss: 0,1330
- Validation inference örneği: 32
- Accuracy: 0,7813
- Macro-F1: 0,7085
- Weighted F1: 0,7719
- Off-label: 0
- Ortalama inference latency: 0,966 saniye/örnek

Bu metrikler yalnızca teknik smoke test sonucudur. Küçük ve doğal dağılımlı
32 örnek üzerinden elde edildiği için model performans bulgusu veya makale
sonucu olarak kullanılmayacaktır.

### Qwen3-8B 10.000 örnekli ilk pilot

İlk ciddi pilot doğal sınıf dağılımından stratified seçilen 10.000 train ve
2.000 validation örneğiyle, 1 epoch çalıştırılmıştır.

- Train süresi: 9.394,8 saniye (2 saat 36 dakika 35 saniye)
- Train loss: 0,0598
- En iyi checkpoint: step 600
- En iyi validation loss: 0,0492
- Validation accuracy: 0,8475
- Validation macro-F1: 0,8098
- Validation weighted F1: 0,8439
- Off-label: 0
- Ortalama inference latency: 0,876 saniye/örnek

En zayıf sınıf `debt_collection` olmuştur (`F1 = 0,6429`); 265 örneğin 95'i
`credit_reporting` olarak tahmin edilmiştir. Sınıf dengeleme deneylerinde bu
karışıklık özellikle izlenecektir.

Bu sonuç yalnızca validation performansıdır. Fine-tuning kazancını izole etmek
için aynı Hugging Face base modelin aynı quantization, prompt ve 2.000
validation örneğinde adaptersız zero-shot sonucu ayrıca ölçülecektir.

### Kontrollü zero-shot ve fine-tuned validation karşılaştırması

Aynı Hugging Face Qwen3-8B checkpoint'i; aynı 4-bit NF4 quantization, prompt,
500-karakter sınırı, decoding ayarları ve aynı 2.000 validation örneği üzerinde
adapter olmadan ve QLoRA adapter ile değerlendirilmiştir.

| Metrik | Zero-shot | Fine-tuned | Mutlak değişim |
|---|---:|---:|---:|
| Accuracy | 0,7115 | 0,8475 | +0,1360 |
| Macro-F1 | 0,5993 | 0,8098 | +0,2104 |
| Weighted F1 | 0,6931 | 0,8439 | +0,1507 |
| Weighted precision | 0,7236 | 0,8433 | +0,1197 |
| Weighted recall | 0,7115 | 0,8475 | +0,1360 |
| Off-label | 1 | 0 | -1 |
| Ortalama latency (sn) | 0,565 | 0,876 | +0,311 |

Sınıf bazında F1 değişimi:

| Sınıf | Zero-shot | Fine-tuned | Mutlak değişim |
|---|---:|---:|---:|
| `retail_banking` | 0,5571 | 0,8603 | +0,3032 |
| `credit_reporting` | 0,8183 | 0,8965 | +0,0781 |
| `credit_card` | 0,5794 | 0,8098 | +0,2303 |
| `mortgages_and_loans` | 0,6905 | 0,8394 | +0,1489 |
| `debt_collection` | 0,3513 | 0,6429 | +0,2916 |

En büyük mutlak kazanımlar `retail_banking` ve `debt_collection` sınıflarında
görülmüştür. Bununla birlikte `debt_collection`, fine-tuning sonrasında da en
zayıf sınıftır. Adapter inference latency'si yaklaşık %55 artmıştır.

Bu karşılaştırma fine-tuning etkisinin güçlü bir pilot göstergesidir; nihai
test sonucu değildir. Aynı validation verisi checkpoint seçimi ve performans
ölçümü için kullanıldığından sonuç model seçimine iyimser olabilir. Makaledeki
nihai genelleme iddiaları yalnızca yapılandırma dondurulduktan sonra internal
test ve legacy benchmark üzerinde yapılacaktır. İlk makaledeki Qwen3-8B skoru
farklı veri dağılımı ve Ollama/GGUF altyapısı nedeniyle bu validation skoruyla
doğrudan karşılaştırılmayacaktır.

### Deney arşivleme durumu

İlk Qwen3-8B deneyleri yerel `experiment_archive/` dizinine alınmıştır:

- Smoke-test output ve adapter
- 10.000 örnekli pilot output, best adapter ve checkpoint'ler
- Kontrollü zero-shot validation predictions ve metrics
- Pilot ve zero-shot SLURM logları
- Conda ortamının `environment.freeze.txt` kaydı
- Deney sırasında kullanılan scripts, SLURM dosyaları ve requirements snapshot'ı
- Yerel dosyalar için `SHA256SUMS` bütünlük manifesti

Pilot ve zero-shot prediction dosyalarının her biri 2.000 satırdır. `record_id`
ve gerçek etiket sıraları birebir aynıdır; bu nedenle kontrollü karşılaştırma
paired yapıdadır. Arşivdeki JSON ve CSV dosyaları parse edilerek doğrulanmıştır.

Smoke-test ve GPU ortam kontrol logları yerel arşivde bulunmuyorsa bunlar
teknik provenance açısından TRUBA'dan ayrıca alınmalıdır; smoke output,
run metadata ve adapter dosyaları mevcut olduğundan bilimsel pilot sonucunda
veri kaybı bulunmamaktadır.

### İkinci aşama validation planı

Sınırlı hiperparametre ve sampling aramasında 1 epoch sonuçları şu şekilde
yorumlanmıştır:

- En iyi genel validation performansı doğal sampling ile elde edilmiştir
  (`macro-F1 = 0,8098`, `weighted F1 = 0,8439`).
- Balanced sampling, genel macro/weighted F1 değerlerini az miktarda düşürmüş,
  ancak `debt_collection` F1 değerini yükseltmiştir.

Bu nedenle test setlerine geçmeden önce yalnızca iki 2-epoch aday çalıştırılacak:

| Aday | Sampling | LR | LoRA r/alpha | Epoch |
|---|---|---:|---:|---:|
| S1 | natural | 2e-4 | 16/32 | 2 |
| S2 | balanced | 2e-4 | 16/32 | 2 |

Bu deneyler `slurm_files/qwen3_8b_second_stage_2epoch.slurm` ile çalıştırılır.
Amaç daha geniş bir hiperparametre araması yapmak değil, ilk turda en makul
görünen iki seçeneğin epoch artışına verdiği tepkiyi ölçmektir.

Seçim metrik önceliği:

1. Validation macro-F1
2. Yakın sonuçlarda `debt_collection` F1
3. Weighted F1
4. Off-label count
5. Ortalama inference latency

Bu aşamada `internal_test.csv` ve `legacy_subsets/` hâlâ kullanılmayacaktır.
Nihai konfigürasyon validation üzerinde dondurulduktan sonra önce seed
stabilitesi, ardından internal test ve legacy benchmark raporlanacaktır.

### İkinci aşama sonucu ve seed stabilitesi

2 epoch validation deneyinde doğal sampling ayarı açık ara en iyi sonucu
vermiştir:

| Aday | Sampling | Epoch | Accuracy | Macro-F1 | Weighted F1 | Debt collection F1 |
|---|---|---:|---:|---:|---:|---:|
| S1 | natural | 2 | 0,8545 | 0,8235 | 0,8531 | 0,7033 |
| Pilot | natural | 1 | 0,8475 | 0,8098 | 0,8439 | 0,6429 |
| S2 | balanced | 2 | 0,8415 | 0,8065 | 0,8397 | 0,6456 |

Bu sonuçla final aday konfigürasyon şu şekilde belirlenmiştir:

- Model: Qwen3-8B
- Fine-tuning: QLoRA
- Train sampling: natural
- Train sample size: 10.000
- Validation sample size: 2.000
- Max narrative chars: 500
- Max length: 512
- Epoch: 2
- Learning rate: 2e-4
- LoRA r/alpha: 16/32

Seed 42 sonucu ikinci aşama S1 çalışmasında mevcuttur. Seed stabilitesi için
aynı konfigürasyon yalnızca seed 43 ve seed 44 ile yeniden çalıştırılacaktır
(`slurm_files/qwen3_8b_seed_stability.slurm`). Seed sonuçları validation
macro-F1 açısından tutarlıysa konfigürasyon dondurulacak; bundan sonra
hyperparameter veya sampling seçimi yapılmayacaktır.

Seed stabilitesi sonuçları konfigürasyonun tutarlı olduğunu göstermiştir:

| Seed | Accuracy | Macro-F1 | Weighted F1 | Debt collection F1 | Off-label |
|---:|---:|---:|---:|---:|---:|
| 42 | 0,8545 | 0,8235 | 0,8531 | 0,7033 | 0 |
| 43 | 0,8545 | 0,8187 | 0,8531 | 0,6975 | 0 |
| 44 | 0,8510 | 0,8181 | 0,8474 | 0,7015 | 0 |

Üç seed ortalaması:

- Accuracy: 0,8533 ± 0,0020
- Macro-F1: 0,8201 ± 0,0030
- Weighted F1: 0,8512 ± 0,0033
- `debt_collection` F1: 0,7008 ± 0,0030

Macro-F1 aralığı yalnızca 0,0054 olduğundan final aday konfigürasyon
dondurulmuştur. Bundan sonra validation üzerinde yeni hiperparametre veya
sampling seçimi yapılmayacak; sonraki adım aynı dondurulmuş konfigürasyonla
internal test ve legacy benchmark değerlendirmesidir.

### Final değerlendirme planı

Final değerlendirme `slurm_files/qwen3_8b_final_evaluation.slurm` ile
çalıştırılacaktır. Bu aşamada validation metriklerine bakılarak yeni karar
alınmayacaktır.

Çalıştırılacak varyantlar:

- HF Qwen3-8B base zero-shot
- Fine-tuned Qwen3-8B seed 42
- Fine-tuned Qwen3-8B seed 43
- Fine-tuned Qwen3-8B seed 44

Değerlendirilecek veri kümeleri:

- `dataset/splits/internal_test.csv`: ana nihai genelleme testi
- `dataset/legacy_subsets/*.csv`: önceki zero-shot makalesiyle karşılaştırmalı
  benchmark

Final çıktı yapısı:

- `outputs/qwen3_8b_final_eval_*/internal_test/metrics.json`
- `outputs/qwen3_8b_final_eval_*/internal_test/predictions.csv`
- `outputs/qwen3_8b_final_eval_*/legacy_subsets/*/metrics.json`
- `outputs/qwen3_8b_final_eval_*/legacy_subsets/*/predictions.csv`
- `outputs/qwen3_8b_final_evaluation_summary.csv`

Ana makale iddiası aynı HF pipeline içinde yapılacaktır:

> HF Qwen3-8B zero-shot ile aynı modelin QLoRA fine-tuned versiyonları
> karşılaştırılır.

Önceki makaledeki Ollama/GGUF zero-shot sonuçları ise legacy benchmark üzerinde
tarihsel/protokol karşılaştırması olarak raporlanacaktır.

### Final internal test sonuçları

Final değerlendirme çıktısında `internal_test.csv` için HF Qwen3-8B base
zero-shot ve üç fine-tuned seed sonucu elde edilmiştir. Bu aşama artık
validation seçimi değildir; dondurulmuş konfigürasyonun nihai test sonucudur.

| Varyant | Accuracy | Macro-F1 | Weighted F1 | Debt collection F1 | Off-label |
|---|---:|---:|---:|---:|---:|
| Base zero-shot | 0,7350 | 0,6294 | 0,7202 | 0,4644 | 1 |
| Fine-tuned seed 42 | 0,8706 | 0,8370 | 0,8696 | 0,7527 | 0 |
| Fine-tuned seed 43 | 0,8686 | 0,8339 | 0,8675 | 0,7512 | 0 |
| Fine-tuned seed 44 | 0,8697 | 0,8342 | 0,8687 | 0,7599 | 0 |

Fine-tuned üç seed ortalaması:

- Accuracy: 0,8696
- Macro-F1: 0,8350
- Weighted F1: 0,8686
- `debt_collection` F1: 0,7546

Base zero-shot'a göre ortalama mutlak kazanımlar:

- Accuracy: +0,1346
- Macro-F1: +0,2056
- Weighted F1: +0,1483
- `debt_collection` F1: +0,2902

Bu sonuç, QLoRA fine-tuning'in internal test üzerinde güçlü ve seed açısından
tutarlı bir performans artışı sağladığını göstermektedir. Mevcut arşivdeki
`qwen3_8b_final_evaluation_summary.csv` yalnızca `internal_test` satırlarını
içerdiği için legacy benchmark çıktıları ayrıca doğrulanmalı ve arşive
eklenmelidir.

Legacy subsetler için ayrı değerlendirme işi
`slurm_files/qwen3_8b_legacy_final_evaluation.slurm` olarak hazırlanmıştır. Bu
iş internal test'i tekrar çalıştırmaz; yalnızca `dataset/legacy_subsets/*.csv`
üzerinde base zero-shot ve üç fine-tuned seed varyantını değerlendirir. Beklenen
özet çıktı `outputs/qwen3_8b_legacy_final_evaluation_summary.csv` dosyasında
48 satırdır.

### Final legacy benchmark sonuçları

Legacy benchmark değerlendirmesi 12 subset ve 4 varyant için tamamlanmıştır:

- 12 legacy subset
- 1 base zero-shot varyant
- 3 fine-tuned seed varyantı
- Toplam 48 metrik satırı
- Toplam 48 `metrics.json` dosyası

Legacy subsetler üzerindeki ortalama performans:

| Metrik | Base zero-shot ort. | Fine-tuned seed ort. | Mutlak kazanım |
|---|---:|---:|---:|
| Accuracy | 0,6593 | 0,8160 | +0,1566 |
| Macro-F1 | 0,6150 | 0,8003 | +0,1853 |
| Weighted F1 | 0,6486 | 0,8163 | +0,1676 |

Seed bazında legacy ortalamaları:

| Seed | Accuracy | Macro-F1 | Weighted F1 |
|---:|---:|---:|---:|
| 42 | 0,8153 | 0,8005 | 0,8157 |
| 43 | 0,8156 | 0,7993 | 0,8161 |
| 44 | 0,8170 | 0,8011 | 0,8170 |

Balanced ve imbalanced legacy subsetlerde macro-F1 kazanımı birbirine çok
yakındır:

- Balanced subsetler: base 0,6150 → fine-tuned 0,8003 (`+0,1853`)
- Imbalanced subsetler: base 0,6194 → fine-tuned 0,8044 (`+0,1849`)

Bu sonuçlar, internal test bulgusuyla tutarlıdır: QLoRA fine-tuning hem yeni
internal test ayrımında hem de önceki zero-shot makalesinde kullanılan legacy
benchmark subsetlerinde base HF zero-shot modele göre belirgin performans
artışı sağlamaktadır.

## İkinci model doğrulaması: Llama 3.1 8B

Qwen3-8B deneyleri fine-tuning etkisini güçlü biçimde göstermektedir; ancak
makalenin önceki zero-shot çalışmaya bağlanması nedeniyle hakemler farklı açık
kaynak LLM'lerde de sonuç görmek isteyebilir. Bu nedenle ikinci model olarak
Llama 3.1 8B eklenecektir.

Bu aşamada yeni hiperparametre araması yapılmayacaktır. Qwen3-8B üzerinde
validation ile seçilip seed stabilitesi gösterilen protokol dondurulmuş şekilde
Llama 3.1 8B'ye uygulanacaktır:

- Model: Llama 3.1 8B Instruct
- Fine-tuning: QLoRA
- Train sampling: natural
- Train sample size: 10.000
- Validation sample size: 2.000
- Epoch: 2
- Learning rate: 2e-4
- LoRA r/alpha: 16/32
- Seeds: 42, 43, 44
- Final evaluation: internal test ve legacy subsets

Bu tasarımın amacı Llama için en iyi hiperparametreleri aramak değildir. Amaç,
Qwen3-8B'de gözlenen fine-tuning kazanımının en az bir ikinci açık kaynak LLM
üzerinde de görülüp görülmediğini test etmektir. Bu nedenle sonuçlar
“confirmatory/robustness experiment” olarak raporlanacaktır.

Llama çalıştırma dosyaları:

- `slurm_files/llama31_8b_fixed_protocol_train.slurm`
- `slurm_files/llama31_8b_final_internal.slurm`
- `slurm_files/llama31_8b_final_legacy.slurm`

Varsayılan HF checkpoint klasörü:

- `/arf/scratch/muhtan/ccc/hf_models/Llama-3.1-8B-Instruct`

Resmi Meta checkpoint'i erişim onayı/token gerektirebileceği için TRUBA tarafında
model indirme adımı ayrıca doğrulanmalıdır.

Llama 3.1 erişimi Hugging Face üzerinde `pending` kalırsa, hesapta erişimi
`accepted` görünen Meta Llama 3 ailesinden `Meta-Llama-3-8B-Instruct` aynı
doğrulama rolü için kullanılabilir. Bu durumda model adı makalede açıkça
“Llama 3 8B Instruct” olarak raporlanacak; Llama 3.1 ile karıştırılmayacaktır.

Llama 3 fallback dosyaları:

- `slurm_files/llama3_8b_fixed_protocol_train.slurm`
- `slurm_files/llama3_8b_final_internal.slurm`
- `slurm_files/llama3_8b_final_legacy.slurm`

Bu fallback, metodolojik olarak aynı amaca hizmet eder: Qwen3-8B'de gözlenen
fine-tuning kazanımının ikinci bir açık kaynak LLM ailesinde de görülüp
görülmediğini kontrol etmek.

### Llama 3.1 8B final sonuçları

Llama 3.1 8B Instruct için dondurulmuş Qwen protokolü değiştirilmeden
uygulanmıştır. Üç seed fine-tuning, internal final test ve legacy benchmark
tamamlanmış; arşive 3 validation seed satırı, 4 internal test satırı ve 48
legacy benchmark satırı alınmıştır.

Internal test sonuçları:

| Metrik | Base zero-shot | Fine-tuned seed ort. | Mutlak kazanım |
|---|---:|---:|---:|
| Accuracy | 0,7319 | 0,8717 | +0,1398 |
| Macro-F1 | 0,6189 | 0,8378 | +0,2189 |
| Weighted F1 | 0,7094 | 0,8705 | +0,1611 |
| `debt_collection` F1 | 0,3354 | 0,7613 | +0,4259 |

Legacy benchmark ortalamaları:

| Metrik | Base zero-shot ort. | Fine-tuned seed ort. | Mutlak kazanım |
|---|---:|---:|---:|
| Accuracy | 0,6394 | 0,8201 | +0,1807 |
| Macro-F1 | 0,5763 | 0,8053 | +0,2289 |
| Weighted F1 | 0,6134 | 0,8204 | +0,2070 |

Legacy macro-F1 seed ortalamaları:

- Seed 42: 0,8040
- Seed 43: 0,8068
- Seed 44: 0,8050

Bu sonuçlar Qwen3-8B bulgusunu güçlendirmektedir: aynı dondurulmuş protokol,
ikinci bir açık kaynak LLM olan Llama 3.1 8B üzerinde de hem internal testte hem
legacy benchmarkta güçlü ve tutarlı fine-tuning kazanımı üretmiştir.

## Makale konumlandırması, novelty ve yapılacak son analizler

Bu çalışmanın novelty'si yeni bir model mimarisi veya yeni bir fine-tuning
algoritması önermek değildir. Çalışma şu şekilde çerçevelenecektir:

> Önceki zero-shot açık kaynak LLM çalışmasını, leakage-kontrollü
> train/validation/test ayrımı, legacy benchmark koruması, iki farklı 8B açık
> kaynak model ailesi, üç seed stabilitesi ve paired zero-shot vs QLoRA
> fine-tuning karşılaştırmasıyla genişlettik.

Bu framing makalenin iddiasını daha doğru kurar: katkı bir algoritma icadı
değil; tekrarlanabilir, leakage-safe ve önceki zero-shot benchmark ile doğrudan
bağlantılı uygulamalı bir değerlendirme protokolüdür.

### Ana novelty noktaları

1. CFPB consumer complaint classification için zero-shot → QLoRA fine-tuning
   geçişinin sistematik analizi yapılmıştır.
2. Önceki makalede kullanılan legacy subsetler eğitimden dışlanmış ve final
   benchmark olarak korunmuştur. Bu, önceki zero-shot çalışma ile yeni
   fine-tuning çalışması arasında metodolojik köprü kurar.
3. Data leakage riski açıkça yönetilmiştir:
   - legacy örnekler train/validation/internal test havuzundan ayrılmıştır,
   - duplicate/near-duplicate anlatıların splitler arasında sızması
     engellenmiştir,
   - internal test ve legacy benchmark hiperparametre seçiminde
     kullanılmamıştır.
4. Aynı Hugging Face pipeline içinde base zero-shot ve QLoRA fine-tuned modeller
   karşılaştırılmıştır. Böylece kazanım altyapı farkından değil, fine-tuning
   etkisinden kaynaklanacak şekilde izole edilmiştir.
5. İki farklı 8B açık kaynak model ailesi kullanılmıştır:
   - Qwen3-8B
   - Llama 3.1-8B
6. Üç seed ile stabilite değerlendirilmiştir:
   - seed 42
   - seed 43
   - seed 44
7. İki ayrı final değerlendirme düzlemi kullanılmıştır:
   - leakage-safe `internal_test.csv`
   - önceki makaleyle bağlantılı `legacy_subsets/`
8. Sınıf bazlı kazanımlar raporlanacaktır. Özellikle zor sınıflardan
   `debt_collection` için fine-tuning sonrası büyük F1 artışı vurgulanacaktır.

Özet novelty cümlesi:

> Novelty, algoritmik bir icattan ziyade reproducible applied evaluation ve
> leakage-safe benchmark extension düzeyindedir.

### Yayın şansını artırmak için eklenecek son analizler

Aşağıdaki analizler makale yazımına geçmeden önce tamamlanacaktır:

1. Internal test için confidence interval hesaplanacak.
   - Accuracy, macro-F1 ve weighted F1 için bootstrap confidence interval
     tercih edilebilir.
   - Seed ortalamalarıyla birlikte raporlanacaktır.
2. Legacy 12 subset üzerinde paired istatistiksel test yapılacak.
   - Base zero-shot ve fine-tuned sonuçları subset bazında paired yapıdadır.
   - Wilcoxon signed-rank test uygun bir adaydır.
   - Ayrı ayrı accuracy, macro-F1 ve weighted F1 için uygulanabilir.
3. Confusion matrix ve sınıf bazlı error analysis eklenecek.
   - Qwen3-8B ve Llama 3.1-8B için internal test confusion matrixleri
     incelenecek.
   - En çok karışan sınıf çiftleri raporlanacak.
   - `debt_collection` sınıfındaki kazanım ayrıca tartışılacak.
4. Latency ve fine-tuning cost tartışması yapılacak.
   - Base zero-shot ve fine-tuned inference latency karşılaştırılacak.
   - Eğitim süresi, GPU tipi ve örnek sayısı raporlanacak.
   - Fine-tuning kazanımının ek maliyeti tartışılacak.
5. Limitations bölümü açık yazılacak.
   - Çalışma yeni bir model mimarisi önermemektedir.
   - QLoRA/LoRA yöntemi bilinen bir PEFT yaklaşımıdır.
   - Bulgular CFPB complaint classification bağlamında değerlendirilmelidir.
   - Daha büyük modeller, farklı veri kümeleri ve daha geniş hiperparametre
     aramaları gelecek çalışma olarak bırakılacaktır.

### Olası hedef dergi notu

İlk uygun hedeflerden biri Turkish Journal of Electrical Engineering and
Computer Sciences olabilir. Derginin kapsamı genel Electrical Engineering ve
Computer Sciences alanlarını kapsadığı için uygulamalı LLM/NLP sınıflandırma
çalışması kapsam açısından uyumlu görünebilir.

Ön değerlendirme:

- Dergi İngilizce, hakemli ve açık erişimdir.
- Açık erişim yayın ücreti olmadığı belirtilmektedir; ancak gönderim öncesi
  güncel author guidelines ve publication fee sayfası mutlaka tekrar kontrol
  edilmelidir.
- Makale DOI alır ve kalıcı erişim/preservation altyapısından yararlanır.
- Çalışmanın uygulamalı yapısı, leakage-safe deney tasarımı ve iki açık kaynak
  LLM ile doğrulama içermesi bu dergi için uygun bir paket oluşturabilir.

Bu hedef için makale yazılırken katkı cümlesi “yeni algoritma” iddiasına değil,
reproducible benchmarking, leakage-safe evaluation ve practical PEFT adaptation
üzerine kurulmalıdır.
