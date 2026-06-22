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
- [ ] 10.000 örnekli pilot QLoRA eğitimi gerçekleştirilecek.
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
