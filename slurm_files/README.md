# TRUBA Qwen3-8B QLoRA Pilotu

Bu pilot, ilk makaledeki `qwen3:8b` model ailesinin Hugging Face checkpoint'i
olan `Qwen/Qwen3-8B` üzerinde 4-bit NF4 QLoRA uygular. Ollama altında bulunan
`qwen3:8b` dosyası Q4 GGUF inference artefaktıdır ve PEFT/QLoRA eğitiminde
doğrudan kullanılamaz. Eğitim için Hugging Face `safetensors` ağırlıkları ayrıca
indirilmelidir.

## TRUBA hedef dizini

```text
/arf/scratch/muhtan/ccc/
├── dataset/
│   ├── splits/
│   └── legacy_subsets/
├── hf_models/
│   └── Qwen3-8B/
├── scripts/
├── slurm_files/
├── requirements-truba.txt
├── outputs/
├── logs/
├── miniforge3/
└── .conda-env/
```

`~/ollama` ve `~/ollama_models` mevcut zero-shot inference ortamı olarak
kalabilir. Fine-tuning süreci bunları değiştirmez.

## 1. Dosyaları TRUBA'ya kopyalama

Yerel bilgisayardan örnek:

```bash
rsync -av \
  dataset scripts slurm_files requirements-truba.txt muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/
```

## 2. Python ortamı

Login node üzerinde `nvidia-smi` komutunun bulunmaması normal olabilir. Login
node GPU düğümü değildir; GPU sürücüsü ve cihazı `akya-cuda` üzerinde ayrılan
compute node içinde kontrol edilmelidir.

Önce GPU ortamı tanılama işini gönder:

```bash
cd /arf/scratch/muhtan/ccc
mkdir -p logs
sbatch slurm_files/gpu_environment_check.slurm
```

`akya-cuda` kuyruğunda CPU talepleri 10 ve katları olmalıdır. Bu nedenle
tanılama ve pilot dosyaları `--cpus-per-task=10` kullanır.

İş tamamlanınca:

```bash
squeue -u muhtan
cat logs/cfpb-gpu-check-JOBID.out
cat logs/cfpb-gpu-check-JOBID.err
```

Bu çıktı GPU modelini, NVIDIA driver sürümünü, desteklenen CUDA seviyesini ve
TRUBA'da mevcut CUDA/PyTorch modüllerini gösterecektir.

İlk denemede oluşturulan standart `.venv`, sistem Python'ında geliştirme
başlıkları bulunmadığı için bitsandbytes/Triton importunda şu hatayı vermiştir:

```text
fatal error: Python.h: No such file or directory
```

Standart venv kendi Python geliştirme başlıklarını sağlamaz. Kullanıcı
yetkisiyle sistem `python3-devel` paketi de kurulamayacağından pilot ortamı,
Python başlıklarını beraberinde getiren Miniforge/Conda ile oluşturulacaktır.
Eski `.venv` artık fine-tuning işlerinde kullanılmayacaktır.

### Bu küme için sabitlenen ortam

Tanılama işi şu ortamı doğrulamıştır:

- GPU: Tesla V100-SXM2-16GB
- NVIDIA driver: 565.57.01
- Driver tarafından bildirilen CUDA seviyesi: 12.7
- Sistem Python: 3.9.16
- Sistem CUDA/PyTorch modülü: yok

Yeni deney ortamı Python 3.10 ve `torch==2.7.1+cu118` kullanacaktır. Yeni NVIDIA
driver daha eski CUDA runtime wheel'lerini çalıştırabildiği için CUDA 11.8 wheel
bu node üzerinde uygundur.

Güncel `scripts/`, `slurm_files/` ve `requirements-truba.txt` dosyalarını
TRUBA'ya kopyaladıktan sonra:

```bash
cd /arf/scratch/muhtan/ccc
bash scripts/setup_truba_env.sh
```

Script şunları otomatik yapar:

- Miniforge'u `/arf/scratch/muhtan/ccc/miniforge3/` altına kurar.
- Python 3.10 ortamını `.conda-env/` altında oluşturur.
- `torch==2.7.1+cu118` ve sabitlenmiş bağımlılıkları kurar.
- `environment.freeze.txt` üretir.
- Ortam içindeki `Python.h` başlığını doğrular.

`torchvision` ve `torchaudio` bu proje için gerekli değildir. Eski `.venv`
silinmek zorunda değildir, ancak SLURM işleri yalnızca `.conda-env` kullanır.

### Neden doğrudan en güncel PyTorch değil?

- `pip install torch` CUDA varyantını deney protokolünde açıkça sabitlemez.
- Nightly/preview paketler tam test edilmemiştir.
- Çok yeni CUDA wheel'i, TRUBA compute node NVIDIA driver'ından daha yeni bir
  runtime isteyebilir.
- Deneylerin yeniden üretilebilirliği için PyTorch ve CUDA varyantı
  `environment.freeze.txt` içinde sabitlenmelidir.

Bu pilot için Python 3.10 ve `torch==2.7.1+cu118` birlikte sabitlenmiştir.
Daha güncel PyTorch denemek istersek ayrı bir Conda ortamı oluşturmalı ve bunu
yeni bir deney ortamı olarak kaydetmeliyiz.

Kurulumdan sonra GPU kontrol işini tekrar gönder:

```bash
sbatch slurm_files/gpu_environment_check.slurm
```

Yeni logda aşağıdakiler görünmelidir:

```text
torch: 2.7.1+cu118
torch wheel CUDA: 11.8
CUDA available: True
GPU: Tesla V100-SXM2-16GB
compute capability: (7, 0)
bitsandbytes: 0.46.1
```

Bu kontrol 22 Haziran 2026 tarihinde `akya1` compute node üzerinde başarıyla
tamamlanmıştır. Doğrulanan ortam:

```text
Python: 3.10.20
Python.h exists: True
torch: 2.7.1+cu118
CUDA available: True
GPU: Tesla V100-SXM2-16GB
bitsandbytes: 0.46.1
```

Bu aşamadan sonra ortam paketleri pilot tamamlanana kadar yükseltilmemelidir.

`torch.cuda.is_available()` doğru olduğu hâlde bitsandbytes importu
`cuda_utils...so` derleme hatası verirse hem `.out` hem `.err` dosyası
incelenmelidir:

```bash
cat logs/cfpb-gpu-check-JOBID.out
cat logs/cfpb-gpu-check-JOBID.err
```

Bu hata çoğunlukla compute node üzerindeki `libcuda.so` linker konumu veya
`Python.h` geliştirme başlığıyla ilgilidir. Güncellenmiş tanılama işi her
ikisinin konumunu ve tam traceback'i raporlar. Kesin linker mesajı görülmeden
bitsandbytes veya PyTorch sürümü değiştirilmemelidir.

## 3. Hugging Face modelini indirme

Login node internete erişebiliyorsa:

```bash
export PATH=/arf/scratch/muhtan/ccc/.conda-env/bin:$PATH
python scripts/download_hf_model.py \
  --model Qwen/Qwen3-8B \
  --destination /arf/scratch/muhtan/ccc/hf_models/Qwen3-8B
```

TRUBA'dan Hugging Face erişimi yoksa model başka bir makinede indirilip
`hf_models/Qwen3-8B/` dizinine `rsync` ile kopyalanmalıdır.

## 4. Ön kontrol

```bash
cd /arf/scratch/muhtan/ccc
export PATH=/arf/scratch/muhtan/ccc/.conda-env/bin:$PATH
python -m py_compile scripts/train_qlora.py scripts/evaluate_adapter.py
test -f dataset/splits/train.csv
test -f dataset/splits/validation.csv
test -f hf_models/Qwen3-8B/config.json
```

## 5. Pilot işi gönderme

Tam pilot öncesinde model yükleme, 4-bit quantization, QLoRA eğitimi, adapter
kaydı ve inference zincirini küçük bir smoke testiyle doğrula:

```bash
cd /arf/scratch/muhtan/ccc
mkdir -p logs outputs
sbatch slurm_files/qwen3_8b_qlora_smoke.slurm
```

Smoke testi yalnızca 64 train ve 32 validation örneği kullanır. Başarılı
olduğunda logun sonunda şu satır görünür:

```text
Smoke test completed: /arf/scratch/muhtan/ccc/outputs/...
```

Çıktıları kontrol et:

```bash
cat logs/cfpb-qwen3-smoke-JOBID.out
cat logs/cfpb-qwen3-smoke-JOBID.err
find outputs/qwen3_8b_qlora_smoke_JOBID -maxdepth 2 -type f | sort
```

Smoke testi başarılı olduktan sonra asıl pilotu gönder:

```bash
cd /arf/scratch/muhtan/ccc
mkdir -p logs outputs
sbatch slurm_files/qwen3_8b_qlora_pilot.slurm
```

İlk smoke testi başarıyla tamamlanmıştır:

```text
Train: 64 örnek, 1 epoch, 112,8 saniye
Validation inference: 32 örnek
Accuracy: 0,7813
Macro-F1: 0,7085
Weighted F1: 0,7719
Off-label: 0
Mean latency: 0,966 saniye/örnek
```

Bu küçük örneklem yalnızca teknik doğrulama içindir; bilimsel performans sonucu
olarak raporlanmayacaktır. Sonraki 10.000 örnekli pilot başarıyla tamamlanmış;
eğitim 2 saat 36 dakika, 2.000 örnekli validation inference ise yaklaşık
8 dakika sürmüştür.

## 6. Kontrollü zero-shot validation baseline

Fine-tuning kazancını ölçmek için aynı Hugging Face Qwen3-8B base modeli, aynı
NF4 quantization, aynı prompt, aynı 500-karakter sınırı ve aynı 2.000 validation
örneği üzerinde adaptersız çalıştırılmalıdır:

```bash
sbatch slurm_files/qwen3_8b_validation_baseline.slurm
```

Sonuç:

```bash
cat outputs/qwen3_8b_zero_shot_validation_JOBID/metrics.json
```

Fine-tuning etkisinin ilk kontrollü tahmini:

```text
Delta macro-F1 = fine-tuned validation macro-F1
                 - base zero-shot validation macro-F1
```

İlk 10.000 örnekli pilotun fine-tuned validation sonucu:

```text
Accuracy: 0,8475
Macro-F1: 0,8098
Weighted F1: 0,8439
Off-label: 0
Best checkpoint: step 600
Train runtime: 2 saat 36 dakika 35 saniye
```

Bu değer hâlâ validation sonucudur; internal veya legacy test sonucu değildir.

Kontrollü base-model zero-shot sonucu:

```text
Accuracy: 0,7115
Macro-F1: 0,5993
Weighted F1: 0,6931
Off-label: 1
Mean latency: 0,565 saniye/örnek
```

Fine-tuning kazanımı:

```text
Accuracy: +0,1360
Macro-F1: +0,2104
Weighted F1: +0,1507
Off-label: 1 → 0
Latency: 0,565 → 0,876 saniye/örnek
```

Bu kontrollü pilot sonucu QLoRA'nın belirgin fayda sağladığını gösterir.
Validation verisi model/checkpoint seçimi için kullanıldığından bu skorlar
nihai test bulgusu olarak raporlanmamalıdır.

## 7. Balanced sampling ve sınırlı hiperparametre araması

İlk arama turu üç balanced-training koşulunu çalıştırır. Her koşul 10.000
training örneği kullanır: her sınıftan tam 2.000 kayıt. Validation seti doğal
dağılımda ve bütün koşullarda aynı kalır.

| Array index | Deney | LR | LoRA r | LoRA alpha | Epoch |
|---:|---|---:|---:|---:|---:|
| 0 | E1 balanced baseline | 2e-4 | 16 | 32 | 1 |
| 1 | E2 düşük LR | 1e-4 | 16 | 32 | 1 |
| 2 | E3 düşük rank | 2e-4 | 8 | 16 | 1 |

LoRA `alpha/r` oranı E1 ve E3'te 2 olarak sabit tutulmuştur. Mevcut E0 natural
pilot (`LR=2e-4`, `r=16`) karşılaştırma noktasıdır.

Güncel dosyaları TRUBA'ya kopyala:

```bash
rsync -av \
  scripts slurm_files requirements-truba.txt \
  muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/
```

Array işini gönder:

```bash
cd /arf/scratch/muhtan/ccc
sbatch slurm_files/qwen3_8b_balanced_search.slurm
```

Array durumunu izle:

```bash
squeue -u muhtan
```

Log adları:

```text
logs/cfpb-qwen3-search-ARRAYJOBID_0.out
logs/cfpb-qwen3-search-ARRAYJOBID_1.out
logs/cfpb-qwen3-search-ARRAYJOBID_2.out
```

Her deney şu ek provenance dosyalarını üretir:

- `train_selection_manifest.csv`
- `validation_selection_manifest.csv`
- `run_metadata.json`
- `trainer_state.json`
- `validation_inference/metrics.json`
- `validation_inference/predictions.csv`
- `best_adapter/`

Üç iş tamamlandıktan sonra:

```bash
python scripts/summarize_search.py \
  --outputs-dir outputs \
  --output outputs/qwen3_8b_search_summary.csv
```

Kazanan öncelikle validation macro-F1 ile seçilir. Eşitliğe yakın durumda
`debt_collection` F1, weighted F1, latency ve adapter rank birlikte
değerlendirilir. Bu tur bitmeden 2-epoch deneyi gönderilmez.

Sonuçları yerel arşive al:

```bash
rsync -av --progress \
  muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/outputs/qwen3_8b_search_\* \
  experiment_archive/outputs/

rsync -av --progress \
  muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/logs/cfpb-qwen3-search-\* \
  experiment_archive/logs/
```

Kopyalamadan sonra yerel checksum manifestini yeniden üret:

```bash
bash scripts/create_archive_checksums.sh
```

İzleme:

```bash
squeue -u muhtan
tail -f logs/cfpb-qwen3-8b-JOBID.out
```

## Pilot kapsamı

- GPU: 1 × V100 16 GB
- Precision: FP16 compute
- Quantization: 4-bit NF4 + double quantization
- LoRA target: tüm linear katmanlar
- Train: doğal dağılımdan stratified 10.000 örnek
- Validation loss: stratified 2.000 örnek
- Validation inference: aynı 2.000 örnek
- Anlatılar: ilk 500 karakter
- Maksimum uzunluk: 512 token
- Effective batch size: 16
- Epoch: 1
- Internal test ve legacy test: çalıştırılmaz

Pilot başarılı olduktan sonra tüm train seti, veri ölçeği deneyleri ve birden
fazla seed için ayrı SLURM işleri hazırlanacaktır. Nihai checkpoint seçimi
validation macro-F1 ile yapılmalı; pilotta TRL'nin kaydettiği validation loss
yalnızca ilk teknik doğrulama için kullanılmaktadır.

## Olası V100 bellek ayarları

CUDA OOM oluşursa sırasıyla:

1. `--max-length 384`
2. `--lora-r 8`
3. `--max-eval-samples 1000`

Train batch size zaten 1'dir. Gradient accumulation bellekten çok effective
batch size ve optimizer adım sayısını etkiler.
