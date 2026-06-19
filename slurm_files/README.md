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
└── .venv/
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

Ardından login node üzerinde sanal ortamı oluştur:

```bash
cd /arf/scratch/muhtan/ccc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Yeni bir venv içinde `torch` bulunmaması beklenen durumdur. PyTorch,
`requirements-truba.txt` içine bilinçli olarak eklenmemiştir; CUDA wheel
kaynağının açıkça seçilmesi gerekir.

### Önerilen başlangıç kurulumu

V100 için muhafazakâr ve yaygın uyumluluk seçeneği CUDA 11.8 PyTorch wheel'idir:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-truba.txt
python -m pip freeze > environment.freeze.txt
```

`torchvision` ve `torchaudio` bu proje için gerekli değildir.

PyTorch'u kurduktan sonra login node üzerinde şu kontrol yapılabilir:


python -c "import torch; print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda)"


Burada `torch.cuda.is_available()` değerinin `False` olması login node üzerinde
tek başına hata değildir. Gerçek GPU doğrulaması compute job içinde yapılır.
Pilot SLURM işi başlangıçta `nvidia-smi` çalıştırır ve eğitim scripti CUDA
görülmüyorsa hemen durur.

### Neden doğrudan “en güncel” PyTorch değil?

- `pip install torch` kullanılınca seçilen wheel'in CUDA varyantı yeterince
  açık değildir.
- Nightly/preview paketler tam test edilmemiştir.
- Çok yeni CUDA wheel'i, TRUBA compute node NVIDIA driver'ından daha yeni bir
  runtime isteyebilir.
- Deneylerin yeniden üretilebilirliği için PyTorch ve CUDA varyantı
  `environment.freeze.txt` içinde sabitlenmelidir.

Resmî PyTorch stable sürümünün `cu118` wheel'i V100 için uygun başlangıçtır.
Tanılama çıktısında TRUBA'nın açıkça önerdiği bir PyTorch modülü veya farklı
CUDA sürümü görülürse kurulum komutu buna göre değiştirilmelidir. Sistem modülü
kullanılacaksa önce modül yüklenmeli, ardından gerekirse venv
`--system-site-packages` ile oluşturulmalıdır; iki farklı PyTorch kurulumu aynı
ortamda karıştırılmamalıdır.

## 3. Hugging Face modelini indirme

Login node internete erişebiliyorsa:

```bash
source /arf/scratch/muhtan/ccc/.venv/bin/activate
python scripts/download_hf_model.py \
  --model Qwen/Qwen3-8B \
  --destination /arf/scratch/muhtan/ccc/hf_models/Qwen3-8B
```

TRUBA'dan Hugging Face erişimi yoksa model başka bir makinede indirilip
`hf_models/Qwen3-8B/` dizinine `rsync` ile kopyalanmalıdır.

## 4. Ön kontrol

```bash
cd /arf/scratch/muhtan/ccc
source .venv/bin/activate
python -m py_compile scripts/train_qlora.py scripts/evaluate_adapter.py
test -f dataset/splits/train.csv
test -f dataset/splits/validation.csv
test -f hf_models/Qwen3-8B/config.json
```

## 5. Pilot işi gönderme

```bash
cd /arf/scratch/muhtan/ccc
mkdir -p logs outputs
sbatch slurm_files/qwen3_8b_qlora_pilot.slurm
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
