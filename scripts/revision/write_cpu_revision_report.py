#!/usr/bin/env python3
"""Assemble a reviewable report; does not edit the manuscript or original paper."""
from __future__ import annotations
from project_paths import project_path

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from audit_revision_data import ROOT, OUT, ARCHIVE


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    overlap = json.loads((OUT / "legacy_overlap_summary.json").read_text())
    method = json.loads((OUT / "statistical_method.json").read_text())
    leakage = pd.read_csv(OUT / "leakage_overlap_summary.csv")
    stats = pd.read_csv(OUT / "paired_group_bootstrap.csv")
    selected = leakage.loc[leakage.left.str.startswith("selected_train") & leakage.right.eq("internal_test") & leakage.representation.eq("llama_tokens512")]
    selected_rows = [[r.left[-2:], r.shared_unique_inputs, r.left_rows_affected, r.right_rows_affected] for r in selected.itertuples()]
    macro_rows = [[r.model, r.analysis, f"{r.base_point:.4f}", f"{r.ft_seed_mean_point:.4f}", f"{r.ft_seed_sd_ddof1:.4f}", f"{r.paired_gain_point:.4f} [{r.gain_low:.4f}, {r.gain_high:.4f}]"] for r in stats.loc[stats.metric.eq("macro_f1")].itertuples()]
    configuration = []
    for path in sorted(ARCHIVE.glob("*/run_metadata.json")):
        data = json.loads(path.read_text())
        record = {"source": str(path.relative_to(ROOT)), "arguments": data.get("arguments"),
                  "train_rows": data.get("train_rows"), "validation_rows": data.get("validation_rows"),
                  "python": data.get("python"), "torch": data.get("torch"), "gpu": data.get("gpu"),
                  "pip_freeze": data.get("pip_freeze")}
        adapter = path.parent / "best_adapter/adapter_config.json"
        if adapter.exists():
            record["adapter_config"] = json.loads(adapter.read_text())
        configuration.append(record)
    (OUT / "archived_implementation_details.json").write_text(json.dumps({
        "training_evidence": configuration,
        "historical_environment": (project_path(ROOT, "experiment_archive/environment/environment.freeze.txt")).read_text().splitlines(),
        "inference_source": "scripts/evaluate_adapter.py", "training_source": "scripts/train_qlora.py",
        "inference_settings_from_source": {"max_narrative_chars": 500, "max_input_length": 512,
            "max_new_tokens": 32, "do_sample": False, "enable_thinking": False,
            "quantization": "4-bit NF4, double quantization, float16 compute",
            "label_extraction": "regex parsing; no token-level constrained decoding"},
        "source_sha256": {name: hashlib.sha256((project_path(ROOT, name)).read_bytes()).hexdigest() for name in ["scripts/evaluate_adapter.py", "scripts/train_qlora.py"]},
        "limits": ["Final Qwen training adapters/manifests/metadata are not present in the local archive; pilot/smoke metadata do not prove the final training configuration.",
                   "Adapter base_model_name_or_path is a local cluster path; revision is null. Exact upstream model commit IDs cannot be inferred.",
                   "No new GPU inference, training, or controlled latency measurement was performed."],
    }, indent=2) + "\n")
    baseline_rows = []
    for name in ["tfidf_logreg_10000", "tfidf_logreg_full"]:
        path = project_path(ROOT, "CLASSIC_TEST/outputs", name, "internal_test/metrics.json")
        d = json.loads(path.read_text())
        baseline_rows.append([name, f"{d['macro_f1']:.4f}", "Eski tek-seed sonucu; yeni validation search değil"])
    text = f"""# CPU revizyon denetimi — 20 Ekim 2026

Bu çalışma yalnızca mevcut veriler ve arşivlenmiş tahminleri kullanır. Yeni dosyalar `REVISION_PAPER/` altına yazıldı; `ORJ_PAPER/`, özgün veri bölmeleri, deney arşivi ve makale metni değiştirilmedi.

## En önemli bulgu: 500 karakter sonrasında bölmeler örtüşüyor

Tam normalize metne göre ayrık olan train/validation/test bölmeleri, modelin gördüğü ilk 500 karakterde ayrık değil. Tüm eğitim havuzuyla internal test arasında **192 benzersiz girdi / 584 test satırı** ortak. Eğitim ve validation havuzlarının birleşimiyle çakışan bağımlılık grupları çıkarıldığında testten **{method['sensitivity']['removed_rows']} satır** çıkarılıyor; **{method['sensitivity']['internal_clean_rows']} satır** kalıyor.

Gerçek Llama eğitim seçim manifestleri mevcut ve seed bazında yeniden oluşturulan örneklemle kayıt sırası ve narrative hash düzeyinde birebir doğrulandı:

{markdown_table(['Seed', 'Ortak benzersiz girdi', 'Etkilenen eğitim satırı', 'Etkilenen test satırı'], selected_rows)}

Bu eşleşmeler ham 500 karakter, normalize 500 karakter, serileştirilmiş prompt ve yeniden oluşturulan tokenizer girdilerinde görülüyor. Her iki arşiv tokenizer'ıyla incelenen 123.551 benzersiz 500-karakter girdisinin hiçbiri prompt dahil 512 token sınırını aşmadı.

Qwen için final eğitim seçim manifestleri ve final tokenizer dosyaları eksik. Seçimler mevcut örnekleme kodu ve Slurm protokolünden yeniden oluşturuldu; Qwen prompt denetimi pilot tokenizer'ına dayanıyor. Bu nedenle Llama için mevcut olan bağımsız kanıt düzeyi Qwen için iddia edilmiyor. Ayrıca denetim inference serileştirmesini yeniden üretir; SFT trainer'ın tam completion/truncation yolunu sertifikalandırmaz.

`leakage_overlap_summary.csv` tüm 250/500 legacy varyantlarını ayrı ayrı inceler. Eğitim seçimleri ile legacy arasında birebir 500-karakter/token girdisi eşleşmesi bulunmadı; ancak kısa legacy girdisinin daha uzun bir eğitim girdisinin öneki olması bu denetimden kaçabilir. Bu yüzden `legacy_short_prefix_audit.csv` ayrıca ilk 250 karakteri karşılaştırır ve bazı önek eşleşmeleri bulur. Bunlar aynı şikâyet kimliğinin kanıtı değil, muhafazakâr bağımlılık işaretleridir. İlgili tüm legacy grupları çıkarılan duyarlılık kümesinde **{method['sensitivity']['legacy_prefix_clean_rows']} / 12.000 satır** kalır.

## Legacy alt kümelerinin bağımsızlığı

12 alt kümedeki **{overlap['rows']} satır**, **{overlap['unique_normalized_narratives']} benzersiz normalize metin / etiket-metin çifti** içeriyor. Metin ve etiket-metin için ayrı 12×12 ortak-sayı, satıra göre örtüşme oranı ve Jaccard matrisleri üretildi.

250 ve 500 karakterlik farklı görünümlerin bağımlılığını da korumak için tam normalize metin VEYA normalize ilk 250 ham karakteri paylaşan kayıtlar bağlı bileşenlerde birleştirildi: **{overlap['conservative_prefix250_groups']} muhafazakâr grup**. Bunlar doğrulanmış benzersiz CFPB şikâyet kimlikleri değildir; aynı öneki paylaşan farklı şikâyetleri birleştirebilir. 5.909 benzersiz görünümün her biri eşit ağırlık alır, alt kümelerdeki tekrarları yeni bağımsız gözlem sayılmaz.

## Eşleştirilmiş grup bootstrap sonuçları

Her analizde **10.000 tekrar**, sabit RNG seed `{method['rng_seed']}` ve yüzde 95 percentile güven aralığı kullanıldı. Aynı grup örneklemi base ve üç FT seed için ortak kullanılıyor. Grupların bütün satırları birlikte yeniden örnekleniyor; grup içindeki boyut farklılıkları korunuyor. Legacy tekrarları benzersiz görünüm başına toplam ağırlık 1 olacak şekilde ağırlıklandırılıyor. Tekrarlanan normalize legacy görünümlerinin tahminleri tüm modeller/seeds için tutarlı bulundu.

Nokta tahmini özgün gözlemlerden hesaplanır; bootstrap ortalamasıyla değiştirilmez. FT skoru, üç seed'in ayrı ayrı hesaplanan metriklerinin aritmetik ortalamasıdır. Seed SD'si `ddof=1` ile ayrı raporlanır; bu bootstrap sabit eğitilmiş modellere koşulludur, eğitim rastgeleliğini güven aralığına katmaz.

{markdown_table(['Model', 'Analiz', 'Base macro-F1', 'FT seed ort.', 'Seed SD', 'Eşleştirilmiş kazanç [GA]'], macro_rows)}

`internal_conservative_clean` tüm train/validation havuzlarındaki girdi eşleşmelerini çıkarır; yalnızca gerçek 10k/2k örneklemlere göre temizlemekten daha katıdır. `legacy_conservative_prefix_clean` kısa önek eşleşmelerinin gruplarını çıkarır. Her ikisi de post-hoc duyarlılık analizidir: **eski eğitimi veya model seçimini onarmaz ve yeni, girdi düzeyinde ayrık deneylerin yerine geçtiği iddia edilemez.** Eski subset-level Wilcoxon testi bu çalışma tarafından doğrulanmış sayılmaz; grup bootstrap farklı bir estimandı değerlendirir.

## Diğer tamamlanan CPU işleri

- Veri akışı ve record-ID kontrolü: 162.415 mevcut kaynak satırı → 10 eksik satır çıkarılması → 162.405 kullanılabilir satır → 8.123 legacy bağlantılı satır çıkarılması → 154.282 satır → 123.427 eğitim / 15.427 validation / 15.428 test. Hazır bölmelerin ID, etiket ve metinleri mevcut `complaints.csv` ile doğrulandı.
- `complaints.csv` zaten beş etiketli ve ön işlenmiş. Özgün CFPB indirmesinden bu dosyaya geliş, orijinal ürün-etiket eşlemesi ve ön işleme adımları mevcut dosyadan geriye doğru kesin olarak çıkarılamıyor. Bunlar yazar kaydı veya eski veri hazırlama kaynağı gerektiriyor.
- Mevcut tahminlerden internal metin uzunluğuna ve legacy 250/500 × dengeli/dengesiz koşuluna göre betimsel sonuçlar hesaplandı (`descriptive_truncation_strata.csv`). Bu sonuçlar kontrollü uzun-bağlam ablation'ı değildir; içerik ve örneklem bileşimi değişebilir.
- Arşivlenmiş eğitim argümanları, LoRA config'leri, kütüphane sürümleri ve inference ayarları `archived_implementation_details.json` içine çıkarıldı. Kesin upstream checkpoint commit kimlikleri arşivden doğrulanamıyor.

Mevcut klasik baseline değerleri, yeni tuning sonucu olarak sunulmadan:

{markdown_table(['Baseline', 'Internal macro-F1', 'Durum'], baseline_rows)}

## Kalan işlerin işlem ihtiyacı

| İş | İşlem ihtiyacı / durum |
| --- | --- |
| Modelin gördüğü girdilere göre grup ayrımını düzeltme ve yeni split manifestleri | CPU; yeni karşılaştırmalı deneylerden önce protokol sabitlenmeli |
| TF-IDF + LR validation search ve üç seed | CPU; henüz yeni tuning yapılmadı. Girdi ayrımı düzeltilmeden yeni final baseline çalıştırılmadı |
| Encoder validation search ve üç seed | Pratik olarak GPU ile yürütülmeli; mevcut tahminler yeni eğitimin yerini tutmaz |
| 500 karakter ile daha uzun girdinin kontrollü karşılaştırması | LLM için yeni GPU inference; eğitim bağlamı da değişirse yeni GPU eğitimi |
| Düzeltilen ayrımla LLM/encoder deneylerini tekrar etme | GPU |
| Aynı donanım, warmup ve tekrarlarla latency ölçme | GPU; mevcut latency kayıtlarından kontrollü ölçüm üretilemez |
| [14]/[15] karşılaştırması, iddiaları daraltma ve hakem yanıtı | GPU gerektirmez; bu çalışmada makale metni değiştirilmedi |
| Özgün CFPB veri çıkarımı ve etiket mapping kaydı | Eksik kaynak/yazar bilgisi; hesaplama sorunu değil |
| Reviewer erişim paketi | CPU ile hazırlanabilir; yayımlama/harici paylaşım yapılmadı |

## Yeniden çalıştırma ve doğrulama

Analiz paketlerini mevcut global ortama dokunmadan `/tmp/cfpb-revision-venv` içinde kullandık. Proje kökünden:

```bash
python -m venv --system-site-packages /tmp/cfpb-revision-venv
/tmp/cfpb-revision-venv/bin/python -m pip install -r scripts/revision/requirements-cpu-revision.txt
/tmp/cfpb-revision-venv/bin/python scripts/revision/audit_revision_data.py
/tmp/cfpb-revision-venv/bin/python scripts/revision/analyze_archived_predictions.py --replicates 10000
/tmp/cfpb-revision-venv/bin/python scripts/revision/write_cpu_revision_report.py
/tmp/cfpb-revision-venv/bin/python -m unittest discover -s scripts/revision -p test_cpu_revision.py -v
```

Bilimsel doğrulamalar: off-label hataları ve ağırlıklı F1, sklearn ile karşılaştırıldı; eşleştirme ve eşit olmayan küme boyutlarını koruyan bootstrap'ın bilinen iki-kümeli dağılımı doğrulandı; etiketten bağımsız geçişli grup birleştirme kontrol edildi. Kullanılan 104 final prediction dosyasının tamamında yeniden hesaplanan accuracy/macro-F1/weighted-F1 arşiv özetleriyle 1e-12 toleransta karşılaştırıldı. Internal tahminler ID ile birleştirildi; legacy'de ID olmadığından kaynak sırası, tüm etiketler ve satır sayısı kontrol edildi (bu sınırlama `prediction_sources.json` içinde kayıtlı).

Büyük satır düzeyi audit cache'leri ve bootstrap çekilişleri bu klasörün `.gitignore` dosyasıyla Git dışında tutulur; özetler ve yöntem raporları kalır.

Yöntem kontrolü için başvurulan birincil belgeler: [SciPy eşleştirilmiş bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [Hugging Face chat template ve özel token davranışı](https://huggingface.co/docs/transformers/v4.47.0/en/chat_templating). Grup örneklemesi bu projede açıkça uygulanmıştır; SciPy'nin standart satır örneklemesi doğrudan kullanılmamıştır. İnference denetimi, mevcut kodun `add_special_tokens=True` davranışını değiştirmeden yeniden üretir.
"""
    (OUT / "CPU_REVISION_REPORT.md").write_text(text, encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "transformers", "threadpoolctl"]}
    (project_path(ROOT, "scripts/revision/requirements-cpu-revision.txt")).write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    print(OUT / "CPU_REVISION_REPORT.md")


if __name__ == "__main__":
    main()
