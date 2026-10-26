#!/usr/bin/env python3
"""Create an honest progress report, reproducibility guide and rebuttal draft."""
import json
from pathlib import Path
import pandas as pd

from prepare_revision_protocol import ROOT, STUDY
from tune_revision_tfidf import EXP


def main():
    from prepare_reviewer_crosswalk import final_response_protected
    if final_response_protected():
        print('Final documents preserved; legacy CPU status/response generator skipped')
        return
    protocol=json.loads((STUDY/'protocol.json').read_text())
    source=json.loads((STUDY/'source_archive_trace.json').read_text())
    stats=pd.read_csv(STUDY/'tfidf_group_bootstrap.csv',dtype={'budget':str})
    paired=pd.read_csv(STUDY/'paired_truncation_metrics.csv')
    lines=['| Budget | Internal macro-F1 [95% CI] | Internal weighted F1 | Legacy unique-input macro-F1 |',
           '| --- | --- | --- | --- |']
    for budget in ['10000','full']:
        x=stats.loc[stats.budget.eq(budget)&stats.role.eq('internal_test')].set_index('metric')
        leg=stats.loc[stats.budget.eq(budget)&stats.role.eq('legacy_unique')&stats.metric.eq('macro_f1')].iloc[0]
        n=next(r['retained_rows'] for r in protocol['split_sizes'] if r['partition']=='train') if budget=='full' else 10000
        lines.append(f"| {n:,} | {x.loc['macro_f1','mean']:.4f} [{x.loc['macro_f1','low']:.4f}, {x.loc['macro_f1','high']:.4f}] | {x.loc['weighted_f1','mean']:.4f} | {leg['mean']:.4f} |")
    table='\n'.join(lines)
    report=f'''# CPU revizyonu — tamamlanan işler ve sınırlar

20 Ekim 2026. ORJ_PAPER, eski veri dosyaları ve deney arşivi korunarak çalışıldı. GPU kullanılmadı.

## 1. Kaynak ve veri ayrımı

`dataset/Dataset.zip` içindeki 162.421 satırdan son altı `name`-only placeholder çıkarıldığında çalışma dosyasının 162.415 satırı **ID, etiket ve metin düzeyinde birebir** elde ediliyor. Kalan kayıtlar değiştirilmemiş. Projenin önceki makalesi Tiwari/Kaggle kaynağını ve CFPB kökenini kaydediyor. Beş etiket arşivde zaten mevcut; güncel pipeline bunları aynen koruyor. Ön-Kaggle ham kategori birleştirme kodu bulunmadı ve varmış gibi sunulmuyor.

Akış: 162.421 → 162.415 (6 placeholder) → 162.405 (10 eksik kayıt) → 154.282 (8.123 eski legacy dışlaması) → 151.050 (3.232 yeni karantina).

Revised train **120.502**, validation **15.120**, internal test **15.428**. Test ve 12 legacy dosyası byte-for-byte korundu. 25 ayrım kontrolünde retained train/validation ile değerlendirme arasında denetlenen girdi/grup paylaşımı sıfır. Değerlendirme önceden görülmüş olduğundan yeni kör test olarak sunulmuyor. Qwen tokenizer denetimi pilot artifact'ına dayanıyor; yeni/final tokenizer kimliği ayrıca doğrulanmalı.

Kanıtlar: `source_archive_trace.json`, `source_placeholder_exclusions.csv`, `label_identity_mapping.csv`, `protocol.json`, `role_assignment_manifest.csv`, `separation_checks.csv`.

## 2. CPU baseline tuning

İki bütçe için ayrı 48 aday (toplam 96) yalnızca sabit 2.000 validation örneğinde arandı. Train vocabulary/IDF fitting dışında hiçbir set fitting'e girmedi. İki seçimin de konfigürasyonu test tahminlerinden önce hash ile sabitlendi. Her bütçe üç seed etiketiyle yeniden fit edildi; lbfgs deterministik olduğundan tahminler birebir aynı. Bu üç tekrar, stokastik eğitimde üç bağımsız seed sonucu gibi sunulmuyor. Sıfır SD ile grup-bootstrap örnekleme belirsizliği ayrı raporlanıyor.

{table}

Bu skorlar eski QLoRA skorlarıyla düzeltilmiş bir model sıralaması olarak sunulmuyor: eğitim havuzu ve sabit seçimler değişti. GPU modelleri aynı revised seçimlerle yeniden eğitilmeli.

Sayısal yeniden üretim kontrolü, aramada kullanılan `fit_transform(train)` ile ilk refit'teki `transform(train)` yolunun CSR sıralaması/1e-16 düzeyindeki farklarının solver durma yolunu etkilediğini gösterdi. Aynı seçilmiş konfigürasyonlar aramadaki hesaplama yoluyla yeniden fit edildi; vocabulary/IDF ve seçilmiş validation skorları birebir doğrulandı. Arama, veri ve seçim kuralı değiştirilmedi. Eski refit denemesinin kaydı `REFIT_NUMERICAL_AUDIT.json` ve `refit_numerical_audit/` altında tutulur; tabloda yalnızca son doğrulanmış sonuçlar vardır.

## 3. İstatistik ve tablolar

Eski LLM sonuçları açıkça **archived** olarak etiketlendi. Aynı metrik fonksiyonu gözlenen nokta tahminini, seed/run özetini ve 10.000 tekrarlı eşleştirilmiş grup bootstrap'ını üretiyor. Bootstrap örneklem ortalaması nokta tahmini yerine kullanılmıyor. Legacy'nin 5.909 görünümü, bağımlılığı koruyan 3.919 muhafazakâr grupta değerlendiriliyor. Alt küme düzeyindeki bağımsızlık/Wilcoxon çıkarımları çıkarıldı. Yeni TF-IDF skorları ayrı tabloda.

## 4. Makale metni

Abstract, veri/yöntem, sonuçlar, Discussion, Limitations, Conclusion ve veri/AI beyanları tutarlı bir çalışma taslağına dönüştürüldü. External benchmark, garanti edilmiş sızıntısızlık, token-level constrained decoding, broad robustness, taxonomy-reuse kanıtı ve deployment/regulatory kabul iddiaları daraltıldı. [14]/[15] konumlandırması korundu. Güncel PDF, CPU tamamlandığını ama GPU rerun'larının beklediğini açıkça gösterir; gönderime hazır final sürüm olarak sunulmaz.

## 5. Kırpma analizi

Kaydedilmiş tahminlerden uzunluk kohortları ve aynı seed/dağılım içindeki tekil 250/500-prefix eşleşmeleri incelendi. Birden fazla adayı olan eşleşmeler, sonuçlara bakılmadan dışlandı. Farklı koşullarda tekrarlanan kısa/uzun çiftler eşit toplam ağırlık alıyor. Bu retrospektif eşleştirme, orijinal CFPB complaint ID doğrulaması veya 500 karakterden uzun yeni bir deney değildir.

Kanıtlar: `truncation_pair_counts.csv`, `truncation_pair_manifest.csv`, `paired_truncation_metrics.csv`, `truncation_method.json`.

## 6. Yanıt ve yerel paket

`REVISION_PAPER/RESPONSE_TO_REVIEWERS.md` her yorum için yapılan işi ve açık kalan kısmı ayırır. Yerel reviewer ZIP'i kod, kaynak/derived veri, split/selection manifestleri, arşiv tahminleri, yeni CPU tahminleri ve konfigürasyonları içerir. SHA-256 manifesti ile doğrulanır. Yayınlama, reviewer erişimi veya harici paylaşım yapılmaz; yerel arşiv, paylaşılmış bir repository olarak tanıtılmaz. Base-model weights ve üçüncü taraf makale PDF'leri pakete girmez.

## GPU hazır olduğunda kalan işler

1. Yeni 10k/2k sabit seçimlerle Qwen ve Llama üç-seed rerun'ları. Nihai tokenizer/checkpoint commit hash'lerini kaydet; girdi denetimini o artifact'larla tekrarla.
2. Encoder'lar için validation search ve üç-seed corrected-data sonuçları. Eski encoder script'i ayrı validation search içermediğinden bu iş tamamlanmış sayılmaz.
3. Aynı test/seed üzerinde daha uzun input deneyi; actual token/truncation denetimi dahil.
4. Aynı hardware, warmup, açık senkronizasyon ve tekrarlarla latency ölçümü.
5. Sonrasında tek corrected-data karşılaştırma tablosu ve final Abstract/Conclusion; gerçek reviewer erişim bilgisi.

## Yeniden üretim

Çalışan CPU ortamının sürümleri `scripts/requirements-cpu-revision.txt` içinde. Script'ler proje kökünden yürütülür. `prepare_revision_protocol.py` dondurulmuş protokolü sessizce üzerine yazmaz. Önceden görülmüş test skorlarına göre veri/arama alanı değiştirilmemelidir.

```bash
python REVISION_PAPER/scripts/trace_source_archive.py
python REVISION_PAPER/scripts/audit_revision_data.py
# Sadece yeni bir çalışma kopyasında, protokol daha önce dondurulmamışsa:
python REVISION_PAPER/scripts/prepare_revision_protocol.py
python REVISION_PAPER/scripts/tune_revision_tfidf.py
python REVISION_PAPER/scripts/build_revision_analysis.py
python REVISION_PAPER/scripts/analyze_revision_truncation.py
python -m unittest discover -s REVISION_PAPER/scripts -p test_cpu_revision.py -v
```

Birincil yöntem belgeleri: [scikit-learn veri sızıntısı/fitting sınırı](https://scikit-learn.org/stable/common_pitfalls), [LogisticRegression solver ve random_state davranışı](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Üç lbfgs refit'inin yorumu ve train-only fitting sınırı bu belgelerle kontrol edildi.
'''
    (STUDY/'CPU_REVISION_COMPLETION_REPORT.md').write_text(report)
    comments=[
      ('R1.1','Legacy statistics and subset dependence','CPU analysis completed',
       'We quantified pairwise overlap using 12-by-12 shared-count, row-fraction and Jaccard matrices for both normalized narratives and label-narrative units. The 12,000 appearances contain 5,909 unique visible label-narrative units. We now use a paired cluster bootstrap with 10,000 replicates, retaining short/long-view dependence in 3,919 conservative groups. Subset-level Wilcoxon significance claims have been removed. Group identities are conservative text dependencies, not verified complaint IDs.',
       'analysis/cpu_revision/legacy_overlap_*.csv; tables/table_revision_archived_legacy.tex; figures/revision_legacy_overlap.pdf'),
      ('R1.2','Leakage after truncation and prompt construction','Audit and revised allocation completed; GPU reruns pending',
       'We audited full-normalized text, the first 500 characters, serialized prompts and reconstructed 512-token inputs. Recorded Llama selections identify 383/375/390 internal-test rows exposed in the three historical training selections. Existing test/legacy files were preserved; 2,925 training and 307 validation rows were quarantined using label-blind connected components. Twenty-five retained-partition representation checks pass. New model results are not claimed until corrected-data GPU training is repeated; Qwen tokenizer provenance remains explicitly limited.',
       'analysis/revision_v1/protocol.json; separation_checks.csv; analysis/cpu_revision/tokenizer_audit_evidence.json'),
      ('R1.3','162,415-row derivation, label mapping and reviewer artifacts','Local derivation/mapping/package completed; external access pending',
       'The source ZIP contains 162,421 rows. The working copy exactly retains its first 162,415 rows; the six omitted trailing narratives contain only the placeholder name. Ten further records lack required fields. We document the complete local numerical flow and show the identity mapping of the five labels already provided in the processed Tiwari resource. We do not claim to recover unavailable pre-Kaggle raw-product aggregation. A local code/data/manifests/predictions package is prepared and hash-verified; public or reviewer access has not yet been provided.',
       'analysis/revision_v1/source_archive_trace.json; source_placeholder_exclusions.csv; label_identity_mapping.csv; reviewer_package/'),
      ('R1.4','Qwen thinking, checkpoints, full QLoRA configuration and library versions','Documented available evidence; immutable checkpoint gaps remain',
       'The Methods now specify thinking disabled, NF4/double quantization/FP16, LoRA rank/alpha/dropout/targets, completion-only loss, optimizer/schedule/warmup, gradient checkpointing, batch/accumulation and generation settings, with recorded library versions. Missing immutable upstream commits and final Qwen training metadata are stated explicitly rather than reconstructed by assumption. New GPU runs must record these artifacts.',
       'analysis/cpu_revision/archived_implementation_details.json; manuscript.tex Methods'),
      ('R1.5','Comparable TF-IDF/encoder tuning, three seeds and consolidated comparison','TF-IDF completed; corrected-data encoder/LLM comparison pending GPU',
       'We completed separate validation-only 48-candidate searches for 10k and full-data TF-IDF/LR budgets. Both selections were frozen before final evaluation; three same-data lbfgs refits give identical predictions, reported as determinism checks with SD zero and separate group-bootstrap intervals. The source solver is deterministic, so these are not presented as stochastic seed evidence. Comparable encoder tuning and three-seed corrected-data GPU runs are still pending; archived and new-training scores are kept separate.',
       'experiments/tfidf_revision_v1/budget_*/validation_search.csv; selected_config.json; repetition_checks.json; tables/table_revision_tfidf.tex'),
      ('R1.6','Paired group bootstrap, more replicates, seed variation and Tables 2/3','CPU reanalysis completed',
       'All confidence intervals now use 10,000 paired group-bootstrap replicates with common sampled group multiplicities for base and fine-tuned variants. Original-data point estimates and intervals share one confusion-count metric implementation. We separately report sample SD across archived runs and state that their selections also varied with seed. Intervals condition on fixed predictions and do not include training or hyperparameter-selection uncertainty. Table numbers may change; labels/cross-references are generated by LaTeX.',
       'scripts/analyze_archived_predictions.py; build_revision_analysis.py; analysis/cpu_revision/paired_group_bootstrap.csv'),
      ('R1.7','Explicit [14]/[15] comparison and claims about reuse/external benchmark/robustness','Text revision completed',
       'The Introduction, Related Work and comparison table acknowledge the prior financial Llama fine-tuning task in Gullu et al. and distinguish it from Roumeliotis et al. zero-shot evaluation. The contribution is the specified within-model adaptation protocol and its trade-offs, not a first LLM application or cross-paper score superiority. Legacy evaluation is described as source-related; taxonomy reuse, broad robustness and methodological superiority beyond audited evidence are not claimed.',
       'analysis/literature/REF14_POSITIONING.md; manuscript.tex Introduction/Related Work/Discussion'),
      ('R1.8','Truncation results, longer context and controlled deployment measurements','Retrospective CPU analyses completed; new inference/latency pending GPU',
       'We report archived outcomes by full narrative length and perform a one-to-one, ambiguity-filtered paired 250/500 legacy-view analysis with group resampling. The latter uses archived predictions and cannot replace a controlled experiment beyond 500 characters. A longer-context GPU experiment and warmup/repetition-controlled latency study remain pending. Archived latency is reported descriptively without acceptability or regulatory-readiness claims.',
       'analysis/revision_v1/paired_truncation_metrics.csv; truncation_method.json; manuscript.tex archived latency limitations'),
    ]
    response=['# Response to reviewers — working draft','',
      '**Revision evidence record:** [REVISION_LOG.md](REVISION_LOG.md) records completed work, superseded numerical results, supporting files and pending tasks. Consult it when updating these responses.','',
      '20 October 2026. This draft describes only completed computations and text changes. Corrected-data GPU experiments and external reviewer access remain pending. Comment headings below are topic summaries of the original comments in Revision.odt; they are not presented as verbatim quotations.','',
      'The historical LLM runs and newly corrected-data CPU baselines use different training allocations. They are not combined into a purported fair final model ranking.','']
    for key,title,status,body,evidence in comments:
        response.extend([f'## {key}: {title}','',f'**Status:** {status}.','',body,'',f'**Evidence:** `{evidence}`.',''])
    response.extend(['## Reviewer #2','',
      'The comments overlap with the major concerns above; the following mapping records the changes without implying completion of the pending GPU work.','',
      '| Concern | Response/evidence | Remaining work |','| --- | --- | --- |',
      '| Positioning against [14] | R1.7; Introduction, Related Work and explicit protocol table | Final reviewer check |',
      '| External benchmark terminology | Source-related preserved legacy benchmark throughout revised text | None for terminology |',
      '| QLoRA/inference detail | R1.4; archived versions/configs documented | Exact model revisions and missing final Qwen metadata |',
      '| 500 characters and 10k budget | Resource-budget framing; R1.8 matched/stratified CPU analysis | Controlled longer-context GPU experiment |',
      '| Statistical clarity | R1.1/R1.6; point estimates unified, paired group bootstrap, variability sources separated | New GPU predictions will need the same pipeline |',
      '| Decoding, robustness and deployment claims | Strict regex parsing terminology; narrowed conclusions | Controlled latency/inference work |',
      '| Similar performance to TF-IDF | R1.5; stronger validation-selected CPU baseline; no superiority claim | Matched corrected-data LLM/encoder ranking |',
      '| Public access and reproducibility | Local hash-verifiable reviewer package | Real access mechanism and upstream artifact limitations |','',
      '## Submission gates still open','',
      '- Corrected-data QLoRA/encoder training and validation search, with fixed selections and three optimisation seeds.',
      '- Controlled input beyond 500 characters and latency measurements.',
      '- Final immutable checkpoint/tokenizer evidence.',
      '- Actual reviewer/public access arrangement, anonymity/licensing review as required by the journal, and author review of the final paper.',''])
    (ROOT/'REVISION_PAPER/RESPONSE_TO_REVIEWERS.md').write_text('\n'.join(response))
    # Preserve the exact-comment audit and expanded R2 responses on regeneration.
    from prepare_reviewer_crosswalk import main as refresh_reviewer_coverage
    refresh_reviewer_coverage()
    print('Written completion report and item-by-item reviewer response draft')


if __name__=='__main__':main()
