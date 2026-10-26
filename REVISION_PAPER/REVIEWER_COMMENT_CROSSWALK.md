# Reviewer requests and final response locations

This index refers to the completed revised documents. R1/R2 IDs are editorial tracking IDs, not original reviewer numbering. All original wording is retained below; the complete original text, including introductory comments, is in [the original extract](../analysis/gpu_preparation/original_reviewer_comments.txt).

[Full response letter](RESPONSE_TO_REVIEWERS.md) · [Machine-readable final coverage](../analysis/postprocessing/REVIEWER_COVERAGE_FINAL.json)

Reproducibility repository: https://github.com/muhammettan28/CFPB_tuning. The responses describe the completed analyses and their scientific limitations; each entry below links the reviewer comment to the corresponding document locations.

## R1.1

> Legacy statistics: The 12 subsets contain only 5,909 unique pairs, so the Wilcoxon test (p = 2⁻¹²) assumes independence that isn’t shown. Provide an overlap matrix and use a group-aware paired analysis.

**Response:** see R1.1 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.5 (p. 6); Manuscript Section 4.2 (p. 8); Supplementary material Section S4 (p. 11); Supplementary material Table S17 (p. 11); Supplementary material Table S18 (p. 11); Supplementary material Table S13 (p. 9).

## R1.2

> Leakage audit: Splits group full narratives, but models see 500-character truncations. Audit overlap after truncation and prompt construction, and narrow the “no leakage” claim.

**Response:** see R1.2 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.2 (p. 4); Manuscript Table 2 (p. 4); Supplementary material Section S1 (p. 2); Supplementary material Table S3 (p. 2); Supplementary material Section S7 (p. 16); Supplementary material Section S8 (p. 17).

## R1.3

> Reproducibility: Explain how the 162,415 rows were derived, give the label mapping and subset construction, and release code, manifests and predictions to reviewers.

**Response:** see R1.3 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.1 (p. 3); Manuscript Section 3.2 (p. 4); Manuscript Section 6 (p. 11); Supplementary material Section S1 (p. 2); Supplementary material Table S16 (p. 11); Supplementary material Section S8 (p. 17).

## R1.4

> Implementation details: State the Qwen3 thinking setting, checkpoint IDs, full QLoRA configuration and library versions.

**Response:** see R1.4 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.3 (p. 4); Supplementary material Table S4 (p. 3); Supplementary material Table S5 (p. 3); Supplementary material Table S6 (p. 4); Supplementary material Section S8 (p. 17).

## R1.5

> Baselines: Tune the encoders and TF-IDF comparably (three seeds, validation search), and compare in one table with uncertainty. Margins over TF-IDF are small.

**Response:** see R1.5 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.4 (p. 5); Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Supplementary material Table S7 (p. 4); Supplementary material Table S8 (p. 5); Supplementary material Table S13 (p. 9); Supplementary material Section S7 (p. 16); Supplementary material Table S28 (p. 16).

## R1.6

> Bootstrap: Use a paired, group-level bootstrap with more replicates, report seed variation separately, and reconcile Tables 2 and 3.

**Response:** see R1.6 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.5 (p. 6); Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Supplementary material Section S3 (p. 6); Supplementary material Table S11 (p. 8); Supplementary material Table S32 (p. 18); Supplementary material Table S33 (p. 18).

## R1.7

> Contribution and claims: Compare explicitly with [14] and [15], and remove or test claims about reuse, “external” benchmark and robustness.

**Response:** see R1.7 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Table 1 (p. 3); Manuscript Section 3.1 (p. 3); Manuscript Section 5 (p. 10); Manuscript Section 6 (p. 11); Supplementary material Section S7 (p. 16).

## R1.8

> Truncation and deployment: Report results by truncation and add a longer-context experiment. Control the latency measurements and avoid claiming acceptability or regulatory readiness.

**Response:** see R1.8 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.6 (p. 6); Manuscript Table 5 (p. 9); Manuscript Table 6 (p. 10); Manuscript Table 7 (p. 10); Manuscript Section 6 (p. 11); Supplementary material Section S5 (p. 12); Supplementary material Table S23 (p. 14); Supplementary material Table S24 (p. 14); Supplementary material Section S6 (p. 14); Supplementary material Table S26 (p. 15); Supplementary material Table S35 (p. 20); Supplementary material Table S36 (p. 20).

## R2.1

> The authors should strengthen the positioning against closely related recent studies, especially Reference [14], and clarify the distinction between the proposed approach and prior LLM-based complaint classification methods.

**Response:** see R2.1 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Table 1 (p. 3); Manuscript Section 5 (p. 10); Supplementary material Section S7 (p. 16).

## R2.2

> The term “external benchmark” should be reconsidered because the benchmark originates from the same CFPB data source.

**Response:** see R2.2 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.1 (p. 3); Manuscript Section 3.5 (p. 6); Supplementary material Section S4 (p. 11).

## R2.3

> The manuscript would also benefit from a more complete description of the QLoRA and inference configurations and clearer justification of the 500-character truncation and 10,000-example training budget.

**Response:** see R2.3 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.3 (p. 4); Manuscript Section 3.4 (p. 5); Manuscript Table 5 (p. 9); Supplementary material Section S2 (p. 3); Supplementary material Section S7 (p. 16); Supplementary material Table S34 (p. 19).

## R2.4

> The statistical evaluation and bootstrap methodology should be described more clearly.

**Response:** see R2.4 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.5 (p. 6); Supplementary material Section S3 (p. 6); Supplementary material Table S11 (p. 8); Supplementary material Table S33 (p. 18).

## R2.5

> Claims concerning constrained decoding, robustness, and deployment should also be moderated where appropriate.

**Response:** see R2.5 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Section 3.3 (p. 4); Manuscript Section 4.3 (p. 8); Manuscript Section 4.4 (p. 8); Manuscript Section 6 (p. 11); Supplementary material Table S15 (p. 10); Supplementary material Section S6 (p. 14).

## R2.6

> Most importantly, the results show that QLoRA achieves performance very close to the TF-IDF + Logistic Regression baseline; therefore, the practical contribution should be framed around the benefits and trade-offs of LLM-based PEFT rather than implying clear classification superiority.

**Response:** see R2.6 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Manuscript Table 6 (p. 10); Manuscript Table 7 (p. 10); Manuscript Section 5 (p. 10); Manuscript Section 7 (p. 11); Supplementary material Table S35 (p. 20); Supplementary material Table S36 (p. 20).

## R2.7

> Finally, improving public access to the code and experimental configurations would substantially improve reproducibility.

**Response:** see R2.7 in [the response letter](RESPONSE_TO_REVIEWERS.md).

**Changes and locations:** Supplementary material Section S8 (p. 17); Supplementary material Table S30 (p. 17); Manuscript Section 6 (p. 11).
