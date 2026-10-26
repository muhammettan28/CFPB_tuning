# Experiment records

- `gpu_revision_v1/`: corrected-data main study; 500 narrative characters and a 512-token ceiling for the principal neural comparison. Separate long-input and serial-latency conditions retain their own subdirectories.
- `ref14_encoder_v1/`: separately declared BERT/RoBERTa parameter sensitivity from reference [14], not the main baseline or an exact replication.
- `tfidf_revision_v1/`: validation search and fixed-seed checks at both the matched 10,000-row and full retained 120,502-row training budgets.
- `provenance/`: pre-relocation source snapshots needed to interpret original code hashes. Use maintained entry points under `scripts/revision/`.

Per-run metadata, actual fitting selections, validation and final predictions,
metrics, completion records, configurations and available trainer histories are
versioned. Neural weights/optimizer states and prediction memoisation databases
are retained locally but excluded from Git. Final analyses and uncertainty are
under `analysis/`; see the root README for the CPU verification commands.
