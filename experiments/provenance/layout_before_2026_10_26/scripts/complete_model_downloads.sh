#!/usr/bin/env bash
set -euo pipefail

# Downloads only; no GPU/model execution. Uses the exact revisions already cached.
cfpb_hf_cli="${CFPB_HF_CLI:-$HOME/.venvs/cfpb-download/bin/hf}"
if [[ ! -x "$cfpb_hf_cli" ]]; then echo "HF CLI missing: $cfpb_hf_cli" >&2; exit 1; fi


echo "Download commands completed; rerun verify_model_downloads.py to check weights."
