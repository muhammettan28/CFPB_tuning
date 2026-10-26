#!/usr/bin/env python3
"""CPU-only audit against actual pinned tokenizers, before any GPU model is loaded."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from revision_gpu_common import (CONFIG, DATA, PREP, PROTOCOL_SHA, artifact_entry, json_text,
    llm_encoding, load_frame, load_plan, sha_file, sha_text, sft_record, verify_data)


def separation_failures(keys):
    sets = {role: set(group.token_hash) for role, group in keys.groupby('role')}
    pairs = [('train','validation'), ('train','internal_test'), ('train','legacy'),
             ('validation','internal_test'), ('validation','legacy')]
    return [{'left': a, 'right': b, 'shared': len(sets[a] & sets[b])} for a, b in pairs
            if sets[a] & sets[b]]


def audit(model, registry):
    # No torch import, environment probing, checkpoint download or training.
    verify_data(include_final=True)
    plan = load_plan()
    entry, snapshot = artifact_entry(registry, model)
    if entry['repo_id'] != plan['models'][model]['repo_id']:
        raise RuntimeError('Model repo differs from prospective plan')
    config = json.loads((snapshot / 'config.json').read_text())
    expected_type = {'qwen':'qwen3','llama':'llama','bert':'bert','distilbert':'distilbert','roberta':'roberta'}[model]
    if config.get('model_type') != expected_type:
        raise RuntimeError('Snapshot model_type differs from requested model')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('Offset-aware input audit requires a fast tokenizer')
    tokenizer.truncation_side = 'right'
    is_llm = plan['models'][model]['family'] == 'llm'
    if is_llm and not tokenizer.eos_token:
        raise RuntimeError('Missing EOS token')
    frames = [('train', 'train_10000.csv', load_frame(DATA/'train_10000.csv')),
              ('validation', 'validation_2000.csv', load_frame(DATA/'validation_2000.csv')),
              ('internal_test', 'internal_test.csv', load_frame(DATA/'internal_test.csv'))]
    frames += [('legacy', p.name, load_frame(p)) for p in sorted((DATA/'legacy_subsets').glob('*.csv'))]
    rows = []
    for role, source, frame in frames:
        for pos, row in enumerate(frame.itertuples(index=False)):
            text = str(row.narrative)
            if is_llm:
                rendered, encoded = llm_encoding(tokenizer, text, 500, 512, offsets=True)
                start = rendered.index('Complaint:\n') + len('Complaint:\n')
                end = start + min(500, len(text))
                retained = max([max(0, min(b, end)-start) for a,b in encoded['offset_mapping'] if a < end and b > start] or [0])
                token_count = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
                if role in {'train','validation'}:
                    sft_record(tokenizer, text, row.product)
            else:
                encoded = tokenizer(text[:500], truncation=True, max_length=512, return_offsets_mapping=True)
                retained = max([b for a,b in encoded['offset_mapping']] or [0])
                token_count = len(tokenizer(text[:500])['input_ids'])
            rows.append({'role': role, 'source_file': source, 'source_row': pos,
                         'record_id': getattr(row, 'record_id', f'{source}:{pos}'),
                         'token_hash': sha_text(json.dumps(encoded['input_ids'])),
                         'tokens_before_cap': token_count, 'tokens_retained': len(encoded['input_ids']),
                         'requested_chars': min(500,len(text)), 'retained_narrative_chars': retained})
    keys = pd.DataFrame(rows)
    failures = separation_failures(keys)
    long_summary = []
    if is_llm:
        # Check long-input evaluation keys against actual 500-character training/validation keys too.
        train_keys = set(keys.loc[keys.role.eq('train'), 'token_hash'])
        val_keys = set(keys.loc[keys.role.eq('validation'), 'token_hash'])
        test = frames[2][2]
        for condition in plan['long_input']['conditions']:
            long_rows = []
            for row in test.itertuples(index=False):
                rendered, encoded = llm_encoding(tokenizer, row.narrative, condition['chars'], condition['tokens'], offsets=True)
                start = rendered.index('Complaint:\n') + len('Complaint:\n')
                chars = min(condition['chars'],len(row.narrative))
                retained = max([max(0,min(b,start+chars)-start) for a,b in encoded['offset_mapping']
                                if a < start+chars and b > start] or [0])
                key = sha_text(json.dumps(encoded['input_ids']))
                if key in train_keys or key in val_keys:
                    failures.append({'condition':condition['name'],'record_id':str(row.record_id),'shared_with':'train_or_validation'})
                long_rows.append({'record_id':row.record_id, 'token_hash':key, 'requested_chars':chars,
                                  'retained_narrative_chars':retained, 'tokens_retained':len(encoded['input_ids']),
                                  'tokens_before_cap':len(tokenizer(rendered,add_special_tokens=False)['input_ids'])})
            long_frame = pd.DataFrame(long_rows)
            long_frame.to_csv(PREP/f'input_keys_{model}_{condition["name"]}.csv.gz',index=False)
            long_summary.append({'condition':condition['name'], 'rows':len(long_frame),
                                 'clipped_rows':int((long_frame.tokens_before_cap > condition['tokens']).sum()),
                                 'max_tokens_before_cap':int(long_frame.tokens_before_cap.max())})
    path = PREP/f'input_keys_{model}.csv.gz'
    keys.to_csv(path,index=False)
    token_files = {str(p.relative_to(snapshot)):sha_file(p) for p in sorted(snapshot.rglob('*'))
                   if p.is_file() and p.suffix in {'.json','.txt','.jinja','.model'} and p.stat().st_size < 100_000_000}
    report = {'model':model, 'artifact':entry, 'protocol_sha256':PROTOCOL_SHA,
              'plan_sha256':sha_file(CONFIG), 'common_code_sha256':sha_file(Path(__file__).with_name('revision_gpu_common.py')),
              'audit_code_sha256':sha_file(Path(__file__)),
              'tokenizer_files_sha256':token_files, 'passed':not failures, 'failures':failures,
              'keys_sha256':sha_file(path), 'rows':len(keys), 'long_conditions':long_summary,
              'rows_over_512':int((keys.tokens_before_cap>512).sum()),
              'scope':'Actual fixed selections and preserved evaluation views. Token identities are label-blind. No semantic-duplication guarantee; long cap retention reported explicitly. SFT completion-mask/target length checked.'}
    (PREP/f'input_audit_{model}.json').write_text(json_text(report))
    if failures:
        raise RuntimeError('Actual tokenizer input overlap found. Do not train; document a protocol amendment first.')
    print(f'{model}: CPU tokenizer audit passed; no GPU model loaded')
    return report
