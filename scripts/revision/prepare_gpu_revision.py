#!/usr/bin/env python3
"""Freeze prospective GPU settings/commands, CPU only; original protocol is untouched."""
from __future__ import annotations
from project_paths import project_path
import itertools
import json
import shlex
import sys
from revision_gpu_common import (CONFIG, DATA, EXP, LABELS, PAPER, PREP, PROTOCOL_SHA,
                                 ROOT, load_frame, sha_file, sha_text, verify_data, write_new)


def main():
    verify_data(include_final=True)
    candidates = [dict(candidate_id=i, learning_rate=lr, epochs=ep)
                  for i, (lr, ep) in enumerate(itertools.product([2e-5, 3e-5, 5e-5], [2, 3]), 1)]
    plan = {
        'version': 'gpu_revision_v1', 'prepared_date': '2026-10-20',
        'status': 'prepared_cpu_only_not_gpu_validated', 'parent_protocol_sha256': PROTOCOL_SHA,
        'train': 'train_10000.csv', 'validation': 'validation_2000.csv',
        'selection_seed': 42, 'optimization_seeds': [42, 43, 44],
        'membership': 'Use all frozen rows in original file order; no subsampling. Training shuffle may vary with optimisation seed.',
        'models': {'qwen': {'repo_id': 'Qwen/Qwen3-8B', 'family': 'llm'},
                   'llama': {'repo_id': 'meta-llama/Llama-3.1-8B-Instruct', 'family': 'llm'},
                   'bert': {'repo_id': 'google-bert/bert-base-uncased', 'family': 'encoder'},
                   'distilbert': {'repo_id': 'distilbert/distilbert-base-uncased', 'family': 'encoder'},
                   'roberta': {'repo_id': 'FacebookAI/roberta-base', 'family': 'encoder'}},
        'artifact_policy': 'Template is unresolved. Register actual locally available immutable snapshots; no auto-download. Exact model variant must be verified before execution; new pins do not recover historical identity.',
        'qlora': {'epochs': 2, 'learning_rate': 2e-4, 'batch_size': 1, 'eval_batch_size': 1,
                  'gradient_accumulation_steps': 16, 'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': .05,
                  'max_narrative_chars': 500, 'max_length': 512, 'target_modules': 'all-linear',
                  'completion_only_loss': True, 'quantization': 'NF4/double/FP16',
                  'optimizer': 'paged_adamw_8bit', 'warmup_ratio': .03, 'lr_scheduler': 'cosine',
                  'eval_steps': 100, 'selection': 'Best validation loss checkpoint within 2 epochs; fixed historical configuration, no new test-driven search.'},
        'encoder': {'grid': candidates, 'search_seed': 42, 'max_narrative_chars': 500,
                    'max_length': 512, 'batch_size': 4, 'gradient_accumulation_steps': 4,
                    'eval_batch_size': 16, 'weight_decay': .01, 'warmup_ratio': .06,
                    'optimizer': 'adamw_torch', 'scheduler': 'linear',
                    'selection': 'Final epoch validation macro-F1; ties within 1e-12 use smallest candidate_id. No test files read by search. Freeze winner before final refits.',
                    'refits': 'Three actual fixed-data optimisation seeds; retain final epoch for the selected epoch budget.'},
        'generation': {'max_new_tokens': 32, 'do_sample': False, 'enable_thinking': False,
                       'batch_size': 1, 'attention_implementation': 'sdpa', 'precision': 'NF4/double/FP16'},
        'serialization': 'Render the chat template once; tokenize with add_special_tokens=False. Pretokenized SFT input_ids/completion_mask use that same prefix, and full targets must fit 512 tokens. This explicit no-duplicate-BOS policy is for new runs, not an asserted historical reconstruction.',
        'long_input': {'conditions': [{'name': 'chars500_tokens512', 'chars': 500, 'tokens': 512},
                                     {'name': 'chars500_tokens2048', 'chars': 500, 'tokens': 2048},
                                     {'name': 'chars1500_tokens2048', 'chars': 1500, 'tokens': 2048}],
                       'evaluation': 'Same complete internal_test IDs; no retraining. 500/2048 is the token-cap control. Main comparison 500/2048 versus 1500/2048; separately report full narratives >500 chars. No legacy long text reconstruction.',
                       'uncertainty': 'Observed macro-F1/weighted-F1/accuracy; 10000 paired label-blind group bootstrap with common multiplicities across conditions and repeats; seed/run SD separately. Reuse frozen internal groups.',
                       'length_audit': 'Record requested/retained narrative characters, pre-cap token count, retained tokens and prompt hash. No silent claim that all requested text was visible.'},
        'latency': {'sample_size': 100, 'per_class': 20, 'warmup_rows': 10, 'repetitions': 3,
                    'selection': 'Per class choose smallest sha256(latency_v1|record_id); never use predictions. Same IDs for all variants and all three length conditions.',
                    'order': 'Per repetition use sha256(order_v1|repetition|record_id). Three repetitions within each fixed variant job; archive start/end GPU state. Separate model loads/jobs leave residual thermal/load drift; do not claim full randomised crossover control.',
                    'metrics': ['generation_seconds', 'end_to_end_seconds', 'input_tokens', 'new_tokens'],
                    'timing': 'Wall clock with CUDA synchronization before/after generate; end-to-end adds tokenization/H2D/decode/regex. Model/tokenizer loading, warmup, disk I/O excluded. Batch=1, no concurrent benchmark; unmerged adapter, same base precision.',
                    'summary': 'Per variant/condition: mean, median, p95, requests/s from measured end-to-end time, output lengths; paired record-level differences using per-record repetition means. This is serial batch-1 throughput, not service capacity.'},
        'failure_policy': 'Abort on hash/membership/audit mismatch. Do not silently change data, token caps, batch, search space or precision after failure; record a dated protocol amendment first.',
        'hardware': 'Unresolved until GPU available. Record GPU/driver/CUDA/torch and pip freeze; choose a compatible GPU environment before executing. CPU checks do not certify CUDA compatibility.'}
    write_new(CONFIG, plan)
    write_new(CONFIG.with_suffix('.sha256'), sha_file(CONFIG) + '  gpu_revision_v1.json\n')
    registry = {key: {'repo_id': value['repo_id'], 'commit': None, 'snapshot': None}
                for key, value in plan['models'].items()}
    write_new(project_path(PAPER, 'configs/model_registry.template.json'), registry)
    test = load_frame(DATA / 'internal_test.csv')
    selected = []
    for label in LABELS:
        group = test.loc[test['product'].eq(label)].copy()
        group['selection_key'] = group.record_id.map(lambda rid: sha_text(f'latency_v1|{rid}'))
        selected.append(group.sort_values('selection_key').head(20))
    import pandas as pd
    sample = pd.concat(selected).sort_values('selection_key')
    cols = ['record_id', 'product', 'narrative_hash', 'selection_key']
    write_new(PREP / 'latency_sample_manifest.csv', sample[cols].to_csv(index=False))
    warmup = load_frame(DATA / 'validation_2000.csv').head(10)
    write_new(PREP / 'latency_warmup_manifest.csv', warmup[['record_id','product','narrative_hash']].to_csv(index=False))
    commands = []
    prefix = ['python', 'scripts/revision/run_revision_gpu.py']
    for model in plan['models']:
        commands.append(prefix + ['audit', '--model', model, '--registry', 'configs/model_registry.local.json'])
        if plan['models'][model]['family'] == 'llm':
            for seed in [42, 43, 44]:
                commands.append(prefix + ['qlora', '--model', model, '--seed', str(seed)])
        else:
            commands.append(prefix + ['encoder-search', '--model', model])
            for seed in [42, 43, 44]:
                commands.append(prefix + ['encoder-refit', '--model', model, '--seed', str(seed)])
    # These are dry-run commands: each real GPU call requires an explicit --execute-gpu.
    for model in ['qwen', 'llama', 'bert', 'distilbert', 'roberta']:
        variants = ['base', '42', '43', '44'] if model in ['qwen', 'llama'] else ['42', '43', '44']
        for variant in variants:
            commands.append(prefix + ['evaluate', '--model', model, '--variant', variant])
            if model in ['qwen', 'llama']:
                for condition in plan['long_input']['conditions']:
                    for action in ['long-input', 'latency']:
                        commands.append(prefix + [action, '--model', model, '--variant', variant,
                                                   '--condition', condition['name']])
    write_new(PREP / 'commands_dry_run.txt', '\n'.join(shlex.join(c) for c in commands) + '\n')
    write_new(PREP / 'preparation_manifest.json', {
        'plan_sha256': sha_file(CONFIG), 'parent_protocol_sha256': PROTOCOL_SHA,
        'latency_sample_sha256': sha_file(PREP / 'latency_sample_manifest.csv'),
        'latency_warmup_sha256': sha_file(PREP / 'latency_warmup_manifest.csv'),
        'commands_sha256': sha_file(PREP / 'commands_dry_run.txt'),
        'gpu_started': False, 'encoder_candidates_per_model': 6,
        'encoder_searches_executed': False, 'models_trained': False})
    print('Frozen prospective plan, 100-record balanced latency cohort, warmup IDs and dry-run commands. No GPU access.')


if __name__ == '__main__': main()
