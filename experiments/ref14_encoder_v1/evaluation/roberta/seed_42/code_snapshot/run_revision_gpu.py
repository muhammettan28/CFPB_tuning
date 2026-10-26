#!/usr/bin/env python3
"""Fixed-data GPU runner. Default is a CPU-only dry run; CUDA requires --execute-gpu."""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from prediction_cache import PredictionCache
from revision_gpu_common import (CONFIG, DATA, EXP, LABELS, PAPER, PREP, PROTOCOL_SHA,
    ROOT, artifact_entry, audit_guard, json_text, llm_encoding, load_frame, load_plan,
    metrics, parse_label, pick_candidate, selection_ids, sft_record, sha_file, sha_text,
    snapshot_hashes, verify_data, versions, write_new)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['audit','smoke','qlora','encoder-search','encoder-refit','evaluate','long-input','latency'])
    p.add_argument('--model', required=True, choices=['qwen','llama','bert','distilbert','roberta'])
    p.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    p.add_argument('--variant', choices=['base','42','43','44'], default='base')
    p.add_argument('--condition', default='chars500_tokens512')
    p.add_argument('--registry', type=Path, default=PAPER/'configs/model_registry.local.json')
    p.add_argument('--execute-gpu', action='store_true', help='Only use after the user reports that the GPU is available')
    p.add_argument('--resume-from-checkpoint', type=Path, default=None)
    return p.parse_args(argv)


def describe_job(args, plan):
    family = plan['models'][args.model]['family']
    if args.action == 'qlora' and family != 'llm':
        raise ValueError('QLoRA requires an LLM')
    if args.action.startswith('encoder') and family != 'encoder':
        raise ValueError('Encoder action requires an encoder')
    if args.action in {'long-input','latency'} and family != 'llm':
        raise ValueError('Prospective long-input/latency protocol is for the two LLMs')
    if family == 'encoder' and args.action == 'evaluate' and args.variant == 'base':
        raise ValueError('An unfitted encoder classification head is not a baseline')
    if args.condition not in {x['name'] for x in plan['long_input']['conditions']}:
        raise ValueError('Condition not in frozen plan')
    return {'action':args.action, 'model':args.model, 'family':family, 'seed':args.seed,
            'variant':args.variant, 'condition':args.condition, 'train_rows':10000,
            'validation_rows':2000, 'resampling':False, 'plan_sha256':sha_file(CONFIG),
            'execution_requested':args.execute_gpu, 'gpu_started':False,
            'unresolved_prerequisites':['Actual immutable registry snapshots', 'CPU input audit for that snapshot',
                                        'Compatible GPU environment', 'Explicit GPU availability from user']}


def new_output(path):
    if path.exists():
        raise RuntimeError(f'Output already exists; refusing to overwrite: {path}')
    path.mkdir(parents=True)
    return path


def metadata(out, args, artifact, audit_hash, snapshot, torch):
    train, val = load_frame(DATA/'train_10000.csv'), load_frame(DATA/'validation_2000.csv')
    if args.action=='smoke': train,val=train.head(32),val.head(8)
    for frame, name in [(train,'train'),(val,'validation')]:
        frame[['record_id','product','narrative_hash']].to_csv(out/f'{name}_selection_manifest.csv',index=False)
    try:
        driver = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        driver = 'unavailable'
    info = {'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'protocol_sha256':PROTOCOL_SHA,'plan_sha256':sha_file(CONFIG), 'artifact':artifact,
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'actual_train_rows':len(train),'actual_validation_rows':len(val),
            'smoke_only':args.action=='smoke',
            'optimization_seed':args.seed if args.action in {'qlora','encoder-search','encoder-refit','smoke'} else int(args.variant) if args.variant!='base' else None,
            'input_audit_sha256':audit_hash, 'python':platform.python_version(),'versions':versions(),
            'cuda_runtime':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),'driver':driver,
            'train_csv_sha256':sha_file(DATA/'train_10000.csv'),
            'validation_csv_sha256':sha_file(DATA/'validation_2000.csv'),
            'train_order_sha256':sha_text(json.dumps(selection_ids(train))),
            'validation_order_sha256':sha_text(json.dumps(selection_ids(val))),
            'code_sha256':{p.name:sha_file(p) for p in [Path(__file__),Path(__file__).with_name('revision_gpu_common.py'),Path(__file__).with_name('audit_gpu_tokenizers.py')]}}
    (out/'run_metadata.json').write_text(json_text(info))
    code_dir=out/'code_snapshot';code_dir.mkdir(exist_ok=True)
    for name in ['run_revision_gpu.py','revision_gpu_common.py','audit_gpu_tokenizers.py','prediction_cache.py','requirements-gpu-revision.txt']:
        source=Path(__file__).with_name(name)
        (code_dir/name).write_bytes(source.read_bytes())
    (out/'base_snapshot_files.sha256.json').write_text(json_text(snapshot_hashes(snapshot)))
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'pip_freeze.txt').write_text(freeze)
    return info


def tokenizer_at(snapshot):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = tok.truncation_side = 'right'
    return tok


def llm_at(snapshot, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                quantization_config=quant,dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,adapter,local_files_only=True)
    return model


def qlora(args, plan, snapshot, out, smoke=False):
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(args.seed)
    tok = tokenizer_at(snapshot)
    def dataset(name):
        frame = load_frame(DATA/name)
        if smoke: frame=frame.head(32 if name.startswith('train') else 8)
        return Dataset.from_list([sft_record(tok,r.narrative,r.product) for r in frame.itertuples(index=False)])
    train, val = dataset('train_10000.csv'),dataset('validation_2000.csv')
    model = llm_at(snapshot)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant':False})
    cfg = plan['qlora']
    config = SFTConfig(output_dir=str(out/'checkpoints'),num_train_epochs=cfg['epochs'],
        learning_rate=cfg['learning_rate'],per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},fp16=True,bf16=False,
        max_length=512,completion_only_loss=True,packing=False,dataset_kwargs={'skip_prepare_dataset':True},
        max_steps=2 if smoke else -1,
        eval_strategy='steps',eval_steps=1 if smoke else 100,save_strategy='steps',save_steps=1 if smoke else 100,save_total_limit=2,
        load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        logging_steps=10,report_to='none',optim='paged_adamw_8bit',warmup_ratio=.03,
        lr_scheduler_type='cosine',seed=args.seed,data_seed=args.seed)
    lora = LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
                      target_modules='all-linear',bias='none',task_type='CAUSAL_LM')
    trainer = SFTTrainer(model=model,args=config,train_dataset=train,eval_dataset=val,
                         processing_class=tok,peft_config=lora)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if getattr(args,'resume_from_checkpoint',None) else None)
    trainer.save_model(str(out/'best_adapter'))
    tok.save_pretrained(out/'best_adapter')
    trainer.save_state()
    result = trainer.evaluate()
    (out/'validation_metrics.json').write_text(json_text(result))
    return {'best_checkpoint':trainer.state.best_model_checkpoint,'validation_metrics':result}


def encoder_dataset(frame, tok, max_length=512):
    from datasets import Dataset
    encoded = tok(frame.narrative.str[:500].tolist(),truncation=True,max_length=max_length)
    encoded['labels'] = [LABELS.index(x) for x in frame['product']]
    return Dataset.from_dict(encoded)


def fit_encoder(snapshot, out, candidate, seed, plan, smoke=False, settings=None):
    from transformers import (AutoModelForSequenceClassification,DataCollatorWithPadding,
                              Trainer,TrainingArguments,set_seed)
    set_seed(seed)  # Before the randomly initialised five-label head is created.
    tok = tokenizer_at(snapshot)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot,local_files_only=True,
                num_labels=len(LABELS),id2label=dict(enumerate(LABELS)),label2id={v:k for k,v in enumerate(LABELS)})
    train_frame=load_frame(DATA/'train_10000.csv')
    val_frame = load_frame(DATA/'validation_2000.csv')
    if smoke: train_frame,val_frame=train_frame.head(32),val_frame.head(8)
    cfg = {**plan['encoder'],**(settings or {})}
    train = encoder_dataset(train_frame,tok,cfg['max_length'])
    val = encoder_dataset(val_frame,tok,cfg['max_length'])
    training = TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=candidate['epochs'],
        learning_rate=candidate['learning_rate'],per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],per_device_eval_batch_size=cfg['eval_batch_size'],
        weight_decay=cfg['weight_decay'],warmup_ratio=cfg['warmup_ratio'],optim='adamw_torch',lr_scheduler_type='linear',
        max_steps=2 if smoke else -1,
        eval_strategy='no',save_strategy='no',logging_steps=1 if smoke else 50,report_to='none',fp16=True,bf16=False,seed=seed,data_seed=seed)
    trainer = Trainer(model=model,args=training,train_dataset=train,processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok,pad_to_multiple_of=8))
    trainer.train()
    logits = trainer.predict(val).predictions
    if isinstance(logits,tuple): logits = logits[0]
    pred = [LABELS[int(i)] for i in logits.argmax(axis=-1)]
    result = metrics(val_frame['product'].tolist(),pred)
    val_frame[['record_id','product']].assign(prediction=pred).to_csv(out/'validation_predictions.csv',index=False)
    trainer.save_model(str(out/'model'))
    tok.save_pretrained(out/'model')
    del trainer,model
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return result


def selected_encoder(model, plan):
    path = EXP/'encoder'/model/'search'/'selected_config.json'
    if sha_file(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('Selected encoder config hash mismatch')
    selected = json.loads(path.read_text())
    search_path = path.parent/'validation_search.json'
    if sha_file(search_path) != selected['validation_search_sha256']:
        raise RuntimeError('Encoder validation search hash mismatch')
    rows = json.loads(search_path.read_text())
    if len(rows) != len(plan['encoder']['grid']) or {r['candidate_id'] for r in rows} != {r['candidate_id'] for r in plan['encoder']['grid']}:
        raise RuntimeError('Encoder validation search incomplete')
    if pick_candidate(rows)['candidate_id'] != selected['candidate']['candidate_id'] or selected['plan_sha256'] != sha_file(CONFIG):
        raise RuntimeError('Encoder validation selection mismatch')
    expected = next(c for c in plan['encoder']['grid'] if c['candidate_id']==selected['candidate']['candidate_id'])
    if selected['candidate'] != expected:
        raise RuntimeError('Selected encoder configuration differs from frozen grid')
    for row in rows:
        grid = next(c for c in plan['encoder']['grid'] if c['candidate_id']==row['candidate_id'])
        if any(row[k] != grid[k] for k in grid):
            raise RuntimeError('Candidate grid changed')
    return selected,path


def search_encoder(args, plan, snapshot, out, artifact):
    rows=[]
    for candidate in plan['encoder']['grid']:
        dest=new_output(out/f'candidate_{candidate["candidate_id"]:02d}')
        result=fit_encoder(snapshot,dest,candidate,plan['encoder']['search_seed'],plan)
        rows.append({**candidate,'validation_macro_f1':result['macro_f1'],'validation_metrics':result})
        (out/'validation_search.json').write_text(json_text(rows))
    winner=pick_candidate(rows)
    candidate=next(c for c in plan['encoder']['grid'] if c['candidate_id']==winner['candidate_id'])
    selected={'candidate':candidate,'validation_macro_f1':winner['validation_macro_f1'],
              'selection_seed':42,'plan_sha256':sha_file(CONFIG),'artifact':artifact,
              'validation_search_sha256':sha_file(out/'validation_search.json'),
              'test_read_during_selection':False}
    write_new(out/'selected_config.json',selected)
    write_new(out/'selected_config.sha256',sha_file(out/'selected_config.json')+'  selected_config.json\n')
    return selected


def trained_source(args, plan, artifact):
    if args.variant=='base': return None,None
    family=plan['models'][args.model]['family']
    dest=EXP/('qlora' if family=='llm' else 'encoder')/args.model/f'seed_{args.variant}'
    done=json.loads((dest/'COMPLETE.json').read_text())
    meta=json.loads((dest/'run_metadata.json').read_text())
    if meta['artifact']!=artifact or meta['protocol_sha256']!=PROTOCOL_SHA or meta['plan_sha256']!=sha_file(CONFIG):
        raise RuntimeError('Trained model provenance mismatch')
    weights=dest/('best_adapter' if family=='llm' else 'model')
    for name,digest in done['saved_model_files_sha256'].items():
        if sha_file(weights/name)!=digest: raise RuntimeError('Saved fitted model changed')
    return weights,dest


def inference_model(args, plan, snapshot, artifact):
    weights,source=trained_source(args,plan,artifact)
    if source:
        recorded=json.loads((source/'base_snapshot_files.sha256.json').read_text())
        if snapshot_hashes(snapshot)!=recorded: raise RuntimeError('Base snapshot changed since training')
    tok=tokenizer_at(snapshot)
    if plan['models'][args.model]['family']=='llm':
        model=llm_at(snapshot,weights)
        model.config.use_cache=True
    else:
        from transformers import AutoModelForSequenceClassification
        model=AutoModelForSequenceClassification.from_pretrained(weights,local_files_only=True).to('cuda')
    model.eval()
    return model,tok


def generate_one(model,tok,text,condition,cache=None):
    import torch
    rendered,encoded=llm_encoding(tok,text,condition['chars'],condition['tokens'],offsets=True)
    offsets=encoded.pop('offset_mapping')
    start=rendered.index('Complaint:\n')+len('Complaint:\n')
    requested=min(len(str(text)),condition['chars'])
    retained=max([max(0,min(b,start+requested)-start) for a,b in offsets
                  if a<start+requested and b>start] or [0])
    key=PredictionCache.key(encoded)
    result=cache.get(key) if cache else None
    if result is None:
        inputs={k:torch.tensor([v],device='cuda') for k,v in encoded.items()}
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=32,do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id,use_cache=True)
        tail=ids[0,inputs['input_ids'].shape[1]:]
        response=tok.decode(tail,skip_special_tokens=True).strip()
        result={'prediction':parse_label(response),'raw_response':response,'new_tokens':len(tail)}
        if cache:cache.put(key,result)
    return {**result,'input_tokens':len(encoded['input_ids']),'prompt_sha256':sha_text(rendered),
            'input_tokens_sha256':sha_text(json.dumps(encoded['input_ids'])),
            'tokens_before_cap':len(tok(rendered,add_special_tokens=False)['input_ids']),
            'requested_narrative_chars':requested,'retained_narrative_chars':retained}


def encoder_inputs(tok,texts,max_length):
    """Truncate before collating, exactly as encoder_dataset + Trainer collator.

    A cap such as 100 need not be a multiple of the padding alignment (8).
    Keep at most 100 real tokens, then pad a batch to 104 if necessary.
    """
    encoded=tok(texts,truncation=True,max_length=max_length,padding=False)
    return tok.pad(encoded,padding=True,pad_to_multiple_of=8,return_tensors='pt')


def predict_encoder(model,tok,frame,batch_size,max_length=512):
    """Match Trainer's validation batches, padding and FP16 autocast."""
    import torch
    labels=[];lengths=[]
    for start in range(0,len(frame),batch_size):
        texts=frame.narrative.iloc[start:start+batch_size].str[:500].tolist()
        inputs=encoder_inputs(tok,texts,max_length).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            ids=model(**inputs).logits.float().argmax(dim=-1).cpu().tolist()
        labels.extend(LABELS[int(i)] for i in ids)
        lengths.extend(inputs['attention_mask'].sum(dim=1).cpu().tolist())
    return labels,lengths


def require_encoder_validation_parity(args,plan,model,tok,out,reference_path=None,max_length=512):
    """Fail before final-test reads if reloaded inference differs from validation."""
    validation=load_frame(DATA/'validation_2000.csv')
    source=reference_path or EXP/'encoder'/args.model/f'seed_{args.variant}'/'validation_predictions.csv'
    saved=pd.read_csv(source,dtype={'record_id':str})
    if saved.record_id.tolist()!=validation.record_id.astype(str).tolist() or saved['product'].tolist()!=validation['product'].tolist():
        raise RuntimeError('Encoder validation inference identities differ')
    predicted,_=predict_encoder(model,tok,validation,plan['encoder']['eval_batch_size'],max_length)
    differences=sum(a!=b for a,b in zip(predicted,saved.prediction.tolist()))
    report={'passed':differences==0,'validation_rows':len(validation),'prediction_differences':differences,
            'reference_path':str(source.relative_to(ROOT)),'reference_sha256':sha_file(source),
            'precision':'FP32 saved parameters with CUDA FP16 autocast, as in Trainer validation',
            'eval_batch_size':plan['encoder']['eval_batch_size'],'pad_to_multiple_of':8,'max_length':max_length,
            'final_test_read_before_check':False}
    (out/'validation_inference_parity.json').write_text(json_text(report))
    if differences:raise RuntimeError('Reloaded encoder predictions differ from Trainer validation; final-test inference blocked')


def evaluate(args,plan,snapshot,out,artifact):
    model,tok=inference_model(args,plan,snapshot,artifact)
    is_llm=plan['models'][args.model]['family']=='llm'
    if not is_llm:require_encoder_validation_parity(args,plan,model,tok,out)
    condition=next(c for c in plan['long_input']['conditions'] if c['name']==args.condition)
    paths=[DATA/'internal_test.csv']
    if args.action=='evaluate':
        condition=plan['long_input']['conditions'][0]
        paths+=sorted((DATA/'legacy_subsets').glob('*.csv'))
    summary={}
    import torch
    cache=None
    if is_llm:
        weights,source=trained_source(args,plan,artifact)
        identity={'artifact':artifact,'variant':args.variant,
            'fitted_model_files':json.loads((source/'COMPLETE.json').read_text())['saved_model_files_sha256'] if source else None,
            'generation_config':{k:v for k,v in model.generation_config.to_dict().items() if not k.startswith('_')},
            'generation_overrides':{'max_new_tokens':32,'do_sample':False,'num_beams':1,'use_cache':True,
                                    'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id},
            'plan_sha256':sha_file(CONFIG),'runner_sha256':sha_file(Path(__file__)),
            'cache_code_sha256':sha_file(Path(__file__).with_name('prediction_cache.py')),
            'gpu_driver':json.loads((out/'run_metadata.json').read_text())['driver'],
            'versions':versions(),'gpu':torch.cuda.get_device_name(0)}
        cache=PredictionCache(EXP/'prediction_cache'/args.model/f'{args.variant}.sqlite3',identity)
        (out/'effective_generation.json').write_text(json_text(identity))
    for path in paths:
        frame=load_frame(path)
        rows=[]
        encoder_predictions=[];encoder_lengths=[]
        if not is_llm:
            encoder_predictions,encoder_lengths=predict_encoder(model,tok,frame,plan['encoder']['eval_batch_size'])
        for i,row in enumerate(frame.itertuples(index=False)):
            if is_llm:
                result=generate_one(model,tok,row.narrative,condition,cache=cache)
            else:
                result={'prediction':encoder_predictions[i],'input_tokens':encoder_lengths[i]}
            rows.append({'record_id':getattr(row,'record_id',f'{path.name}:{i}'),
                         'source_file':path.name,'source_row':i,'true_label':row.product,**result})
            if is_llm and (i+1)%500==0:
                cache.flush()
                print(f'{args.model}/{args.variant}/{path.stem}: {i+1}/{len(frame)} rows, cache hits={cache.hits}, fresh generations={cache.misses}',flush=True)
        predictions=pd.DataFrame(rows)
        predictions.to_csv(out/f'{path.stem}_predictions.csv.gz',index=False)
        summary[path.stem]=metrics(predictions.true_label.tolist(),predictions.prediction.tolist())
    if cache:
        (out/'prediction_cache_stats.json').write_text(json_text({'fingerprint':cache.fingerprint,'hits':cache.hits,'fresh_generations':cache.misses,
            'policy':'Exact token IDs/masks with fixed checkpoint/settings only. Every evaluation row retained. Latency bypasses cache.'}))
        cache.close()
    (out/'metrics.json').write_text(json_text(summary))
    return summary


def gpu_state():
    try:
        return subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,uuid,name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): return 'unavailable'


def require_no_other_compute(out):
    """Keep desktop graphics; reject concurrent CUDA compute during latency."""
    text=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids={int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}
    other=sorted(pids-{os.getpid()})
    if other:
        (out/'concurrent_compute_detected.json').write_text(json_text({'utc':datetime.now(timezone.utc).isoformat(),'other_pids':other}))
        raise RuntimeError('Concurrent GPU compute detected; controlled latency stage stopped without terminating other processes')


def latency(args,plan,snapshot,out,artifact):
    import torch
    require_no_other_compute(out)
    prep=json.loads((PREP/'preparation_manifest.json').read_text())
    for file,key in [('latency_sample_manifest.csv','latency_sample_sha256'),('latency_warmup_manifest.csv','latency_warmup_sha256')]:
        if sha_file(PREP/file)!=prep[key]: raise RuntimeError('Frozen latency cohort changed')
    model,tok=inference_model(args,plan,snapshot,artifact)
    condition=next(c for c in plan['long_input']['conditions'] if c['name']==args.condition)
    sample=pd.read_csv(PREP/'latency_sample_manifest.csv')
    test=load_frame(DATA/'internal_test.csv').set_index('record_id',drop=False)
    warmup=load_frame(DATA/'validation_2000.csv').set_index('record_id',drop=False)
    warmup_ids=pd.read_csv(PREP/'latency_warmup_manifest.csv').record_id
    (out/'gpu_state_before.csv').write_text(gpu_state())
    for rid in warmup_ids: generate_one(model,tok,warmup.loc[rid,'narrative'],condition)
    torch.cuda.synchronize()
    rows=[]
    for repetition in range(plan['latency']['repetitions']):
        require_no_other_compute(out)
        ids=sorted(sample.record_id.tolist(),key=lambda rid:sha_text(f'order_v1|{repetition}|{rid}'))
        for order,rid in enumerate(ids):
            text=test.loc[rid,'narrative']
            torch.cuda.synchronize(); overall=time.perf_counter()
            rendered,encoded=llm_encoding(tok,text,condition['chars'],condition['tokens'])
            inputs={k:torch.tensor([v],device='cuda') for k,v in encoded.items()}
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=32,do_sample=False,num_beams=1,
                    pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
            torch.cuda.synchronize(); end=time.perf_counter()
            tail=generated[0,inputs['input_ids'].shape[1]:]
            response=tok.decode(tail,skip_special_tokens=True).strip()
            predicted=parse_label(response)
            torch.cuda.synchronize(); overall_end=time.perf_counter()
            rows.append({'record_id':rid,'repetition':repetition,'order':order,
                'generation_seconds':end-start,'end_to_end_seconds':overall_end-overall,
                'input_tokens':len(encoded['input_ids']),'new_tokens':len(tail),'prediction':predicted,
                'prompt_sha256':sha_text(rendered)})
            if (order+1)%25==0:require_no_other_compute(out)
    frame=pd.DataFrame(rows);frame.to_csv(out/'latency_records.csv',index=False)
    (out/'gpu_state_after.csv').write_text(gpu_state())
    result={key:{'mean':float(frame[key].mean()),'median':float(frame[key].median()),
                 'p95':float(frame[key].quantile(.95))} for key in ['generation_seconds','end_to_end_seconds']}
    result['serial_requests_per_second']=len(frame)/float(frame.end_to_end_seconds.sum())
    result['rows']=len(frame)
    (out/'latency_summary.json').write_text(json_text(result))
    return result


def main(argv=None):
    args=arguments(argv);plan=load_plan();description=describe_job(args,plan)
    if args.action=='audit':
        if not args.registry.exists():
            print(json_text({**description,'blocked':'Registry is unresolved; no GPU access or tokenizer audit performed'}));return
        from audit_gpu_tokenizers import audit
        audit(args.model,args.registry);return
    if not args.execute_gpu:
        print(json_text(description));return
    verify_data(include_final=args.action in {'evaluate','long-input','latency'})
    artifact,snapshot,audit_hash=audit_guard(args.registry,args.model)
    if artifact['repo_id']!=plan['models'][args.model]['repo_id']:
        raise RuntimeError('Wrong model repo')
    # Reject unsupported library API changes before touching CUDA.
    pinned={'transformers':'4.57.6','trl':'0.23.1','peft':'0.17.1'}
    installed=versions()
    for name,version in pinned.items():
        if args.action=='qlora' or (args.action=='smoke' and description['family']=='llm') or name=='transformers':
            if installed[name]!=version: raise RuntimeError(f'{name} must be {version}; record explicit environment amendments otherwise')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback is intentionally disabled')
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one available GPU using CUDA_VISIBLE_DEVICES; do not change the effective batch silently')
    family=plan['models'][args.model]['family']
    selected=None
    if args.action=='smoke': dest=PAPER/'analysis/gpu_execution/smoke'/args.model
    elif args.action=='qlora': dest=EXP/'qlora'/args.model/f'seed_{args.seed}'
    elif args.action=='encoder-search': dest=EXP/'encoder'/args.model/'search'
    elif args.action=='encoder-refit':
        selected,selection_path=selected_encoder(args.model,plan)
        if selected['artifact']!=artifact: raise RuntimeError('Encoder search artifact changed')
        dest=EXP/'encoder'/args.model/f'seed_{args.seed}'
    elif args.action=='latency': dest=EXP/'latency'/args.model/args.variant/args.condition
    else: dest=EXP/'evaluation'/args.action/args.model/args.variant/(args.condition if args.action=='long-input' else 'suite')
    if args.resume_from_checkpoint:
        if args.action!='qlora' or not dest.is_dir() or (dest/'COMPLETE.json').exists():
            raise RuntimeError('Resume requires an incomplete existing QLoRA run')
        old=json.loads((dest/'run_metadata.json').read_text())
        if old['artifact']!=artifact or old['plan_sha256']!=sha_file(CONFIG) or old['protocol_sha256']!=PROTOCOL_SHA:
            raise RuntimeError('Resume provenance mismatch')
        if not args.resume_from_checkpoint.resolve().is_relative_to((dest/'checkpoints').resolve()):
            raise RuntimeError('Checkpoint is outside this run')
        out=dest
        with (out/'resume_events.jsonl').open('a') as f:
            f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(args.resume_from_checkpoint),'runner_sha256':sha_file(Path(__file__))})+'\n')
    else:
        out=new_output(dest)
        metadata(out,args,artifact,audit_hash,snapshot,torch)
    if args.action=='smoke':
        torch.cuda.reset_peak_memory_stats()
        result=qlora(args,plan,snapshot,out,smoke=True) if family=='llm' else fit_encoder(snapshot,out,plan['encoder']['grid'][0],args.seed,plan,smoke=True)
        result={'scope':'Hardware/API smoke only; excluded from scientific results.','details':result,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
        if family=='llm':
            gc.collect();torch.cuda.empty_cache()
            smoke_model=llm_at(snapshot,out/'best_adapter');smoke_model.eval()
            smoke_tok=tokenizer_at(snapshot)
            records=[]
            for row in load_frame(DATA/'validation_2000.csv').head(2).itertuples(index=False):
                records.append({'record_id':row.record_id,**generate_one(smoke_model,smoke_tok,row.narrative,plan['long_input']['conditions'][0])})
            (out/'smoke_generation.json').write_text(json_text(records))
            result['smoke_generation_rows']=len(records)
    elif args.action=='qlora': result=qlora(args,plan,snapshot,out)
    elif args.action=='encoder-search': result=search_encoder(args,plan,snapshot,out,artifact)
    elif args.action=='encoder-refit':
        write_new(out/'selected_config.json',selected)
        write_new(out/'selection_source.sha256',sha_file(selection_path)+'  selected_config.json\n')
        result=fit_encoder(snapshot,out,selected['candidate'],args.seed,plan)
        (out/'validation_metrics.json').write_text(json_text(result))
    elif args.action=='latency': result=latency(args,plan,snapshot,out,artifact)
    else: result=evaluate(args,plan,snapshot,out,artifact)
    done={'completed':True,'action':args.action,'result':result}
    if args.action in {'qlora','encoder-refit'}:
        weights=out/('best_adapter' if family=='llm' else 'model')
        done['saved_model_files_sha256']=snapshot_hashes(weights)
    (out/'COMPLETE.json').write_text(json_text(done))
    print(f'Completed {args.action}: {out.relative_to(ROOT)}')


if __name__=='__main__': main()
