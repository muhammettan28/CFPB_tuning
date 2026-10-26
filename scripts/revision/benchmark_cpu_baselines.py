#!/usr/bin/env python3
"""Time fixed baseline implementations on CPU; never changes archived fits or CUDA jobs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
import argparse, gc, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision_gpu_common import ROOT, DATA, PREP, EXP, LABELS, load_frame, sha_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['sparse', 'encoders'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists(): parser.error('Output must be new; recorded evidence is never overwritten')
    out.mkdir(parents=True)
    test = load_frame(DATA/'internal_test.csv')
    valid = load_frame(DATA/'validation_2000.csv')
    for frame in [test, valid]:
        frame['record_id'] = frame.record_id.astype(str)
    test = test.set_index('record_id', drop=False)
    valid = valid.set_index('record_id', drop=False)
    ids = pd.read_csv(PREP/'latency_sample_manifest.csv').record_id.astype(str).tolist()
    warm = pd.read_csv(PREP/'latency_warmup_manifest.csv').record_id.astype(str).tolist()
    assert len(ids) == 100 and len(warm) == 10
    sources = {str(p.relative_to(ROOT)): sha_file(p) for p in [DATA/'internal_test.csv',DATA/'validation_2000.csv',PREP/'latency_sample_manifest.csv',PREP/'latency_warmup_manifest.csv']}
    environment = {'cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                   'logical_cpus': os.cpu_count(), 'threads': 4, 'python': platform.python_version(), 'platform': platform.platform(),
                   'load_average_before': os.getloadavg(), 'cpu_exclusive': False, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                   'inference_batch_size': 1, 'warmup_rows':10, 'timed_rows':100, 'repetitions':3,
                   'max_narrative_chars':500, 'checkpoint_seed':42, 'precision':'FP32 encoders; float64 sparse',
                   'measurement_scope':'End to end: tokenisation/vectorisation, classifier inference and label decoding; model loading/warmup/I/O excluded.'}
    records, fits = [], []
    systems = ['tfidf_10000', 'tfidf_full'] if args.part == 'sparse' else ['bert', 'distilbert', 'roberta']
    for system in systems:
        if args.part == 'sparse':
            import joblib, sklearn
            from tune_revision_tfidf import vectorizer, fit_classifier
            budget = system.removeprefix('tfidf_'); directory = ROOT/'experiments/tfidf_revision_v1'/('budget_'+budget)
            selection = json.loads((directory/'selected_config.json').read_text())
            config = selection['config']; protocol = json.loads((ROOT/'analysis/revision_v1/protocol.json').read_text())['tfidf_search']
            train_path = DATA/('train_10000.csv' if budget == '10000' else 'train.csv')
            train = load_frame(train_path)
            stored_vec = joblib.load(directory/'selected_vectorizer.joblib'); stored = joblib.load(directory/'seed_42/classifier.joblib')
            with threadpool_limits(limits=4):
                started = time.perf_counter(); vec = vectorizer(config); x = vec.fit_transform(train.narrative.str[:500]); clf = fit_classifier(x,train['product'],config,protocol,42); fit_seconds = time.perf_counter()-started
                assert vec.vocabulary_ == stored_vec.vocabulary_; np.testing.assert_array_equal(vec.idf_,stored_vec.idf_)
                np.testing.assert_array_equal(clf.predict(vec.transform(test.narrative.str[:500])),stored.predict(stored_vec.transform(test.narrative.str[:500])))
            fits.append({'system':system,'seed':42,'train_rows':len(train),'training_seconds':fit_seconds,'device':'CPU','scope':'Fresh vectorizer fit_transform plus fixed selected classifier fit; validation/search/loading excluded','prediction_parity_rows':len(test)})
            sources[str(train_path.relative_to(ROOT))] = sha_file(train_path)
            for file in [directory/'selected_config.json',directory/'seed_42/classifier.joblib',directory/'selected_vectorizer.joblib']:
                sources[str(file.relative_to(ROOT))] = sha_file(file)
            def predict(text):
                return str(clf.predict(vec.transform([str(text)[:500]]))[0])
            environment['sklearn'] = sklearn.__version__
        else:
            import torch, transformers
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            torch.set_num_threads(4); torch.set_num_interop_threads(1) if system == 'bert' else None
            directory = EXP/'encoder'/system/'seed_42'/'model'
            tokenizer = AutoTokenizer.from_pretrained(directory,local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True).to('cpu').float().eval()
            assert all(p.device.type == 'cpu' for p in model.parameters())
            for file in directory.glob('*.safetensors'):sources[str(file.relative_to(ROOT))] = sha_file(file)
            environment.update(torch=torch.__version__,transformers=transformers.__version__)
            def predict(text):
                encoded = tokenizer(str(text)[:500],truncation=True,max_length=512,padding=True,pad_to_multiple_of=8,return_tensors='pt')
                with torch.inference_mode():index = int(model(**encoded).logits.argmax(-1).item())
                return str(model.config.id2label[index])
        for rid in warm:predict(valid.loc[rid,'narrative'])
        with threadpool_limits(limits=4):
            for repetition in range(3):
                ordered = sorted(ids,key=lambda rid:hashlib.sha256(f'order_v1|{repetition}|{rid}'.encode()).hexdigest())
                for order,rid in enumerate(ordered):
                    text = test.loc[rid,'narrative']; started = time.perf_counter(); prediction = predict(text); elapsed = time.perf_counter()-started
                    assert prediction in LABELS
                    records.append(dict(system=system,record_id=rid,repetition=repetition,order=order,seconds=elapsed,prediction=prediction,true_label=test.loc[rid,'product']))
        pd.DataFrame(records).to_csv(out/'latency_records.csv',index=False)
        if fits:pd.DataFrame(fits).to_csv(out/'training_records.csv',index=False)
        print(system,'completed; mean seconds',np.mean([r['seconds'] for r in records if r['system']==system]),flush=True)
        if args.part == 'encoders':del model,tokenizer
        else:del clf,vec,stored,stored_vec,x,train
        gc.collect()
    environment['load_average_after'] = os.getloadavg()
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    (out/'sources_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps({'completed':True,'cpu_only':True,'part':args.part,'records':len(records),'script_sha256':sha_file(Path(__file__)),'record_sha256':sha_file(out/'latency_records.csv')},indent=2)+'\n')

if __name__ == '__main__':main()
