#!/usr/bin/env python3
"""CPU/disk-only validation of pinned cache files against fetched Hub metadata."""
from __future__ import annotations
from project_paths import project_path
import argparse
import hashlib
import json
import math
import struct
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

PAPER=Path(__file__).resolve().parents[2]/'REVISION_PAPER'
PREP=project_path(PAPER, 'analysis/gpu_preparation')


def digest(path,git=False):
    h=hashlib.sha1() if git else hashlib.sha256()
    if git:h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safetensors_check(path):
    size=path.stat().st_size
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Missing header length')
        length=struct.unpack('<Q',raw)[0]
        if length>100_000_000 or 8+length>size:raise ValueError('Invalid header length')
        header=json.loads(f.read(length))
    intervals=[]
    for name,value in header.items():
        if name=='__metadata__':continue
        a,b=value['data_offsets']
        if not 0<=a<=b<=size-8-length:raise ValueError('Out-of-bounds tensor')
        intervals.append((a,b))
    end=0
    for a,b in sorted(intervals):
        if a!=end:raise ValueError('Tensor offsets have gaps/overlap')
        end=b
    if end!=size-8-length:raise ValueError('Tensor data size mismatch')
    return {'tensor_count':len(intervals),'header_and_offsets_valid':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/hub')
    args=parser.parse_args()
    specs=json.loads((project_path(PAPER, 'configs/model_registry.template.json')).read_text())
    remote=json.loads((PREP/'MODEL_REMOTE_METADATA.json').read_text())
    registry={};results={}
    for model,spec in specs.items():
        repo=args.cache_dir/('models--'+spec['repo_id'].replace('/','--'))
        commit=(repo/'refs/main').read_text().strip()
        snapshot=repo/'snapshots'/commit
        registry[model]={'repo_id':spec['repo_id'],'commit':commit,'snapshot':str(snapshot.resolve())}
        info=remote[model]
        if not info['metadata_verified'] or info['commit']!=commit:
            raise RuntimeError(f'{model}: remote revision metadata is not verified')
        config=json.loads((snapshot/'config.json').read_text())
        index_path=snapshot/'model.safetensors.index.json'
        index=json.loads(index_path.read_text()) if index_path.exists() else None
        expected=sorted(set(index['weight_map'].values())) if index else ['model.safetensors']
        file_checks=[];weight_headers={}
        for path in sorted(snapshot.rglob('*')):
            if path.is_dir():continue
            rel=str(path.relative_to(snapshot))
            if not path.exists():raise RuntimeError(f'Broken cache symlink: {path}')
            metadata=info['files'].get(rel)
            if metadata is None:raise RuntimeError(f'File absent from pinned Hub revision: {rel}')
            expected_hash=metadata.get('lfs_sha256') or metadata['git_blob_id']
            actual=digest(path,git=metadata.get('lfs_sha256') is None)
            if actual!=expected_hash or path.stat().st_size!=metadata['size']:
                raise RuntimeError(f'Checksum/size mismatch: {model}/{rel}')
            file_checks.append({'file':rel,'bytes':path.stat().st_size,'hash':actual,
                                'algorithm':'sha256' if metadata.get('lfs_sha256') else 'git_blob_sha1','matches_hub':True})
            if rel in expected:weight_headers[rel]=safetensors_check(path)
        missing=[name for name in expected if not (snapshot/name).is_file()]
        missing_bytes=sum(info['files'][name]['size'] for name in missing)
        if not missing and index:
            actual_names=set()
            for name in expected:
                with (snapshot/name).open('rb') as f:
                    n=struct.unpack('<Q',f.read(8))[0]
                    actual_names.update(k for k in json.loads(f.read(n)) if k!='__metadata__')
            if actual_names!=set(index['weight_map']):raise RuntimeError('Tensor index/header mismatch')
        results[model]={'repo_id':spec['repo_id'],'commit':commit,'model_type':config['model_type'],
            'snapshot':str(snapshot),'expected_weight_files':expected,'missing_weight_files':missing,
            'missing_weight_bytes':missing_bytes,'all_required_weights_present':not missing,
            'present_file_checks':file_checks,'safetensors_headers':weight_headers,
            'status':'weights_present_and_hub_hashes_verified' if not missing else 'incomplete_weights_missing'}
        print(f'{model}: {len(file_checks)} existing files match pinned Hub hashes; {len(missing)} weight files missing',flush=True)
    registry_path=project_path(PAPER, 'configs/model_registry.local.json')
    text=json.dumps(registry,indent=2)+'\n'
    if registry_path.exists() and registry_path.read_text()!=text:
        raise RuntimeError('Existing registry differs; refusing to replace pins')
    registry_path.write_text(text)
    report={'date':datetime.now(ZoneInfo('Europe/Istanbul')).date().isoformat(),'models':results,'complete_models':sum(v['all_required_weights_present'] for v in results.values()),
        'required_models':5,'all_models_complete':all(v['all_required_weights_present'] for v in results.values()),
        'missing_weight_bytes_total':sum(v['missing_weight_bytes'] for v in results.values()),
        'method':'Pinned Hub API revision/file metadata, full local file hashes, safetensors header/offset inspection. No weights loaded into a model.',
        'gpu_started':False,'download_started':False,'registry_records_identity_only_not_weight_readiness':True}
    (PREP/'MODEL_DOWNLOAD_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    # One include pattern: safe with the locally installed HF CLI. Existing files are reused.
    lines=['#!/usr/bin/env bash','set -euo pipefail','',
           '# Downloads only; no GPU/model execution. Uses the exact revisions already cached.',
           'cfpb_hf_cli="${CFPB_HF_CLI:-$HOME/.venvs/cfpb-download/bin/hf}"',
           'if [[ ! -x "$cfpb_hf_cli" ]]; then echo "HF CLI missing: $cfpb_hf_cli" >&2; exit 1; fi','']
    import shlex
    for model,item in results.items():
        if item['missing_weight_files']:
            lines.append('"$cfpb_hf_cli" download '+shlex.quote(item['repo_id'])+' --revision '+item['commit']+' --include "*.safetensors"')
    lines+=['','echo "Download commands completed; rerun verify_model_downloads.py to check weights."']
    (project_path(PAPER, 'scripts/complete_model_downloads.sh')).write_text('\n'.join(lines)+'\n')
    print(f'Complete models: {report["complete_models"]}/5; missing {report["missing_weight_bytes_total"]:,} bytes. No download/GPU started.')


if __name__=='__main__':main()
