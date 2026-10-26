"""Shared frozen-data checks. Importing this module never imports torch or probes CUDA."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'REVISION_PAPER'
DATA = PAPER / 'data/revision_v1'
STUDY = PAPER / 'analysis/revision_v1'
PREP = PAPER / 'analysis/gpu_preparation'
CONFIG = PAPER / 'configs/gpu_revision_v1.json'
EXP = PAPER / 'experiments/gpu_revision_v1'
PROTOCOL_SHA = '89fd382208515e946ab2a941d07809cec8b9fc659d554048f008cf2dea225d0f'
# Read literal constants only: original GPU modules are not imported or changed.
_tree = ast.parse((ROOT / 'scripts/evaluate_adapter.py').read_text())
_ns = {}
for _node in _tree.body:
    if isinstance(_node, ast.Assign) and getattr(_node.targets[0], 'id', '') in {'LABELS', 'SYSTEM_PROMPT'}:
        exec(compile(ast.Module(body=[_node], type_ignores=[]), '<prompt constants>', 'exec'), _ns)
LABELS, SYSTEM_PROMPT = _ns['LABELS'], _ns['SYSTEM_PROMPT']
PRODUCT_RE = re.compile(r'"product"\s*:\s*"([^"]+)"')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n'


def write_new(path, value):
    """Idempotent for identical content; refuse to replace a frozen artifact."""
    path = Path(path)
    text = value if isinstance(value, str) else json_text(value)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'Refusing to overwrite frozen file: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def verify_data(include_final=False):
    if sha_file(STUDY / 'protocol.json') != PROTOCOL_SHA:
        raise RuntimeError('Original revision_v1 protocol hash changed')
    protocol = json.loads((STUDY / 'protocol.json').read_text())
    selected = ['train_10000.csv', 'validation_2000.csv']
    if include_final:
        selected += ['internal_test.csv', 'legacy_test_manifest.csv']
        selected += [f'legacy_subsets/{p.name}' for p in sorted((DATA / 'legacy_subsets').glob('*.csv'))]
    for rel in selected:
        path = DATA / rel
        expected = protocol['files_sha256'][str(path.relative_to(ROOT))]
        if sha_file(path) != expected:
            raise RuntimeError(f'Frozen dataset changed: {path}')
    return protocol


def load_frame(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if 'product' not in frame:
        frame = frame.rename(columns={'true_label': 'product'})
    if not {'product', 'narrative'}.issubset(frame):
        raise ValueError(f'Missing columns: {path}')
    if not set(frame['product']).issubset(LABELS) or frame.narrative.eq('').any():
        raise ValueError(f'Invalid labels/text: {path}')
    return frame


def selection_ids(frame):
    return [str(x) for x in frame.record_id]


def load_plan():
    expected = CONFIG.with_suffix('.sha256').read_text().split()[0]
    if sha_file(CONFIG) != expected:
        raise RuntimeError('GPU plan hash mismatch')
    return json.loads(CONFIG.read_text())


def pick_candidate(rows, tolerance=1e-12):
    """Tie rule uses the best score globally, not a non-transitive pairwise tie."""
    best = max(float(r['validation_macro_f1']) for r in rows)
    return min((r for r in rows if best - float(r['validation_macro_f1']) <= tolerance),
               key=lambda r: int(r['candidate_id']))


def artifact_entry(registry_path, model):
    entry = json.loads(Path(registry_path).read_text())[model]
    commit = entry.get('commit')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise RuntimeError(f'{model}: a real immutable 40-character snapshot commit is required')
    snapshot = Path(entry.get('snapshot') or '').expanduser().resolve()
    if snapshot.name != commit or snapshot.parent.name != 'snapshots' or not (snapshot / 'config.json').is_file():
        raise RuntimeError(f'{model}: use the actual local Hugging Face snapshots/<commit> directory')
    if snapshot.parent.parent.name != 'models--' + entry['repo_id'].replace('/', '--'):
        raise RuntimeError(f'{model}: snapshot cache repository does not match repo_id')
    return entry, snapshot


def snapshot_hashes(snapshot):
    return {str(p.relative_to(snapshot)): sha_file(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}


def audit_guard(registry_path, model):
    entry, snapshot = artifact_entry(registry_path, model)
    path = PREP / f'input_audit_{model}.json'
    audit = json.loads(path.read_text())
    if not audit['passed'] or audit['protocol_sha256'] != PROTOCOL_SHA or audit['plan_sha256'] != sha_file(CONFIG):
        raise RuntimeError('Missing/past/failed input audit')
    if audit['artifact'] != entry:
        raise RuntimeError('Artifact differs from audited model/tokenizer')
    if audit['common_code_sha256'] != sha_file(Path(__file__)):
        raise RuntimeError('Input serialization code changed after tokenizer audit')
    if audit['audit_code_sha256'] != sha_file(Path(__file__).with_name('audit_gpu_tokenizers.py')):
        raise RuntimeError('Tokenizer-audit code changed; rerun the CPU audit')
    for name, expected in audit['tokenizer_files_sha256'].items():
        if sha_file(snapshot / name) != expected:
            raise RuntimeError(f'Tokenizer changed after audit: {name}')
    return entry, snapshot, sha_file(path)


def messages(text, chars):
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Complaint:\n' + str(text)[:chars]}]


def prompt(tokenizer, text, chars):
    return tokenizer.apply_chat_template(messages(text, chars), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def parse_label(response):
    found = PRODUCT_RE.findall(response)
    label = found[-1].strip() if found else None
    return label if label in LABELS else '__off_label__'


def metrics(true, predicted):
    from sklearn.metrics import accuracy_score, f1_score
    return {'n': len(true), 'accuracy': float(accuracy_score(true, predicted)),
            'macro_f1': float(f1_score(true, predicted, labels=LABELS, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, predicted, labels=LABELS, average='weighted', zero_division=0)),
            'off_label_count': sum(p == '__off_label__' for p in predicted)}


def versions():
    names = ['torch', 'transformers', 'datasets', 'peft', 'trl', 'bitsandbytes', 'accelerate', 'pandas', 'scikit-learn']
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def llm_encoding(tokenizer, text, chars, tokens, offsets=False):
    rendered = prompt(tokenizer, text, chars)
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=True,
                        max_length=tokens, return_offsets_mapping=offsets)
    return rendered, encoded


def sft_record(tokenizer, text, label, max_length=512):
    rendered = prompt(tokenizer, text, 500)
    target = json.dumps({'product': label}, ensure_ascii=False) + tokenizer.eos_token
    ids = tokenizer(rendered + target, add_special_tokens=False)['input_ids']
    prompt_ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    if ids[:len(prompt_ids)] != prompt_ids or len(ids) > max_length:
        raise RuntimeError('SFT boundary mismatch or target truncation; protocol amendment needed')
    return {'input_ids': ids, 'completion_mask': [0] * len(prompt_ids) + [1] * (len(ids) - len(prompt_ids))}
