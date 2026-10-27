"""Resolve recorded pre-relocation paths without rewriting frozen evidence.

JSON protocols, run metadata and source manifests retain their original bytes.
Only maintained tools translate those recorded locations to the current layout.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'REVISION_PAPER'
BUILD = ROOT / '.build/revision_paper'
LABELS = ROOT / 'analysis/document_labels'
PATH_MAPPING = {
    'REVISION_PAPER/protocols/GPU_EXPERIMENT_PROTOCOL.md': 'experiments/provenance/execution_plans/GPU_EXPERIMENT_PROTOCOL.md',
    'REVISION_PAPER/protocols/REF14_BASELINE_PROTOCOL.md': 'experiments/provenance/execution_plans/REF14_BASELINE_PROTOCOL.md',
    'REVISION_PAPER/experiments': 'experiments',
    'REVISION_PAPER/analysis': 'analysis',
    'REVISION_PAPER/configs': 'configs',
    'REVISION_PAPER/data/revision_v1': 'dataset/revision_v1',
    'REVISION_PAPER/scripts': 'scripts/revision',
    'REVISION_PAPER/protocols': 'protocols',
    'REVISION_PAPER/templates': 'templates/revision_paper',
    'REVISION_PAPER/build': '.build/revision_paper',
    'REVISION_PAPER/reviewer_package': 'reviewer_package',
    'REVISION_PAPER/literature': 'literature',
    'REVISION_PAPER/REVISION_LOG.md': 'docs/revision_history/REVISION_LOG.md',
    'REVISION_PAPER/TWO_PASS_REVIEW.md': 'docs/revision_history/TWO_PASS_REVIEW.md',
    'REVISION_PAPER/README.md': 'docs/revision_history/PREVIOUS_REVISION_GUIDE.md',
}


def _translate(path, mapping):
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        prefix = Path(old)
        if path.is_relative_to(prefix):
            return Path(new) / path.relative_to(prefix)
    return path


def project_path(*parts):
    """Accept current or historical project paths; never modify an evidence file."""
    path = Path(*parts)
    if path.is_absolute():
        if not path.is_relative_to(ROOT):
            return path
        path = path.relative_to(ROOT)
    return ROOT / _translate(path, PATH_MAPPING)


def recorded_path(path, root=ROOT):
    """Recover the original key used in the immutable data-protocol hash map."""
    relative = Path(path).relative_to(root)
    return str(_translate(relative, {new: old for old, new in PATH_MAPPING.items()}))
