"""Locations of generated LaTeX evidence; keep the manuscript directory tidy."""
from project_paths import project_path
from pathlib import Path
from project_paths import ROOT, BUILD, LABELS

PAPER = ROOT/'REVISION_PAPER'


def compiled_artifact(document: str, suffix: str) -> Path:
    """Prefer build/ while supporting previous extracted reviewer archives."""
    if document not in {'manuscript', 'supplementary_material', 'response_to_reviewers'}:
        raise ValueError('Unknown revision document: '+document)
    if suffix not in {'.aux', '.log'}:
        raise ValueError('Unsupported compiled evidence: '+suffix)
    current = project_path(PAPER, 'build', document+suffix)
    if current.exists():
        return current
    if suffix == '.aux' and (LABELS/(document+'.labels')).exists():
        return LABELS/(document+'.labels')
    return PAPER/(document+suffix)
