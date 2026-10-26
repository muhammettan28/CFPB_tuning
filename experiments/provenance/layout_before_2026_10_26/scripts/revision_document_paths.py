"""Locations of generated LaTeX evidence; keep the manuscript directory tidy."""
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]


def compiled_artifact(document: str, suffix: str) -> Path:
    """Prefer build/ while supporting previous extracted reviewer archives."""
    if document not in {'manuscript', 'supplementary_material', 'response_to_reviewers'}:
        raise ValueError('Unknown revision document: '+document)
    if suffix not in {'.aux', '.log'}:
        raise ValueError('Unsupported compiled evidence: '+suffix)
    current = PAPER/'build'/(document+suffix)
    return current if current.exists() else PAPER/(document+suffix)
