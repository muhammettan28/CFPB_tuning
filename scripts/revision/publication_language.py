"""Keep publication prose separate from the history of manuscript revisions."""
import re

# Identifiers, artifact filenames and citation keys are preserved verbatim.
TEX_IDENTIFIER = re.compile(
    r'(\\begin\{Verbatim\}.*?\\end\{Verbatim\}|'
    r'\\(?:label|ref|input|includegraphics|path|nolinkurl|cite)'
    r'(?:\[[^\]]*\])?\{[^}]*\})', re.S
)


def visible_parts(tex):
    return TEX_IDENTIFIER.split(tex)


def publication_terms(tex):
    parts = visible_parts(tex)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\bLegacy\b', 'Benchmark', parts[i])
        parts[i] = re.sub(r'\blegacy\b', 'benchmark', parts[i])
    return ''.join(parts)


def editorial_history_hits(tex):
    """Inspect visible text; legitimate model adaptation and token counts are allowed."""
    prose = '\n'.join(visible_parts(tex)[::2])
    patterns = [
        r'\b(?:revision|revised|historical|earlier|newly|pre-revision|post-revision)\b',
        r'\bold(?:\s*/\s*|\s+and\s+)new\b',
        r'\b(?:archived|original|previous|new)\s+(?:runs|experiments|models|allocation|protocol|study|results|predictions)\b',
        r'\b(?:before|after|pre)[ -](?:quarantine|revision)\b',
        r'\b(?:author review|author approval|submission task|remaining submission|not been uploaded)\b',
        r'\b(?:we (?:added|removed|replaced)|was corrected|initially requested)\b',
    ]
    return [{'pattern': pattern, 'text': m.group(0)}
            for pattern in patterns for m in re.finditer(pattern, prose, re.I)]


def internal_workflow_hits(tex):
    """Keep author-only preparation notes out of publication and response text."""
    prose = '\n'.join(visible_parts(tex)[::2])
    patterns = [
        r'\blocal author[ -]review version\b',
        r'\bit has not been submitted\b',
        r'\b(?:actual )?(?:external )?reviewer/public access (?:still )?(?:needs|must)\b',
        r'\bno external access mechanism\b',
        r'\bthe author must arrange\b',
        r'\bauthor submission tasks\b',
        r'\bkept this part of the request explicitly open\b',
        r'\bprivate repository\b',
        r'\bwithout narrating changes between manuscript versions\b',
        r'\bno reviewer/public upload was performed\b',
        r'\bfinal author review remain(?:s)? separate\b',
        r'\b(?:cleanup date|delivery-readiness receipt|completed local experiment analysis|local completion date)\b',
        r'\b(?:acknowledgment and/or disclaimers|funding|conflict of interest|AI-assisted technologies), if any\b',
    ]
    return [{'pattern': pattern, 'text': m.group(0)}
            for pattern in patterns for m in re.finditer(pattern, prose, re.I)]
