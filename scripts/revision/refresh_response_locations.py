#!/usr/bin/env python3
"""Refresh response locations from the actual clean, tracked and supplementary PDFs.

Only location paragraphs and their introductory convention are changed. Existing
author edits to the reviewer quotations and response prose are preserved.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'REVISION_PAPER'
REPORT = ROOT / 'analysis/postprocessing/RESPONSE_LOCATIONS.json'

# Add the omitted evidence to the relevant response, rather than appending every
# manuscript section to every comment.
ADDITIONS = {
    'R1.3': ['Manuscript Data availability statement'],
    'R1.7': ['Manuscript Section 1', 'Manuscript Section 2'],
    'R1.8': ['Manuscript Section 4.3', 'Manuscript Section 4.4'],
    'R2.1': ['Manuscript Section 1', 'Manuscript Section 2'],
    'R2.2': ['Manuscript Section 4.2'],
    'R2.3': ['Manuscript Section 4.3'],
    'R2.5': ['Manuscript Section 4.2'],
    'R2.6': ['Manuscript Abstract'],
    'R2.7': ['Manuscript Data availability statement'],
}
CONVENTION = ('For manuscript locations, page numbers are given for both the clean '
              'and yellow-highlighted (tracked) versions. Supplementary locations '
              'refer to the supplementary material.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c)).split()


@lru_cache(None)
def pages(stem):
    text = subprocess.check_output(
        ['pdftotext', '-layout', str(PAPER/(stem+'.pdf')), '-'], text=True)
    return text.rstrip('\f').split('\f')


@lru_cache(None)
def section_titles(stem):
    section = subsection = 0
    titles = {}
    prefix = 'S' if stem == 'supplementary_material' else ''
    for kind, title in re.findall(r'\\(section|subsection)\{([^{}]+)\}',
                                  (PAPER/(stem+'.tex')).read_text()):
        if kind == 'section':
            section += 1
            subsection = 0
            number = prefix+str(section)
        else:
            subsection += 1
            number = prefix+str(section)+'.'+str(subsection)
        titles[number] = title
    return titles


def locate(stem, kind, number=None, title=None):
    if kind == 'Section':
        title = section_titles('manuscript' if stem == 'tracked' else stem)[number]
        pattern = r'^\s*(?:\d+\s+)?'+re.escape(number)+r'\.?\s+(.+?)\s*$'
    elif kind == 'Table':
        pattern = r'^\s*Table\s+'+re.escape(number)+r'[:.]\s+(.+)'
    elif kind == 'Abstract':
        pattern = r'^\s*(?:\d+\s+)?Abstract:\s+(.+)'
    else:
        pattern = r'^\s*(?:\d+\s+)?'+re.escape(kind)+r'\s*$'
    found = []
    for page, content in enumerate(pages(stem), 1):
        # The supplement's first page contains a table of contents, not sections.
        if stem == 'supplementary_material' and page == 1:
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            match = re.match(pattern, line)
            if match and (kind != 'Section' or normal(match.group(1)) == normal(title)):
                found.append({'document': stem, 'page': page, 'line': line_number,
                              'matched_line': line.strip()})
    if len(found) != 1:
        raise ValueError(f'Expected one PDF heading: {stem}, {kind}, {number}: {found}')
    return found[0]


def location(key):
    match = re.fullmatch(r'(Manuscript|Supplementary material) (Section|Table) ([\dS.]+)', key)
    if match:
        document, kind, number = match.groups()
    else:
        document, kind, number = 'Manuscript', key.removeprefix('Manuscript '), None
        if kind not in {'Abstract', 'Data availability statement'}:
            raise ValueError('Unrecognised response location: '+key)
    stem = 'manuscript' if document == 'Manuscript' else 'supplementary_material'
    evidence = [locate(stem, kind, number)]
    if document == 'Manuscript':
        evidence.append(locate('tracked', kind, number))
        qualifier = ''
        if kind == 'Section' and number in {'1', '2'}:
            qualifier = ', '+section_titles('manuscript')[number]
        clean_page, tracked_page = (e['page'] for e in evidence)
        if clean_page == tracked_page:
            display = key+qualifier+f" (clean/tracked p. {clean_page})"
        else:
            display = key+qualifier+f" (clean p. {clean_page}; tracked p. {tracked_page})"
    else:
        display = key+f" (p. {evidence[0]['page']})"
    return {'document': stem, 'kind': kind, 'number': number,
            'key': key, 'display': display, 'evidence': evidence}


def refresh(path=PAPER/'response_to_reviewers.tex'):
    original = path.read_text()
    records, changes = [], []
    pattern = (r'(\\subsection\*\{(R[12]\.\d+) [^\n]+\}.*?'
               r'\\textbf\{Changes and locations\.\} )([^\n]+)')

    def replace(match):
        response_id = match.group(2)
        old = match.group(3)
        keys = []
        for item in re.split(r'; (?=(?:Manuscript|Supplementary material) )', old.rstrip('.')):
            key = re.sub(r' \((?:p\.|clean(?:/tracked)? p\.).*\)$', '', item)
            key = re.sub(r', (?:Introduction|Related work)$', '', key)
            keys.append(key)
        keys += [key for key in ADDITIONS.get(response_id, []) if key not in keys]
        entries = [location(key) for key in keys]
        # Follow the actual document order, including sections on the same page.
        entries.sort(key=lambda e: (e['document'] != 'manuscript',
                                   0 if e['kind'] == 'Abstract' else e['evidence'][0]['page'],
                                   e['evidence'][0]['line']))
        for entry in entries:
            records.append(dict(response_id=response_id, **entry))
        updated = '; '.join(entry['display'] for entry in entries)+'.'
        changes.append({'response_id': response_id, 'before': old, 'after': updated})
        return match.group(1)+updated

    updated, count = re.subn(pattern, replace, original, flags=re.S)
    if count != 15 or len({e['response_id'] for e in records}) != 15:
        raise ValueError('Expected exactly 15 complete response location paragraphs')
    old_convention = 'All document locations refer to the clean manuscript and supplementary material.'
    if old_convention in updated:
        updated = updated.replace(old_convention, CONVENTION, 1)
    elif CONVENTION not in updated:
        raise ValueError('Review the location convention before updating it')
    # No quote, response argument, figure, table or scientific result is rewritten.
    strip = lambda text: re.sub(r'\\textbf\{Changes and locations\.\} [^\n]+', '', text)
    assert strip(original).replace(old_convention, CONVENTION, 1) == strip(updated)
    report = {'schema': 'rendered_pdf_locations_v2', 'units': 15,
              'references': records, 'changes': changes,
              'source_pdf_sha256': {stem: sha(PAPER/(stem+'.pdf')) for stem in
                                    ['manuscript', 'tracked', 'supplementary_material']},
              'response_arguments_and_quotes_unchanged': True,
              'page_numbers_verified_in_actual_pdfs': True}
    path.write_text(updated)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def validate(report):
    """Recheck saved page evidence against the PDFs that will be submitted."""
    for stem, expected in report['source_pdf_sha256'].items():
        assert sha(PAPER/(stem+'.pdf')) == expected, stem
    for entry in report['references']:
        current = location(entry['key'])
        assert current['evidence'] == entry['evidence'], entry
        assert current['display'] == entry['display'], entry
    return len(report['references'])


if __name__ == '__main__':
    report = refresh()
    validate(report)
    print(f"Updated {report['units']} responses; checked {len(report['references'])} locations in the actual PDFs")
