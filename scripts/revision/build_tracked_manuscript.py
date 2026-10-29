#!/usr/bin/env python3
"""Build an independent tracked manuscript from submitted and clean LaTeX sources.

Only the current manuscript is displayed. Added or replaced text is highlighted
in yellow; deleted content is omitted. Original and clean files are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FLOAT = re.compile(r'\\begin\{(table|figure)\}.*?\\end\{\1\}', re.S)
LABEL = re.compile(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError('Recursive input: '+str(path))
    def include(match):
        target = path.parent / match.group(1)
        if not target.suffix:
            target = target.with_suffix('.tex')
        return flatten(target, (*stack, path))
    return re.sub(r'\\input\{([^}]+)\}', include, path.read_text())


def compile_tex(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 7):
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                              '-output-directory='+str(output), source.name],
                             cwd=source.parent, capture_output=True, text=True)
        (output/(source.stem+'_build.txt')).write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(run.stdout[-3500:])
        log = (output/source.with_suffix('.log').name).read_text()
        # The journal's pagewise line numbers need a third pass in a fresh folder.
        if attempt >= 3 and not re.search(
                r'Label\(s\) may have changed|Table widths have changed|There were undefined|Rerun to get|Citation\(s\) may have changed', log):
            return log
    raise RuntimeError('References did not stabilise: '+source.name)


def canonical_body(text):
    text = text.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
    text = re.sub(r'% TRACKED-LEGEND-BEGIN.*?% TRACKED-LEGEND-END', '', text, flags=re.S)
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    text = text.replace(r'\color{blue}', '')
    return re.sub(r'\s+', '', text)


def unwrap_highlights(text):
    "Remove only our one-argument markup, retaining nested LaTeX verbatim."
    token = r'\TrackedHighlight{'
    while token in text:
        start = text.index(token)
        pos, depth = start + len(token), 1
        while depth:
            if pos >= len(text):
                raise ValueError('Unbalanced highlight argument')
            if text[pos] == '\\':
                pos += 2
                continue
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        content = text[start+len(token):pos-1]
        text = text[:start]+content+text[pos:]
    return text.replace(r'\rowcolors{1}{TrackedYellow}{TrackedYellow}', '')


def build(original, clean, output, work):
    if output in {original, clean} or output.is_relative_to(original.parent) or work.is_relative_to(original.parent):
        raise ValueError('Original sources are read-only; tracked output must be separate from both manuscripts')
    original_hash, clean_hash = digest(original), digest(clean)
    old, new = flatten(original), flatten(clean)
    normal_title = lambda text: re.sub(r'\\begin\{document\}\s*\\maketitle\s*',
                                      lambda _: '\\begin{document}\n\\maketitle\n\n', text)
    old, new = normal_title(old), normal_title(new)
    work.mkdir(parents=True, exist_ok=True)
    blocks, identities, counts = {}, {}, {}
    # Float schemas are compared as complete objects. Only current objects survive
    # acceptance, so deleted tables, captions and graphics never reach the PDF.
    for role, text in [('Old', old), ('New', new)]:
        def atom(match):
            block = match.group(0)
            identity = re.sub(r'\s+', '', block)
            if identity not in identities:
                ident = 'F'+''.join(chr(65+int(n)) for n in str(len(identities)))
                identities[identity] = r'\TrackedFloat{'+ident+'}'
            token = identities[identity]
            if token not in blocks:
                blocks[token] = {'kind': match.group(1), 'block': block, 'roles': set()}
            blocks[token]['roles'].add(role)
            return '\n\n'+token+'\n\n'
        text = FLOAT.sub(atom, text)
        counts[role] = len(FLOAT.findall(old if role == 'Old' else new))
        (work/(role.lower()+'_atomic.tex')).write_text(text)
    run = subprocess.run(['latexdiff', '--type=UNDERLINE', '--subtype=SAFE',
                          '--graphics-markup=none', '--math-markup=0',
                          '--append-textcmd=keywords', '--disable-citation-markup', '--no-label',
                          str(work/'old_atomic.tex'), str(work/'new_atomic.tex')],
                         capture_output=True, text=True, check=True)
    (work/'latexdiff_stderr.txt').write_text(run.stderr)
    # Preserve additions under our own name while latexrevise discards every
    # deletion and all diff scaffolding. Start from the clean preamble below.
    marked = re.sub(r'\\DIFadd(?:FL)?\{', lambda _: r'\TrackedHighlight{', run.stdout)
    accepted_run = subprocess.run(['latexrevise', '--accept'], input=marked,
                                 capture_output=True, text=True, check=True)
    (work/'latexrevise_stderr.txt').write_text(accepted_run.stderr)
    accepted = accepted_run.stdout
    body = accepted.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
    changed_objects = 0
    for token, obj in blocks.items():
        if 'New' not in obj['roles']:
            if token in body:
                raise RuntimeError('Deleted object survived acceptance: '+token)
            continue
        if body.count(token) != 1:
            raise RuntimeError('Current object lost or duplicated: '+token)
        block = obj['block']
        if obj['roles'] == {'New'}:
            changed_objects += 1
            # Caption highlighting remains readable, and tables use background
            # colour without changing their contents, widths or row spacing.
            block = block.replace(r'\caption{', r'\caption{\TrackedHighlight{', 1)
            # Find the caption's balanced closing brace (not nested \cite etc.).
            start = block.index(r'\caption{')+len(r'\caption{\TrackedHighlight{')
            pos, depth = start, 1
            while depth:
                if block[pos] == '\\':
                    pos += 2
                    continue
                if block[pos] == '{': depth += 1
                elif block[pos] == '}': depth -= 1
                pos += 1
            block = block[:pos]+'}'+block[pos:]
            if obj['kind'] == 'table':
                block = re.sub(r'(\\begin\{table\}(?:\[[^]]*\])?)',
                               lambda m: m.group(1)+'\n'+r'\rowcolors{1}{TrackedYellow}{TrackedYellow}', block, count=1)
        body = body.replace(token, block)
    preamble = new.split(r'\begin{document}', 1)[0]
    preamble += r'''
% Current manuscript only: yellow denotes added or replaced material.
% Deleted text and all deletion markup are deliberately omitted.
\let\TrackedMathUnderline\ul
\let\ul\relax
\usepackage{soul,colortbl}
\let\ul\TrackedMathUnderline
\definecolor{TrackedYellow}{RGB}{255,245,128}
\sethlcolor{TrackedYellow}
\makeatletter
\def\SOUL@hlpreamble{\setul{}{2.2ex}\let\SOUL@stcolor\SOUL@hlcolor\SOUL@stpreamble}
\makeatother
\DeclareRobustCommand{\TrackedHighlight}[1]{\hl{#1}}
\soulregister\cite7
\soulregister\ref7
\soulregister\texttt7
\soulregister\textbf7
\soulregister\emph7
\soulregister\textit7
\soulregister\url7
\pdfstringdefDisableCommands{\def\TrackedHighlight#1{#1}}
\hypersetup{pdftitle={QLoRA Fine-Tuning for Consumer Complaint Classification -- Highlighted changes}}
'''
    tracked = preamble+r'\begin{document}'+body+r'\end{document}'+'\n'
    projection = unwrap_highlights(tracked)
    (work/'accepted_projection.tex').write_text(projection)
    if canonical_body(projection) != canonical_body(new):
        raise RuntimeError('Removing yellow highlights does not recover the complete clean manuscript')
    if re.search(r'\\DIF(?:add|del)|\\sout|TrackedOld|TrackedDeleted', body):
        raise RuntimeError('Deleted text or diff markup survived acceptance')
    output.write_text(tracked)
    assert digest(original) == original_hash and digest(clean) == clean_hash
    return {'original_tex_sha256': original_hash, 'clean_tex_sha256': clean_hash,
            'tracked_tex_sha256': digest(output), 'original_objects': counts['Old'],
            'current_objects': counts['New'], 'highlighted_current_objects': changed_objects,
            'original_graphic_assets': [], 'displayed_deletions': 0,
            'presentation': 'current manuscript only; yellow highlights on added or replaced material',
            'accepted_projection_matches_clean_body': True,
            'highlighted_text_spans': body.count(r'\TrackedHighlight{')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ROOT/'ORJ_PAPER/manuscript.tex')
    parser.add_argument('--clean', type=Path, default=ROOT/'REVISION_PAPER/manuscript.tex')
    parser.add_argument('--output', type=Path, default=ROOT/'REVISION_PAPER/tracked.tex')
    parser.add_argument('--work', type=Path, default=ROOT/'.build/revision_paper/tracked_work')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    report = build(args.original.resolve(), args.clean.resolve(), args.output.resolve(), args.work.resolve())
    if args.compile:
        log = compile_tex(args.output.resolve(), args.work.resolve())
        if re.search(r'There were undefined|Citation .* undefined|Reference .* undefined|multiply defined', log):
            raise RuntimeError('Unresolved or duplicate references in tracked manuscript')
        pdf = args.work/args.output.with_suffix('.pdf').name
        shutil.copy2(pdf, args.output.with_suffix('.pdf'))
        report['tracked_pdf_sha256'] = digest(pdf)
    (args.work/'BUILD_TRACKED_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
