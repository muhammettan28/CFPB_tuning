#!/usr/bin/env python3
"""Build an independent tracked manuscript from submitted and clean LaTeX sources.

Text uses latexdiff word markup. Changed floats are kept as complete objects so
different table schemas cannot be interleaved. Original files are read-only.
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


def build(original, clean, output, work):
    if output in {original, clean} or output.is_relative_to(original.parent) or work.is_relative_to(original.parent):
        raise ValueError('Original sources are read-only; tracked output must be separate from both manuscripts')
    original_hash, clean_hash = digest(original), digest(clean)
    old = flatten(original)
    new = flatten(clean)
    normal_title = lambda text: re.sub(r'\\begin\{document\}\s*\\maketitle\s*',
                                      lambda _: '\\begin{document}\n\\maketitle\n\n', text)
    old, new = normal_title(old), normal_title(new)
    baseline = work/'baseline'
    baseline.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, baseline/'manuscript.tex')
    for name in ['elektr.cls', 'elksty.tex', 'elk-creative.eps',
                 'elk-creative-eps-converted-to.pdf']:
        shutil.copy2(original.parent/name, baseline/name)
    for directory in ['tables', 'figures']:
        (baseline/directory).mkdir(exist_ok=True)
        for path in (original.parent/directory).glob('*'):
            if path.suffix in {'.tex', '.pdf', '.png'}:
                shutil.copy2(path, baseline/directory/path.name)
    compile_tex(baseline/'manuscript.tex', baseline)
    labels = {key: number for key, number, _ in LABEL.findall((baseline/'manuscript.aux').read_text())}
    # Deleted references retain submitted numbers without unresolved/deleted labels.
    old = re.sub(r'\\ref\{([^}]+)\}', lambda m: labels[m.group(1)], old)
    assets = []
    def graphic(match):
        path = original.parent/match.group(2)
        if not path.suffix:
            path = path.with_suffix('.pdf')
        dest = output.parent/'figures'/('tracked_original_'+path.name)
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        assets.append(str(dest.relative_to(output.parent)))
        return r'\includegraphics'+(match.group(1) or '')+'{'+str(dest.relative_to(output.parent))+'}'
    old = re.sub(r'\\includegraphics(\[[^]]*\])?\{([^}]+)\}', graphic, old)
    blocks = {}
    counts = {}
    for role, text in [('Old', old), ('New', new)]:
        def atom(match):
            ident = 'F'+''.join(chr(65+int(n)) for n in str(len(blocks)))
            token = r'\Tracked'+role+'Float{'+ident+'}'
            blocks[token] = (role, match.group(1), match.group(0))
            return '\n\n'+token+'\n\n'
        text = FLOAT.sub(atom, text)
        counts[role] = sum(value[0] == role for value in blocks.values())
        (work/(role.lower()+'_atomic.tex')).write_text(text)
    run = subprocess.run(['latexdiff', '--type=UNDERLINE', '--subtype=SAFE',
                          '--graphics-markup=none', '--math-markup=0',
                          '--append-textcmd=keywords', '--disable-citation-markup', '--no-label',
                          str(work/'old_atomic.tex'), str(work/'new_atomic.tex')],
                         capture_output=True, text=True, check=True)
    (work/'latexdiff_stderr.txt').write_text(run.stderr)
    tracked = run.stdout
    for token, (role, kind, block) in blocks.items():
        if tracked.count(token) != 1:
            raise RuntimeError('Opaque object lost or duplicated: '+token)
        if role == 'Old':
            tracked = tracked.replace('%DIFDELCMD < '+token, token)
            label = re.search(r'\\label\{([^}]+)\}', block)
            number = labels[label.group(1)] if label else '?'
            block = re.sub(r'\\label\{[^}]+\}', '', block)
            block = block.replace(r'\caption{', r'\TrackedDeletedCaption{'+kind.title()+'}{'+number+'}{')
        color = 'red' if role == 'Old' else 'blue'
        block = re.sub(r'(\\begin\{(?:table|figure)\}(?:\[[^]]*\])?)',
                       lambda m: m.group(1)+r'\color{'+color+'}', block, count=1)
        tracked = tracked.replace(token, block)
    # latexdiff rolls back a deleted section counter, but LaTeX also resets its
    # subsection counter. Use non-numbered review headings for deleted headings
    # so current sections retain the clean manuscript's numbering.
    tracked = re.sub(
        r'\\(section|subsection)\{\\DIFdel\{([^{}]*)\}\}(?:\s|%DIFAUXCMD[^\n]*\n)*'
        r'\\addtocounter\{\1\}\{-1\}%DIFAUXCMD',
        lambda m: r'\TrackedDeletedHeading{'+m.group(1)+'}{'+m.group(2)+'}', tracked)
    preamble = r'''
% Review markings do not affect the independently maintained clean manuscript.
\hypersetup{hypertexnames=false,pdftitle={QLoRA Fine-Tuning for Consumer Complaint Classification -- Tracked changes}}
\newcommand{\TrackedDeletedCaption}[3]{\par\smallskip\noindent\textbf{Submitted #1 #2 (deleted).} #3\par\smallskip}
\newcommand{\TrackedDeletedHeading}[2]{\par\medskip\noindent\textbf{\textcolor{red}{\sout{#2}}}\par\nobreak\smallskip}
\setlength{\emergencystretch}{4em}
'''
    tracked = tracked.replace(r'\begin{document}', preamble+'\n'+r'\begin{document}', 1)
    legend = r'''
% TRACKED-LEGEND-BEGIN
\begingroup\small\noindent
\textbf{Tracked changes.} \DIFadd{Blue underlined text is added};
\DIFdel{red struck-through text is deleted}. Unmarked text is retained.
Blue tables and figure captions identify added or replacement objects;
red objects labelled ``Submitted ... (deleted)'' retain the submitted content.
Locations in the response letter refer to the clean manuscript.
\par\endgroup\medskip
% TRACKED-LEGEND-END
'''
    tracked = tracked.replace(r'\maketitle', r'\maketitle'+'\n'+legend, 1)
    output.write_text(tracked)
    projection_input = re.sub(r'% TRACKED-LEGEND-BEGIN.*?% TRACKED-LEGEND-END',
                              '', tracked, flags=re.S)
    accepted = subprocess.run(['latexrevise', '--accept'], input=projection_input,
                             capture_output=True, text=True, check=True).stdout
    (work/'accepted_projection.tex').write_text(accepted)
    if canonical_body(accepted) != canonical_body(new):
        raise RuntimeError('Accepting changes does not recover the complete clean manuscript')
    assert digest(original) == original_hash and digest(clean) == clean_hash
    return {'original_tex_sha256': original_hash, 'clean_tex_sha256': clean_hash,
            'tracked_tex_sha256': digest(output), 'original_objects': counts['Old'],
            'current_objects': counts['New'], 'original_graphic_assets': assets,
            'accepted_projection_matches_clean_body': True,
            'markup_additions': tracked.count(r'\DIFadd{'),
            'markup_deletions': tracked.count(r'\DIFdel{')}


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
