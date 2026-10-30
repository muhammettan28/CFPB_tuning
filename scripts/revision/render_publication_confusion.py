#!/usr/bin/env python3
"""Render the saved confusion counts for publication, without rerunning models.

Requires Times New Roman installed locally or its licensed TTF files supplied
with --font-directory. Fonts are embedded in the PDF, not redistributed as files.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'analysis/postprocessing/corrected_confusion_counts.npz'
FIGURES = ROOT/'REVISION_PAPER/figures'
NAMES = {'qwen_base': 'Qwen3 base', 'qwen_ft': 'Qwen3 QLoRA',
         'llama_base': 'Llama 3.1 base', 'llama_ft': 'Llama 3.1 QLoRA'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(font_directory=None):
    font_directory = Path(font_directory or Path.home()/'.cache/cfpb-publication-fonts')
    if font_directory.exists():
        for path in font_directory.glob('*.ttf'):
            font_manager.fontManager.addfont(path)
    font = Path(font_manager.findfont('Times New Roman', fallback_to_default=False))
    source_hash = sha(SOURCE)
    FIGURES.mkdir(exist_ok=True)
    files, annotations, values = {}, {}, {}
    settings = {'font.family': 'Times New Roman', 'font.size': 8,
                'axes.titlesize': 9, 'axes.labelsize': 8,
                'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                'axes.linewidth': 0.6, 'lines.linewidth': 0.7,
                'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                'pdf.fonttype': 42, 'ps.fonttype': 42}
    # Keep the aspect ratio of the existing figure; use an explicit 16-cm width.
    with plt.rc_context(settings), np.load(SOURCE) as matrices:
        fig, axes = plt.subplots(2, 2, figsize=(16/2.54, (16*7.8/9)/2.54),
                                 constrained_layout=True)
        display = ['Banking', 'Credit rep.', 'Card', 'Loans', 'Debt']
        for ax, key in zip(axes.flat, NAMES):
            counts = matrices[key]
            # Preserve the original estimator, including off-label predictions
            # in the denominator and three-refit mean counts for adapted models.
            normalized = counts[:5, :5]/counts[:5].sum(axis=1, keepdims=True)
            values[key] = normalized.tolist()
            annotations[key] = [[f'{value:.2f}' for value in row] for row in normalized]
            image = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(5), display, rotation=30, ha='right')
            ax.set_yticks(range(5), display)
            ax.set_title(NAMES[key])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, annotations[key][i][j], ha='center', va='center',
                            fontsize=7.3, color='white' if normalized[i, j] > .55 else 'black')
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7,
                     label='Within-true-class fraction')
        for extension, dpi in [('pdf', 300), ('png', 600), ('tiff', 1200)]:
            target = FIGURES/('corrected_confusion_matrices.'+extension)
            kwargs = {'pil_kwargs': {'compression': 'tiff_lzw'}} if extension == 'tiff' else {}
            fig.savefig(target, dpi=dpi, facecolor='white', **kwargs)
            if extension == 'tiff':
                with Image.open(target) as image:
                    rgb = image.convert('RGB')
                rgb.save(target, compression='tiff_lzw', dpi=(dpi, dpi))
            files[extension] = {'path': str(target.relative_to(ROOT)), 'sha256': sha(target)}
        plt.close(fig)
    assert sha(SOURCE) == source_hash
    report = {'passed': True, 'font_family': 'Times New Roman',
              'font_sha256': sha(font), 'pdf_font_type': 42,
              'figure_width_cm': 16, 'figure_height_cm': 16*7.8/9,
              'png_dpi': 600, 'tiff_dpi': 1200, 'tiff_compression': 'LZW', 'tiff_mode': 'RGB',
              'source_counts_sha256': source_hash, 'source_counts_unchanged': True,
              'class_order': ['retail_banking', 'credit_reporting', 'credit_card',
                              'mortgages_and_loans', 'debt_collection'],
              'normalized_fractions': values, 'displayed_annotations': annotations,
              'files': files, 'model_execution_performed': False}
    (ROOT/'analysis/postprocessing/CONFUSION_FIGURE_VALIDATION.json').write_text(
        json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-directory', type=Path)
    report = render(parser.parse_args().font_directory)
    print(json.dumps({key: report[key] for key in ['font_family', 'figure_width_cm',
                                                 'png_dpi', 'tiff_dpi', 'source_counts_unchanged']}, indent=2))
