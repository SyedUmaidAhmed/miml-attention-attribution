"""Generate the Figure-1 overview / graphical abstract of the attribution paper.

Companion to analysis/make_attribution_figures.py; the same rule applies —
every empirical coordinate is read from results/instance_level/*.json and from
the regime summary written by analysis/make_attribution_tables.py
(paper_attribution/tables/attribution_summary.json). Panels (a)-(c) are
mechanism schematics (bag, architectures, theorem) and contain no data; every
number in panel (d) and in the annotations is loaded, and the headline
annotations ("315 models at chance", "+37.5", "40.4 -> 93.2") are re-verified
against the JSONs at run time and the script aborts if they no longer hold.

Outputs (default --outdir paper_attribution/figures/):
    overview.pdf            full-width Figure 1 (four panels, ~2.4:1)
    graphical_abstract.pdf  Elsevier graphical abstract (exactly 13 cm x 5 cm)

Usage:
    python3 analysis/make_overview_figure.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results', 'instance_level')
SUMMARY = os.path.join(REPO, 'paper_attribution', 'tables',
                       'attribution_summary.json')

# Same families / palette / markers as make_attribution_figures.py.
LABEL_AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}
ORDER = ['attnmil', 'gated_attn', 'transmil_lite',
         'dsmil', 'ml_decoder', 'q2l', 'hap_only']
PLAIN = {
    'attnmil': 'AB-MIL', 'gated_attn': 'Gated AB-MIL', 'transmil_lite': 'TransMIL',
    'dsmil': 'DSMIL', 'ml_decoder': 'ML-Decoder', 'q2l': 'Q2L', 'hap_only': 'HAP',
}
COLOR = {
    'attnmil': '#0072B2', 'gated_attn': '#56B4E9', 'transmil_lite': '#009E73',
    'dsmil': '#D55E00', 'ml_decoder': '#000000', 'q2l': '#CC79A7',
    'hap_only': '#E69F00',
}
MARKER = {
    'attnmil': 's', 'gated_attn': '^', 'transmil_lite': 'D',
    'dsmil': 'v', 'ml_decoder': 'o', 'q2l': 'P', 'hap_only': '*',
}
DS_ORDER = ['letter_frost', 'letter_carroll', 'mscv2', 'hja_birdsong',
            'voc2012seg', 'voc2012segml']
DS_PLAIN = {'letter_frost': 'Letter Frost', 'letter_carroll': 'Letter Carroll',
            'mscv2': 'MSRC v2', 'hja_birdsong': 'Bird Song',
            'voc2012seg': 'VOC-Patch-Seg',
            'voc2012segml': 'VOC-Patch-Seg-ML'}
KEY_FOR_DISPLAY = {'MSRC~v2': 'mscv2', 'Letter Frost': 'letter_frost',
                   'Letter Carroll': 'letter_carroll',
                   'Bird Song': 'hja_birdsong',
                   'VOC-Patch-Seg': 'voc2012seg',
                   'VOC-Patch-Seg-ML': 'voc2012segml'}
PARTITIONED = ['letter_frost', 'letter_carroll', 'mscv2']
VOC_SEG = 'voc2012seg'


def use_r1():
    """DMKD revision: r1 results, 8 architectures, no-crop VOC."""
    global RESULTS, ORDER, DS_ORDER, DS_PLAIN, KEY_FOR_DISPLAY, VOC_SEG
    RESULTS = os.path.join(REPO, 'results', 'instance_level_r1')
    ORDER = ['attnmil', 'gated_attn', 'transmil_lite', 'dsmil_orig', 'dsmil',
             'ml_decoder', 'q2l', 'hap_only']
    PLAIN.update(dsmil_orig='DSMIL', dsmil='DSMIL-sep')
    COLOR.update(dsmil_orig='#D55E00', dsmil='#882255')
    MARKER.update(dsmil_orig='v', dsmil='X')
    DS_ORDER = ['letter_frost', 'letter_carroll', 'mscv2', 'hja_birdsong',
                'voc2012seg_nc', 'voc2012segml_nc']
    DS_PLAIN.update(voc2012seg_nc='VOC-Patch-Seg', voc2012segml_nc='VOC-Patch-Seg-ML')
    KEY_FOR_DISPLAY = dict(KEY_FOR_DISPLAY, **{'VOC-Patch-Seg': 'voc2012seg_nc',
                                               'VOC-Patch-Seg-ML': 'voc2012segml_nc'})
    VOC_SEG = 'voc2012seg_nc'

GREEN = '#009E73'   # ceiling / headroom, as in ceiling_vs_achieved.pdf
INK = '#111111'
GRAY = '#555555'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
})


# ---------------------------------------------------------------- data loading

def load():
    cells = {}
    for f in glob.glob(os.path.join(RESULTS, '*__*.json')):
        d = json.load(open(f))
        cells[(d['method'], d['dataset'])] = d
    if not cells:
        sys.exit('no result cells found in ' + RESULTS)
    if not os.path.exists(SUMMARY):
        sys.exit('missing ' + SUMMARY +
                 ' — run analysis/make_attribution_tables.py first')
    summary = json.load(open(SUMMARY))
    geom = {KEY_FOR_DISPLAY[k]: v for k, v in summary['regimes'].items()}

    data = {}
    for ds in DS_ORDER:
        methods = {}
        for m in ORDER:
            d = cells.get((m, ds))
            if d is None:
                continue
            im = d['instance_metrics_mean_std']
            methods[m] = {'p1': im['InstP@1'][0], 'p1_sd': im['InstP@1'][1],
                          'chance': im['ChanceP@1'][0],
                          'n_models': len(d.get('instance_per_model', []))}
        if methods:
            data[ds] = methods

    # Re-verify the hard-coded annotations before printing them on a figure.
    n_agn = sum(data[ds][m]['n_models'] for ds in PARTITIONED
                for m in LABEL_AGNOSTIC if m in data[ds])
    max_dev = max(abs(data[ds][m]['p1'] - data[ds][m]['chance'])
                  for ds in PARTITIONED for m in LABEL_AGNOSTIC
                  if m in data[ds])
    best_delta = max(100 * (data[ds][m]['p1'] - data[ds][m]['chance'])
                     for ds in PARTITIONED for m in data[ds]
                     if m not in LABEL_AGNOSTIC)
    voc = geom[VOC_SEG]
    if (n_agn != 315 or max_dev > 6e-17 or abs(best_delta - 37.5) > 0.05
            or abs(100 * voc['chance'] - 40.4) > 0.05
            or abs(100 * voc['ceiling'] - 93.2) > 0.05):
        sys.exit(f'annotation check failed: n_agn={n_agn}, '
                 f'max_dev={max_dev:.2e}, best_delta={best_delta:.2f}, '
                 f'voc chance/ceiling={100*voc["chance"]:.1f}/'
                 f'{100*voc["ceiling"]:.1f} — '
                 'update the hard-coded annotations to match the data')
    return data, geom


# ------------------------------------------------------------ drawing helpers

def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_tag(ax, tag, title, fs=8.0):
    ax.text(0.0, 1.045, '$\\mathbf{(%s)}$ %s' % (tag, title),
            transform=ax.transAxes, fontsize=fs, va='bottom', ha='left')


def attn_bars(ax, x0, y0, w, h, vals, hot, bar='#b5b5b5'):
    """A tiny attention histogram; the top-1 bar is drawn in ink."""
    n = len(vals)
    bw = w / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + i * bw + 0.14 * bw, y0),
                               0.72 * bw, h * v, zorder=3,
                               facecolor=INK if i == hot else bar,
                               edgecolor='none'))
    ax.plot([x0, x0 + w], [y0, y0], color='#333333', lw=0.6, zorder=3)


# ---------------------------------------------------------------- panel (a)

def panel_bag(ax):
    blank(ax)
    panel_tag(ax, 'a', 'a MIML bag')
    ax.add_patch(FancyBboxPatch((0.05, 0.20), 0.90, 0.60,
                                boxstyle='round,pad=0.02,rounding_size=0.05',
                                facecolor='#fafafa', edgecolor='#888888',
                                linewidth=0.8))
    ax.text(0.09, 0.845, 'bag $B_i$,  labels $Y_i=\\{l_1,l_2\\}$',
            fontsize=7.8, ha='left', va='center')

    groups = [   # (positions, facecolor, edgecolor, label under the group)
        ([(0.185, 0.62), (0.255, 0.37)], '#404040', '#404040',
         '$W_i(l_1)$', INK),
        ([(0.45, 0.62), (0.52, 0.37)], '#b0b0b0', '#707070',
         '$W_i(l_2)$', INK),
        ([(0.80, 0.62), (0.87, 0.37)], 'white', '#909090',
         'no label', GRAY),
    ]
    for pts, fc, ec, lab, tc in groups:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=100, facecolor=fc, edgecolor=ec,
                   linewidth=0.9, zorder=4)
        ax.text(sum(xs) / 2, 0.095, lab, fontsize=7.0, ha='center',
                va='center', color=tc)
    ax.text(0.5, -0.075, 'which instance carries which label?',
            fontsize=7.4, ha='center', va='center', color=GRAY,
            style='italic')


# ---------------------------------------------------------------- panel (b)

def panel_arch(ax):
    blank(ax)
    panel_tag(ax, 'b', 'two attention designs')

    va = [0.28, 1.0, 0.22, 0.38, 0.18]          # shared map (schematic)
    v1 = [0.30, 1.0, 0.20, 0.25, 0.15]          # per-label maps (schematic)
    v2 = [0.18, 0.25, 0.22, 1.0, 0.35]

    # -- label-agnostic (top half)
    ax.text(0.02, 0.955, 'label-agnostic', fontsize=8.0, fontweight='bold',
            va='center')
    ax.text(0.02, 0.885, 'AB-MIL, Gated, TransMIL', fontsize=6.9,
            color=GRAY, va='center')
    attn_bars(ax, 0.06, 0.64, 0.32, 0.17, va, hot=1)
    ax.text(0.22, 0.578, '$a_i(\\cdot)$ — one map', fontsize=7.4,
            ha='center', va='center')
    for ytxt in (0.82, 0.72, 0.62):
        ax.add_patch(FancyArrowPatch((0.42, 0.71), (0.60, ytxt),
                                     arrowstyle='-|>', mutation_scale=6,
                                     color='#777777', lw=0.7,
                                     shrinkA=0, shrinkB=1))
    ax.text(0.635, 0.82, 'label $l_1$', fontsize=7.4, va='center')
    ax.text(0.635, 0.72, 'label $l_2$', fontsize=7.4, va='center')
    ax.text(0.635, 0.62, '$\\cdots$ label $l_L$', fontsize=7.4, va='center')
    ax.text(0.985, 0.528, 'reused for every label', fontsize=7.0,
            color=GRAY, va='center', ha='right', style='italic')

    ax.plot([0.02, 0.98], [0.485, 0.485], color='#cccccc', lw=0.6)

    # -- label-conditional (bottom half)
    ax.text(0.02, 0.425, 'label-conditional', fontsize=8.0, fontweight='bold',
            va='center')
    ax.text(0.02, 0.355, 'DSMIL, ML-Decoder, Q2L, HAP', fontsize=7.1,
            color=GRAY, va='center')
    attn_bars(ax, 0.06, 0.12, 0.32, 0.16, v1, hot=1)
    attn_bars(ax, 0.56, 0.12, 0.32, 0.16, v2, hot=3)
    ax.text(0.22, 0.048, '$a_i(l_1,\\cdot)$', fontsize=7.4, ha='center',
            va='center')
    ax.text(0.72, 0.048, '$a_i(l_2,\\cdot)$', fontsize=7.4, ha='center',
            va='center')
    ax.text(0.5, -0.075, 'one attention map per label', fontsize=7.4,
            ha='center', va='center', color=GRAY, style='italic')


# ---------------------------------------------------------------- panel (c)

def gauge(ax, x, chance, ceiling, y0=0.14, y1=0.60, half=0.075):
    """Vertical chance/ceiling gauge in axis coordinates (values in [0,1])."""
    def yy(v):
        return y0 + v * (y1 - y0)
    ax.plot([x, x], [y0, y1], color='#bbbbbb', lw=0.7, zorder=1)
    if ceiling - chance > 1e-9:
        ax.add_patch(Rectangle((x - half + 0.02, yy(chance)),
                               2 * (half - 0.02), yy(ceiling) - yy(chance),
                               facecolor=GREEN, alpha=0.13,
                               edgecolor='none', zorder=2))
    ax.plot([x - half, x + half], [yy(ceiling)] * 2, color=GREEN, lw=1.6,
            zorder=4)
    ax.plot([x - half + 0.015, x + half - 0.015], [yy(chance)] * 2,
            color=INK, lw=0.9, linestyle='--', zorder=5)
    return yy


def panel_theory(ax):
    blank(ax)
    panel_tag(ax, 'c', 'Theorem: a ceiling $\\Gamma$')
    ax.text(0.5, 0.965,
            '$\\Gamma=\\sum_i \\max_j \\mathrm{cov}_i(j)\\,/\\,\\sum_i m_i$',
            fontsize=7.9, ha='center', va='center')
    ax.text(0.42, 0.815, 'attribution of any label-\n'
            'agnostic model $\\leq \\Gamma$ — set by\n'
            'label geometry, not training',
            fontsize=6.9, ha='center', va='center', color=GRAY,
            style='italic')

    # left gauge: partition -> collapse
    yy = gauge(ax, 0.21, 0.5, 0.5)
    ax.text(0.21, yy(0.5) + 0.075, '$\\Gamma$ = chance', fontsize=7.5,
            ha='center', va='center')
    ax.text(0.21, 0.028, '$W_i(l)$ partition\n$\\Rightarrow$ collapse\nto chance',
            fontsize=6.9, ha='center', va='top')

    # right gauge: overlap -> headroom
    yy = gauge(ax, 0.78, 0.10, 0.92)
    ax.text(0.78, yy(0.92) + 0.062, '$\\Gamma$', fontsize=7.6, ha='center',
            va='center', color=GREEN)
    ax.text(0.685, yy(0.10), 'chance', fontsize=7.2, ha='right',
            va='center')
    ax.text(0.78, yy(0.51), 'headroom', fontsize=6.9, ha='center',
            va='center', color=GRAY, rotation=90)
    ax.text(0.78, 0.028, 'no partition\n$\\Rightarrow$ band\nopens',
            fontsize=6.9, ha='center', va='top')


# ---------------------------------------------------------------- panel (d)

def panel_results(ax, data, geom):
    datasets = [d for d in DS_ORDER if data.get(d)]
    n_total = sum(v['n_models'] for d in datasets for v in data[d].values())
    panel_tag(ax, 'd', f'{n_total} trained models (P@1, %)')
    ypos = {d: len(datasets) - 1 - i for i, d in enumerate(datasets)}
    sp = 0.105 if len(ORDER) == 7 else 0.09
    offsets = {m: (i - (len(ORDER) - 1) / 2) * sp for i, m in enumerate(ORDER)}

    for row, ds in enumerate(datasets):
        y = len(datasets) - 1 - row
        g = geom[ds]
        ch, ce = 100 * g['chance'], 100 * g['ceiling']
        ax.axhspan(y - 0.5, y + 0.5, color='#f5f5f5' if row % 2 else 'white',
                   zorder=0)
        ax.fill_betweenx([y - 0.34, y + 0.34], ch, max(ce, ch + 0.12),
                         color=GREEN, alpha=0.13, zorder=1, linewidth=0)
        ax.plot([ce, ce], [y - 0.37, y + 0.37], color=GREEN, lw=1.6, zorder=2)
        ax.plot([ch, ch], [y - 0.31, y + 0.31], color=INK, linestyle='--',
                lw=0.9, zorder=5)
        for m, v in data[ds].items():
            agn = m in LABEL_AGNOSTIC
            ax.plot([100 * v['p1']], [y + offsets[m]], color=COLOR[m],
                    marker=MARKER[m], markersize=3.9,
                    markerfacecolor='none' if agn else COLOR[m],
                    markeredgewidth=1.0, linestyle='none', zorder=4)

    # Regime annotations; the numbers are re-verified in load().
    kw = dict(fontsize=7.9, ha='right', va='center', color='#333333')
    yf = ypos['letter_frost']
    ax.text(99.5, yf + 0.17, 'partition $\\Rightarrow$', **kw)
    ax.text(99.5, yf - 0.16, 'all 315 label-agnostic', **kw)
    ax.text(99.5, yf - 0.52,
            'models exactly at chance', **kw)
    ax.text(99.5, yf - 1.50, 'label-conditional up to $+37.5$', **kw)
    nxt = [d for d in datasets if d not in PARTITIONED]
    if nxt:
        ax.text(14.0, ypos[nxt[0]] + 0.5, 'no partition: band opens',
                fontsize=7.9, ha='left', va='center', color='#333333')

    two_line = {'letter_frost': 'Letter\nFrost',
                'letter_carroll': 'Letter\nCarroll', 'mscv2': 'MSRC v2',
                'hja_birdsong': 'Bird\nSong',
                'voc2012seg': 'VOC-Patch\nSeg',
                'voc2012segml': 'VOC-Patch\nSeg-ML',
                'voc2012seg_nc': 'VOC-Patch\nSeg',
                'voc2012segml_nc': 'VOC-Patch\nSeg-ML'}
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([two_line[d] for d in reversed(datasets)],
                       fontsize=7.4, linespacing=1.15)
    ax.set_ylim(-0.55, len(datasets) - 0.45)
    ax.set_xlim(13, 100)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=7.2, length=2.2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)


def legend_rows():
    ag = [Line2D([], [], color=COLOR[m], marker=MARKER[m], linestyle='none',
                 markersize=4.2, markerfacecolor='none', markeredgewidth=1.0,
                 label=PLAIN[m] + ' (agnostic)')
          for m in ORDER if m in LABEL_AGNOSTIC]
    co = [Line2D([], [], color=COLOR[m], marker=MARKER[m], linestyle='none',
                 markersize=4.2, label=PLAIN[m])
          for m in ORDER if m not in LABEL_AGNOSTIC]
    ref = [Line2D([], [], color=INK, linestyle='--', lw=0.9, label='chance'),
           Line2D([], [], color=GREEN, lw=1.6, label='ceiling $\\Gamma$'),
           Patch(facecolor=GREEN, alpha=0.13, label='headroom')]
    return ag + co + ref


# ------------------------------------------------------------------ figure 1

def fig_overview(data, geom, path):
    # Two rows at the sn-jnl text width (~5.1 in), so the figure is placed
    # near 1:1 and its lettering stays at about 7 pt in print.
    fig = plt.figure(figsize=(5.4, 5.3))
    top = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.22, 1.02],
                           left=0.02, right=0.99, top=0.93, bottom=0.58,
                           wspace=0.30)
    bot = fig.add_gridspec(1, 1, left=0.14, right=0.97, top=0.48,
                           bottom=0.13)
    panel_bag(fig.add_subplot(top[0]))
    panel_arch(fig.add_subplot(top[1]))
    panel_theory(fig.add_subplot(top[2]))
    panel_results(fig.add_subplot(bot[0]), data, geom)

    fig.legend(handles=legend_rows(), loc='lower center', ncol=4,
               frameon=False, fontsize=8.0, handletextpad=0.45,
               columnspacing=1.1, bbox_to_anchor=(0.5, -0.005))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------- graphical abstract

def fig_abstract(data, geom, path):
    """Exactly 13 cm x 5 cm (saved without bbox cropping); text >= ~7 pt."""
    fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
    fig.text(0.5, 0.975, 'Does attention know which instance '
             'carries which label?',
             fontsize=8.4, fontweight='bold', ha='center', va='top')

    # Left: the two designs, family-level, plus the theorem in one sentence.
    axl = fig.add_axes([0.03, 0.30, 0.40, 0.50])
    blank(axl)
    va = [0.28, 1.0, 0.22, 0.38, 0.18]
    v2 = [0.18, 0.25, 0.22, 1.0, 0.35]
    axl.text(0.0, 1.06, 'label-agnostic: one map for all labels',
             fontsize=7.6, va='center')
    attn_bars(axl, 0.02, 0.68, 0.26, 0.28, va, hot=1)
    axl.text(0.32, 0.775, '$\\rightarrow\\ l_1, l_2, \\ldots, l_L$',
             fontsize=7.4, va='center')
    axl.text(0.0, 0.475, 'label-conditional: one map per label',
             fontsize=7.6, va='center')
    attn_bars(axl, 0.02, 0.09, 0.26, 0.26, va, hot=1)
    attn_bars(axl, 0.36, 0.09, 0.26, 0.26, v2, hot=3)
    axl.text(0.66, 0.19, '$a(l,\\cdot)$', fontsize=7.4, va='center')
    axl.text(0.0, -0.30, 'Theorem: label-agnostic attribution '
             '$\\leq \\Gamma$,\nfixed by label geometry; witnesses\n'
             'partition each bag $\\Rightarrow \\Gamma$ = chance.',
             fontsize=7.5, va='center', ha='left', linespacing=1.35)

    # Right: one partitioned + one overlapping benchmark, real numbers.
    axr = fig.add_axes([0.50, 0.30, 0.475, 0.50])
    ds_show = ['mscv2', VOC_SEG]
    for row, ds in enumerate(ds_show):
        y = len(ds_show) - 1 - row
        g = geom[ds]
        ch, ce = 100 * g['chance'], 100 * g['ceiling']
        axr.fill_betweenx([y - 0.24, y + 0.24], ch, max(ce, ch + 0.15),
                          color=GREEN, alpha=0.13, zorder=1, linewidth=0)
        axr.plot([ce, ce], [y - 0.28, y + 0.28], color=GREEN, lw=1.7,
                 zorder=2)
        axr.plot([ch, ch], [y - 0.24, y + 0.24], color=INK, linestyle='--',
                 lw=1.0, zorder=5)
        for i, m in enumerate(ORDER):
            if m not in data[ds]:
                continue
            agn = m in LABEL_AGNOSTIC
            axr.plot([100 * data[ds][m]['p1']], [y + (i - (len(ORDER) - 1) / 2) * 0.068],
                     marker='o', markersize=3.4,
                     color=GRAY if agn else '#D55E00',
                     markerfacecolor='none' if agn else '#D55E00',
                     markeredgewidth=1.0, linestyle='none', zorder=4)
    axr.text(21, 1.60, 'partition (MSRC v2): label-agnostic = chance',
             fontsize=7.5, ha='left', va='center')
    axr.text(22, -0.62,
             'overlap (VOC-Patch): band opens 40.4 $\\rightarrow$ 93.2',
             fontsize=7.5, ha='left', va='center')
    axr.set_ylim(-1.0, 1.95)
    axr.set_xlim(20, 100)
    axr.set_xticks([20, 40, 60, 80, 100])
    axr.tick_params(axis='x', labelsize=7.5, length=2.0, pad=1.2)
    axr.set_yticks([])
    axr.spines[['top', 'right', 'left']].set_visible(False)
    axr.set_xlabel('instance-attribution P@1 (%)', fontsize=7.5, labelpad=1.2)

    handles = [
        Line2D([], [], color=GRAY, marker='o', linestyle='none',
               markersize=3.6, markerfacecolor='none', markeredgewidth=1.0,
               label='label-agnostic'),
        Line2D([], [], color='#D55E00', marker='o', linestyle='none',
               markersize=3.6, label='label-conditional'),
        Line2D([], [], color=INK, linestyle='--', lw=1.0, label='chance'),
        Line2D([], [], color=GREEN, lw=1.7, label='ceiling $\\Gamma$')]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.995, -0.015), ncol=2, frameon=False,
               fontsize=7.0, handletextpad=0.3, columnspacing=0.7,
               handlelength=1.3, labelspacing=0.25)
    fig.savefig(path)     # no bbox cropping: keep the exact 13 x 5 cm canvas
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir',
                   default=os.path.join(REPO, 'paper_attribution', 'figures'))
    p.add_argument('--rev', default='v1', choices=['v1', 'r1'])
    args = p.parse_args()
    if args.rev == 'r1':
        use_r1()
    os.makedirs(args.outdir, exist_ok=True)

    data, geom = load()
    f1 = os.path.join(args.outdir, 'overview.pdf')
    fig_overview(data, geom, f1)
    print('wrote', f1)
    f2 = os.path.join(args.outdir, 'graphical_abstract.pdf')
    fig_abstract(data, geom, f2)
    print('wrote', f2)


if __name__ == '__main__':
    main()
