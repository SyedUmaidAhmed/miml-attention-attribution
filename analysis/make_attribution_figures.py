"""Generate the two figures of the instance-attribution paper.

Companion to analysis/make_attribution_tables.py; same rule applies — every
coordinate is read from results/instance_level/ and from the witness-structure
computation in make_attribution_tables.regime_table(). Nothing is typed in.
A previous submission was criticised for a figure drawn from placeholder data;
this script exists so that cannot recur.

Outputs (default --outdir paper_attribution/figures/):
    ceiling_vs_achieved.pdf   chance / ceiling / achieved P@1 per benchmark
    ap_vs_attribution.pdf     bag AP against attribution, per benchmark

Usage:
    python3 analysis/make_attribution_figures.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.make_attribution_tables import (      # noqa: E402
    DS_DISPLAY, DS_ORDER, KEY_FOR_DISPLAY, LABEL_AGNOSTIC, ORDER,
    load_cells, regime_table)

# Plain-text method names; the .tex tables carry the \cite{} versions.
PLAIN = {
    'attnmil': 'AB-MIL', 'gated_attn': 'Gated AB-MIL', 'transmil_lite': 'TransMIL',
    'dsmil': 'DSMIL', 'ml_decoder': 'ML-Decoder', 'q2l': 'Q2L', 'hap_only': 'HAP',
}
# Okabe-Ito, colourblind-safe. Label-agnostic methods take the blue-green end,
# label-conditional the warm end, so the two families read apart in greyscale too.
COLOR = {
    'attnmil': '#0072B2', 'gated_attn': '#56B4E9', 'transmil_lite': '#009E73',
    'dsmil': '#D55E00', 'ml_decoder': '#000000', 'q2l': '#CC79A7',
    'hap_only': '#E69F00',
}
DS_PLAIN = {k: v.replace('~', ' ') for k, v in DS_DISPLAY.items()}
MARKER = {
    'attnmil': 's', 'gated_attn': '^', 'transmil_lite': 'D',
    'dsmil': 'o', 'ml_decoder': 'o', 'q2l': 'o', 'hap_only': 'o',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
})


def collect(cells):
    """(dataset -> {method -> dict}) plus the per-dataset chance/ceiling."""
    _, regimes = regime_table()
    geom = {KEY_FOR_DISPLAY[name]: v for name, v in regimes.items()}
    data = {}
    for ds in DS_ORDER:
        methods = {}
        for m in ORDER:
            d = cells.get((m, ds))
            if d is None:
                continue
            im = d['instance_metrics_mean_std']
            methods[m] = {
                'ap': d['bag_metrics_mean_std']['AveragePrecision'][0],
                'p1': im['InstP@1'][0], 'p1_sd': im['InstP@1'][1],
                'auc': im['InstAUC'][0], 'chance': im['ChanceP@1'][0],
            }
        if methods:
            data[ds] = methods
    return data, geom


def legend_handles():
    ag = [Line2D([], [], color=COLOR[m], marker=MARKER[m], linestyle='none',
                 markersize=4.5, markerfacecolor='none', markeredgewidth=1.1,
                 label=PLAIN[m]) for m in ORDER if m in LABEL_AGNOSTIC]
    co = [Line2D([], [], color=COLOR[m], marker=MARKER[m], linestyle='none',
                 markersize=4.5, label=PLAIN[m])
          for m in ORDER if m not in LABEL_AGNOSTIC]
    return ag, co


def fig_ceiling(data, geom, path):
    """One lane per benchmark: chance, label-agnostic ceiling, achieved P@1.

    The point of the figure is the pair of regimes. Where the witness sets
    partition the bag the shaded headroom band has zero width and the three
    label-agnostic markers stack on the chance line; where they do not, the band
    opens and the same three architectures spread out inside it.
    """
    datasets = [d for d in DS_ORDER if d in data]
    fig, ax = plt.subplots(figsize=(6.5, 0.72 * len(datasets) + 1.35))

    lane_h = 1.0
    offsets = {m: (i - 3) * 0.105 for i, m in enumerate(ORDER)}

    for row, ds in enumerate(datasets):
        y = len(datasets) - 1 - row
        g = geom[ds]
        ch, ce = 100 * g['chance'], 100 * g['ceiling']

        ax.axhspan(y - lane_h / 2, y + lane_h / 2,
                   color='#f5f5f5' if row % 2 else 'white', zorder=0)
        # Headroom: everything a label-agnostic model could in principle gain.
        ax.fill_betweenx([y - 0.33, y + 0.33], ch, max(ce, ch + 0.12),
                         color='#009E73', alpha=0.13, zorder=1, linewidth=0)
        ax.plot([ce, ce], [y - 0.36, y + 0.36], color='#009E73',
                linestyle='-', linewidth=1.6, zorder=2)
        # drawn last so it stays legible on the lanes where ceiling == chance
        ax.plot([ch, ch], [y - 0.30, y + 0.30], color='#111111',
                linestyle='--', linewidth=0.9, zorder=5)

        for m, v in data[ds].items():
            agn = m in LABEL_AGNOSTIC
            ax.errorbar(100 * v['p1'], y + offsets[m], xerr=100 * v['p1_sd'],
                        color=COLOR[m], marker=MARKER[m], markersize=4.5,
                        markerfacecolor='none' if agn else COLOR[m],
                        markeredgewidth=1.1, elinewidth=0.7, capsize=1.6,
                        linestyle='none', zorder=4)

        head = 100 * g['headroom']
        ax.text(101.5, y, f'headroom {head:+.1f} pts', va='center', ha='left',
                fontsize=6.6, color='#555555')

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DS_PLAIN[d] for d in reversed(datasets)], fontsize=7.5)
    ax.set_ylim(-0.55, len(datasets) - 0.45)
    ax.set_xlim(0, 100)
    ax.set_xlabel('instance-attribution precision@1 (%)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    ag, co = legend_handles()
    ref = [Line2D([], [], color='#111111', linestyle='--', linewidth=0.9,
                  label='chance'),
           Line2D([], [], color='#009E73', linewidth=1.6, label='ceiling $\\Gamma$'),
           Patch(facecolor='#009E73', alpha=0.13, label='headroom')]
    fig.legend(handles=ag + co + ref, loc='lower center', ncol=5,
               frameon=False, fontsize=7, handletextpad=0.5,
               columnspacing=1.4, bbox_to_anchor=(0.5, -0.015))

    fig.subplots_adjust(left=0.17, right=0.83, top=0.97, bottom=0.14)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_slopes(data, geom, path):
    """Bag AP beside attribution P@1, one slopegraph per benchmark.

    Crossing lines are rank inversions between the two axes; the ratio of the
    two spans printed under each panel is the magnification quoted in the text.
    """
    datasets = [d for d in DS_ORDER if d in data]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.9, 2.75))
    if len(datasets) == 1:
        axes = [axes]

    stats = {}
    for ax, ds in zip(axes, datasets):
        ms = list(data[ds])
        ap = {m: data[ds][m]['ap'] for m in ms}
        p1 = {m: data[ds][m]['p1'] for m in ms}
        # Each axis is scaled to its own observed range, so the figure compares
        # orderings and relative position; the spans are printed as numbers.
        def norm(v, lo, hi):
            return 0.5 if hi - lo < 1e-12 else (v - lo) / (hi - lo)

        alo, ahi = min(ap.values()), max(ap.values())
        plo, phi = min(p1.values()), max(p1.values())
        cond = [m for m in ms if m not in LABEL_AGNOSTIC]
        span_ap = 100 * (max(ap[m] for m in cond) - min(ap[m] for m in cond))
        span_p1 = 100 * (max(p1[m] for m in cond) - min(p1[m] for m in cond))
        stats[ds] = (span_ap, span_p1, span_p1 / span_ap if span_ap else None)

        for m in ms:
            agn = m in LABEL_AGNOSTIC
            ya, yp = norm(ap[m], alo, ahi), norm(p1[m], plo, phi)
            ax.plot([0, 1], [ya, yp], color=COLOR[m], linewidth=1.0,
                    linestyle=':' if agn else '-',
                    alpha=0.55 if agn else 0.95, zorder=2)
            for x, yy in ((0, ya), (1, yp)):
                ax.plot([x], [yy], color=COLOR[m], marker=MARKER[m],
                        markersize=4.0, markeredgewidth=1.0,
                        markerfacecolor='none' if agn else COLOR[m], zorder=3)

        ax.set_title(DS_PLAIN[ds], fontsize=7.5, pad=22)
        ax.text(0, 1.115, 'bag AP', transform=ax.transAxes, ha='left',
                fontsize=6.6, color='#555555')
        ax.text(1, 1.115, 'P@1', transform=ax.transAxes, ha='right',
                fontsize=6.6, color='#555555')
        ax.text(0, -0.055, f'{alo:.3f}', transform=ax.transAxes, ha='left',
                va='top', fontsize=6.2, color='#777777')
        ax.text(0, 1.0, f'{ahi:.3f}', transform=ax.transAxes, ha='left',
                va='bottom', fontsize=6.2, color='#777777')
        ax.text(1, -0.055, f'{100*plo:.1f}', transform=ax.transAxes, ha='right',
                va='top', fontsize=6.2, color='#777777')
        ax.text(1, 1.0, f'{100*phi:.1f}', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=6.2, color='#777777')
        mag = stats[ds][2]
        ax.text(0.5, -0.17, '%.1f $\\rightarrow$ %.1f pts%s'
                % (span_ap, span_p1,
                   '' if mag is None else '  ($\\times$%.1f)' % mag),
                transform=ax.transAxes, ha='center', fontsize=6.4,
                color='#333333')

        ax.set_xlim(-0.22, 1.22)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ('top', 'right', 'bottom'):
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axvline(0, color='#bbbbbb', linewidth=0.6, zorder=1)
        ax.axvline(1, color='#bbbbbb', linewidth=0.6, zorder=1)

    ag, co = legend_handles()
    fig.legend(handles=ag + co, loc='lower center', ncol=7, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.24, wspace=0.45)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='paper_attribution/figures')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cells = load_cells()
    if not cells:
        sys.exit('no result cells found in results/instance_level')
    data, geom = collect(cells)

    f1 = os.path.join(args.outdir, 'ceiling_vs_achieved.pdf')
    fig_ceiling(data, geom, f1)
    f2 = os.path.join(args.outdir, 'ap_vs_attribution.pdf')
    stats = fig_slopes(data, geom, f2)

    print(f'wrote {f1}')
    print(f'wrote {f2}')
    for ds, (a, b, mag) in stats.items():
        print(f'  {ds:17s} label-conditional spread: bag AP {a:5.1f} pts -> '
              f'P@1 {b:5.1f} pts  (x{mag:.1f})')


if __name__ == '__main__':
    main()
