"""Write VERIFICATION_REPORT.md: a data-backed reproducibility report.

Produced for the Neurocomputing resubmission (2026-09-05) so that the claims
of the manuscript can be audited against the released code and stored results
without reading the LaTeX. Everything in the report is read from
results/instance_level/*.json, the benchmark files, the environment, and the
live output of analysis/verify_paper_claims.py; nothing is typed in.

Usage:
    python3 analysis/make_verification_report.py [--out VERIFICATION_REPORT.md]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

RESULTS = 'results/instance_level'
ORDER = ['attnmil', 'gated_attn', 'transmil_lite',
         'dsmil', 'ml_decoder', 'q2l', 'hap_only']
DISPLAY = {'attnmil': 'AB-MIL', 'gated_attn': 'Gated AB-MIL',
           'transmil_lite': 'TransMIL', 'dsmil': 'DSMIL',
           'ml_decoder': 'ML-Decoder', 'q2l': 'Q2L', 'hap_only': 'HAP'}
DS_ORDER = ['letter_frost', 'letter_carroll', 'mscv2', 'voc2012seg',
            'voc2012segml']
DS_DISPLAY = {'letter_frost': 'Letter Frost', 'letter_carroll': 'Letter Carroll',
              'mscv2': 'MSRC v2', 'voc2012seg': 'VOC-Patch-Seg',
              'voc2012segml': 'VOC-Patch-Seg-ML'}
AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}


def sha256(path, n=12):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def env_block():
    lines = [f'- Python {platform.python_version()} on {platform.platform()}']
    for mod in ('torch', 'numpy', 'scipy', 'sklearn', 'matplotlib'):
        try:
            m = __import__(mod)
            lines.append(f'- {mod} {getattr(m, "__version__", "?")}')
        except Exception:                                   # pragma: no cover
            lines.append(f'- {mod} (not importable)')
    try:
        import torch
        dev = ('mps' if torch.backends.mps.is_available()
               else 'cuda' if torch.cuda.is_available() else 'cpu')
        lines.append(f'- default device: {dev}')
    except Exception:                                       # pragma: no cover
        pass
    try:
        rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      text=True).strip()
        lines.append(f'- git revision: `{rev}`')
    except Exception:
        pass
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='VERIFICATION_REPORT.md')
    args = ap.parse_args()

    cells = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, '*__*.json'))):
        d = json.load(open(f))
        d['_file'] = f
        cells[(d['method'], d['dataset'])] = d

    # live verifier output
    ver = subprocess.run([sys.executable, 'analysis/verify_paper_claims.py'],
                         capture_output=True, text=True)
    ver_lines = [l for l in ver.stdout.splitlines()
                 if l.strip().startswith(('PASS', 'FAIL')) or 'checks' in l]
    n_pass = sum(1 for l in ver_lines if l.strip().startswith('PASS'))
    n_fail = sum(1 for l in ver_lines if l.strip().startswith('FAIL'))

    out = []
    out.append('# Verification report')
    out.append('')
    out.append(f'Generated {_dt.datetime.now().strftime("%Y-%m-%d %H:%M")} by '
               '`analysis/make_verification_report.py`. Every number below is '
               'read from the stored result files or produced by running the '
               'checks; none is typed in.')
    out.append('')
    out.append('## 1. Environment')
    out.append('')
    out.append(env_block())
    out.append('')
    out.append('## 2. Claim verifier')
    out.append('')
    out.append(f'`python3 analysis/verify_paper_claims.py` — **{n_pass} passed, '
               f'{n_fail} failed** (exit code {ver.returncode}).')
    out.append('')
    out.append('```')
    out.extend(ver_lines)
    out.append('```')
    out.append('')
    out.append('## 3. Result cells')
    out.append('')
    out.append('One JSON per (architecture, benchmark). `models` is the number '
               'of independently trained networks whose attribution was '
               'scored; `P@1`, `chance`, `AUC` are means over them; `AP` is the '
               'bag-level average precision of the cell (ensemble on the '
               'classical benchmarks, single model per fold on VOC-Patch). '
               '`sha256` is the first 12 hex digits of the file digest.')
    out.append('')
    out.append('| Architecture | Benchmark | folds | models | AP | P@1 | chance | Δ | AUC | max \\|P@1−chance\\| per model | wall min | sha256 |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for ds in DS_ORDER:
        for m in ORDER:
            d = cells.get((m, ds))
            if d is None:
                continue
            im = d['instance_metrics_mean_std']
            ap = d['bag_metrics_mean_std']['AveragePrecision'][0]
            p1, ch, auc = im['InstP@1'][0], im['ChanceP@1'][0], im['InstAUC'][0]
            dev = max(abs(r['InstP@1'] - r['ChanceP@1'])
                      for r in d['instance_per_model'])
            out.append(f'| {DISPLAY[m]} | {DS_DISPLAY[ds]} | {d["num_folds"]} | '
                       f'{len(d["instance_per_model"])} | {ap:.3f} | {100*p1:.1f} | '
                       f'{100*ch:.1f} | {100*(p1-ch):+.1f} | {auc:.3f} | '
                       f'{dev:.1e} | {d["wall_clock_seconds"]/60:.0f} | '
                       f'`{sha256(d["_file"])}` |')
    out.append('')
    n_models = sum(len(d['instance_per_model']) for d in cells.values())
    n_ag = sum(len(d['instance_per_model']) for (m, ds), d in cells.items()
               if m in AGNOSTIC and not ds.startswith('voc'))
    out.append(f'Total trained models scored: **{n_models}** '
               f'({n_ag} label-agnostic on the three partitioned benchmarks).')
    out.append('')
    out.append('## 4. Theorem check, model by model')
    out.append('')
    out.append('Corollary 1 predicts that every label-agnostic model on a '
               'partitioned benchmark scores exactly at its fold\'s chance rate. '
               'The table gives, per cell, the largest deviation over all '
               'trained models; the theorem is confirmed if every entry is at '
               'floating-point noise.')
    out.append('')
    out.append('| Architecture | Letter Frost | Letter Carroll | MSRC v2 |')
    out.append('|---|---|---|---|')
    for m in sorted(AGNOSTIC, key=ORDER.index):
        row = [DISPLAY[m]]
        for ds in DS_ORDER[:3]:
            d = cells[(m, ds)]
            row.append(f'{max(abs(r["InstP@1"] - r["ChanceP@1"]) for r in d["instance_per_model"]):.2e}')
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')
    out.append('## 5. Benchmark files')
    out.append('')
    from run_all_enhanced import DATA_DIR
    out.append(f'Data directory: `{DATA_DIR}`')
    out.append('')
    out.append('| File | bytes | sha256 |')
    out.append('|---|---|---|')
    for name in ['MSCV2.mat', 'letter_frost.mat', 'letter_carroll.mat',
                 'voc2012seg.mat', 'voc2012seg_10CV.mat', 'voc2012seg_patch_gt.npz',
                 'voc2012segml.mat', 'voc2012segml_10CV.mat',
                 'voc2012segml_patch_gt.npz', 'voc2012_patch_gt.npz']:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            out.append(f'| `{name}` | {os.path.getsize(p):,} | `{sha256(p)}` |')
        else:
            out.append(f'| `{name}` | (absent) | — |')
    try:
        z = np.load(os.path.join(DATA_DIR, 'voc2012_patch_gt.npz'), allow_pickle=True)
        out.append('')
        out.append(f'Patch-label overlap threshold stored in `voc2012_patch_gt.npz`: '
                   f'**{float(z["threshold"])}**.')
    except Exception:
        pass
    out.append('')
    out.append('## 6. Configuration uniformity')
    out.append('')
    keys = ['embed_dim', 'attn_dim', 'nhead', 'ff_dim', 'transformer_layers',
            'hap_topk', 'dropout', 'batch_size', 'weight_decay', 'gradient_clip',
            'label_smoothing', 'phase1_epochs', 'phase1_lr', 'phase2_max_epochs',
            'phase2_lr', 'early_stop_patience', 'swa_start', 'bag_weight']
    ref = {'classical': cells[('hap_only', 'mscv2')]['config'],
           'voc': cells[('hap_only', 'voc2012seg')]['config']}
    exceptions = []
    for (m, ds), d in cells.items():
        r = ref['voc' if ds.startswith('voc') else 'classical']
        for k in keys:
            if d['config'].get(k) != r.get(k):
                exceptions.append(f'- `{k}` = {d["config"].get(k)} for {DISPLAY[m]} on '
                                  f'{DS_DISPLAY[ds]} (reference {r.get(k)})')
    out.append(f'All {len(cells)} cells were compared on {len(keys)} training '
               'and architecture settings against the reference configuration '
               'of their column (Table 6 of the manuscript).')
    out.append('')
    out.append('\n'.join(exceptions) if exceptions else '- no exceptions')
    out.append('')
    out.append('## 7. Attention read-out audit')
    out.append('')
    try:
        import torch
        from baselines.transmil_lite import TransMILLite
        m = TransMILLite(input_dim=8, num_labels=3)
        x = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, dtype=torch.bool); mask[0, 3:] = True
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        y = (torch.rand(2, 3) > 0.5).float()
        out_ = m(x, mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out_['logits'], y)
        loss.backward()
        untrained = [n for n, p in m.named_parameters()
                     if p.requires_grad and p.grad is None]
        m.eval()
        with torch.no_grad():
            a = m(x, mask=mask)['attn_weights']
        out.append(f'- TransMIL parameters that receive no gradient from the '
                   f'bag loss: {untrained if untrained else "none"}')
        out.append(f'- TransMIL returned attention: shape {tuple(a.shape)}, rows '
                   f'sum to {a[0, 0].sum().item():.6f}, padded entries zero: '
                   f'{bool((a[0, 0, 3:] == 0).all())}, identical across labels: '
                   f'{torch.allclose(a[:, 0], a[:, -1])}')
        eff = {m_: d['used_eff_attn'] for (m_, ds), d in cells.items() if ds == 'mscv2'}
        out.append('- tensor scored per architecture (`used_eff_attn` flag in the '
                   'result files): ' + ', '.join(f'{DISPLAY[k]}={"eff_attn" if v else "attn_weights"}'
                                                 for k, v in eff.items()))
    except Exception as exc:                                # pragma: no cover
        out.append(f'- audit could not run: {exc}')
    out.append('')
    out.append('## 8. How to regenerate this report and the manuscript')
    out.append('')
    out.append('```bash')
    out.append('python3 analysis/make_attribution_tables.py --outdir paper_attribution/tables')
    out.append('python3 analysis/make_attribution_figures.py')
    out.append('python3 analysis/make_overview_figure.py')
    out.append('python3 analysis/attribution_stats.py --out paper_attribution/tables/attribution_stats.json')
    out.append('python3 analysis/make_setup_tables.py --outdir paper_attribution/tables')
    out.append('python3 analysis/verify_paper_claims.py')
    out.append('python3 analysis/make_verification_report.py')
    out.append('```')
    out.append('')
    open(args.out, 'w').write('\n'.join(out) + '\n')
    print(f'wrote {args.out}: verifier {n_pass} pass / {n_fail} fail, '
          f'{len(cells)} cells, {n_models} models')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
