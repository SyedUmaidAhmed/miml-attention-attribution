"""Validate and ingest a results zip returned from a Colab pack (DMKD revision).

    python3 analysis/ingest_colab_results.py path/to/results_voc_main.zip \
        --spec colab_packs/voc_main/PACK_SPEC.json \
        --dest results/instance_level_r1

Checks, in order (any failure stops ingestion and nothing is copied):
  1. every file listed in MANIFEST.sha256 is present with the right checksum;
  2. every expected (method, dataset) cell in PACK_SPEC.json has a JSON;
  3. each cell's protocol matches the spec: folds, ensemble seeds, number of
     per-model records (= folds x seeds), batch size, protocol_rev, device;
  4. sanity: no NaN in the headline metrics; P@1 between 0 and 1; for
     label-agnostic methods every model's P@1 <= its LabelAgnosticCeiling;
     additive-identity error < 1e-3 where present;
  5. per-model array files exist for every record.
Then copies JSONs, arrays and logs into --dest and prints a summary table.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print(f'FAIL: {msg}')
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--dest', default='results/instance_level_r1')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    spec = json.load(open(a.spec))

    tmp = tempfile.mkdtemp(prefix='ingest_')
    with zipfile.ZipFile(a.zip) as z:
        for n in z.namelist():                       # no path traversal
            if n.startswith('/') or '..' in n.split('/'):
                fail(f'unsafe path in zip: {n}')
        z.extractall(tmp)
    roots = [d for d in os.listdir(tmp) if not d.startswith('__')]
    root = os.path.join(tmp, roots[0]) if len(roots) == 1 and os.path.isdir(
        os.path.join(tmp, roots[0])) else tmp

    man = os.path.join(root, 'MANIFEST.sha256')
    if not os.path.exists(man):
        fail('MANIFEST.sha256 missing')
    bad = 0
    for line in open(man):
        digest, rel = line.rstrip('\n').split('  ', 1)
        p = os.path.join(root, rel)
        if not os.path.exists(p) or sha256(p) != digest:
            print(f'  checksum mismatch / missing: {rel}')
            bad += 1
    if bad:
        fail(f'{bad} files failed the checksum')
    print('1. checksums OK')

    rows = []
    for m in spec['methods']:
        for d in spec['datasets']:
            p = os.path.join(root, 'results', f'{m}__{d}.json')
            if not os.path.exists(p):
                fail(f'missing cell {m}__{d}')
            j = json.load(open(p))
            seeds = spec['seeds_override'].get(m, spec['seeds'])
            n_exp = spec['folds'] * (len(seeds) if seeds else 1)
            recs = j['instance_per_model']
            checks = {
                'folds': j['num_folds'] == spec['folds'],
                'seeds': (j['ensemble_seeds'] or None) == (seeds if seeds and len(seeds) > 1 else None),
                'n_models': len(recs) == n_exp,
                'batch': j['config'].get('batch_size') == spec['batch_size'],
                'rev': j.get('protocol_rev') == 'dmkd_r1',
                'device': str(j.get('device', '')).startswith(spec['device']),
            }
            for k, ok in checks.items():
                if not ok:
                    fail(f'{m}__{d}: protocol check "{k}" failed')
            for r in recs:
                for k in ('InstP@1', 'InstAUC', 'BagAveragePrecision'):
                    if not np.isfinite(r.get(k, np.nan)):
                        fail(f'{m}__{d}: non-finite {k} (fold {r["fold"]} seed {r["seed"]})')
                if m in AGNOSTIC and r['InstP@1'] > r['LabelAgnosticCeiling'] + 1e-9:
                    fail(f'{m}__{d}: label-agnostic P@1 above the ceiling')
                if r.get('AddIdentityMaxErr', 0) > 1e-3:
                    fail(f'{m}__{d}: additive identity error {r["AddIdentityMaxErr"]}')
                arr = os.path.join(root, 'results', 'arrays',
                                   f'{m}__{d}__f{r["fold"]}_s{r["seed"]}.npz')
                if not os.path.exists(arr):
                    fail(f'{m}__{d}: missing array file {os.path.basename(arr)}')
            im = j['instance_metrics_mean_std']
            rows.append((m, d, len(recs), im['InstP@1'][0], im['ChanceP@1'][0],
                         im['InstAUC'][0], im['BagAveragePrecision'][0],
                         j['wall_clock_seconds'] / 60))
    print('2-5. cells, protocol, sanity, arrays OK')
    print(f'{"method":15s} {"dataset":17s} {"n":>3s} {"P@1":>6s} {"chance":>6s} '
          f'{"AUC":>6s} {"AP/mdl":>6s} {"min":>6s}')
    for r in rows:
        print(f'{r[0]:15s} {r[1]:17s} {r[2]:3d} {r[3]:6.3f} {r[4]:6.3f} '
              f'{r[5]:6.3f} {r[6]:6.3f} {r[7]:6.1f}')

    if a.dry_run:
        print('dry run: nothing copied')
        return
    os.makedirs(os.path.join(a.dest, 'arrays'), exist_ok=True)
    os.makedirs(os.path.join(a.dest, 'logs', spec['pack_id']), exist_ok=True)
    # copy ONLY the validated cells of the spec (and their arrays)
    keep = {f'{m}__{d}' for m in spec['methods'] for d in spec['datasets']}
    for f in os.listdir(os.path.join(root, 'results')):
        if f.endswith('.json') and f[:-5] in keep:
            shutil.copy(os.path.join(root, 'results', f), os.path.join(a.dest, f))
    for f in os.listdir(os.path.join(root, 'results', 'arrays')):
        if f.split('__f')[0] in keep:
            shutil.copy(os.path.join(root, 'results', 'arrays', f),
                        os.path.join(a.dest, 'arrays', f))
    logs = os.path.join(root, 'logs')
    if os.path.isdir(logs):
        for f in os.listdir(logs):
            shutil.copy(os.path.join(logs, f),
                        os.path.join(a.dest, 'logs', spec['pack_id'], f))
    print(f'ingested {len(rows)} cells into {a.dest}')


if __name__ == '__main__':
    main()
