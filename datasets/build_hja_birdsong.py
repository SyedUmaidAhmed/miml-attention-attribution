"""Build hja_birdsong.mat (DD format) from the HJA Bird Song Birds.mat.

HJA Bird Song (Briggs et al., KDD 2012): 548 recordings (bags), 10,232
syllable segments (instances), 38-d segment features, 13 species (labels).
The source file is a DD struct with fields X, Xnor, XtB, B, YB, YR, Y:

  X     raw 38-d segment features (the loader z-scores per fold with training
        statistics, as for letter_frost / letter_carroll / MSCV2, so the raw
        matrix is used; Xnor is X z-scored over ALL 10,232 instances with
        ddof=1, i.e. it uses test-fold statistics, and is not written).
  YR    per-instance species labels (instance ground truth; <=1 label per
        instance, OR over each bag equals YB exactly).
  Y     the bag labels broadcast to every instance (Y == YB[XtB-1]); not an
        instance ground truth, not written.

Output fields match letter_frost.mat exactly: DD.{X, YR, XtB, B, YB}, with YB
in {-1,+1} (int16) as in letter_frost.

    python3 -m datasets.build_hja_birdsong [--src PATH] [--out-dir DIR ...]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(
    _HERE, '..', '..', '..', 'Old-MIML Code', 'Birds-MIML-SELFVIT', 'Birds.mat')
DEFAULT_OUT_DIRS = [
    os.path.join(_HERE, 'data', 'MIML-SELFVIT'),
    os.path.join(_HERE, '..', 'VIT_MIL', 'Code', 'MIML-SELFVIT'),
]
OUT_NAME = 'hja_birdsong.mat'


def build(src):
    d = loadmat(src)['DD'][0, 0]
    X = np.asarray(d['X'], dtype=np.float64)
    XtB = np.asarray(d['XtB']).reshape(-1, 1).astype(np.float64)
    B = np.asarray(d['B']).reshape(-1, 1).astype(np.float64)
    YB01 = np.asarray(d['YB']).astype(np.int16)
    YR = np.asarray(d['YR']).astype(np.uint8)

    # Sanity: the facts the attribution study relies on.
    xtb = XtB.ravel().astype(int)
    assert X.shape == (10232, 38) and YR.shape == (10232, 13)
    assert B.shape[0] == 548 and YB01.shape == (548, 13)
    assert YR.sum(1).max() <= 1
    or_yr = np.stack([YR[xtb == b].max(0) for b in B.ravel().astype(int)])
    assert np.array_equal(or_yr, YB01), 'OR of YR over bag != YB'

    YB = np.where(YB01 > 0, 1, -1).astype(np.int16)
    return {'DD': {'X': X, 'YR': YR, 'XtB': XtB, 'B': B, 'YB': YB}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--out-dir', action='append', default=None)
    args = ap.parse_args()
    payload = build(args.src)
    for od in (args.out_dir or DEFAULT_OUT_DIRS):
        if not os.path.isdir(od):
            print(f'skip (no dir): {od}')
            continue
        path = os.path.join(od, OUT_NAME)
        savemat(path, payload, do_compression=True)
        print(f'wrote {os.path.abspath(path)}')


if __name__ == '__main__':
    main()
