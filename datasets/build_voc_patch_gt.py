"""Build the VOC-Patch instance-level ground-truth file.

Emits one .npz whose row order is EXACTLY `voc_miml._read_image_ids(devkit)` —
the trainval.txt order that `build_voc_miml` uses when writing `voc2012.mat` —
so patch labels line up with the extracted feature bags index-for-index.

Contents:
    image_ids     (n,)          image id strings
    bag_labels    (n, 20)       bool, image-level labels (OR over objects)
    patch_bbox    (n, 16, 20)   bool, patch labels derived from bounding boxes
    patch_seg     (n, 16, 20)   bool, patch labels from segmentation masks
    has_seg       (n,)          bool, True where a pixel-exact mask exists
    threshold     scalar        the overlap threshold used

Usage:
    python3 -m datasets.build_voc_patch_gt --thresh 0.05
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .voc_miml import (
    _voc_root, _ensure_voc_downloaded, _read_image_ids, _read_label_vector,
    NUM_CLASSES, GRID, DEFAULT_OUT_DIR,
)
from .voc_patch_labels import (
    patch_labels_from_bbox, patch_labels_from_segmentation,
)

PATCHES = GRID * GRID


def build(out_dir=DEFAULT_OUT_DIR, thresh=0.05, limit=None):
    root = _voc_root()
    devkit = _ensure_voc_downloaded(root)
    ids = _read_image_ids(devkit)
    if limit:
        ids = ids[:limit]
    n = len(ids)

    bag = np.zeros((n, NUM_CLASSES), dtype=bool)
    p_bbox = np.zeros((n, PATCHES, NUM_CLASSES), dtype=bool)
    p_seg = np.zeros((n, PATCHES, NUM_CLASSES), dtype=bool)
    has_seg = np.zeros(n, dtype=bool)

    log_every = max(1, n // 20)
    for i, iid in enumerate(ids):
        bag[i] = _read_label_vector(devkit, iid) > 0

        b = patch_labels_from_bbox(iid, devkit, GRID, thresh)
        if b is not None:
            p_bbox[i] = b

        s = patch_labels_from_segmentation(iid, devkit, GRID, thresh)
        if s is not None:
            p_seg[i] = s
            has_seg[i] = True

        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f'[voc-gt] {i+1}/{n}', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'voc2012_patch_gt.npz')
    np.savez_compressed(
        out, image_ids=np.array(ids), bag_labels=bag,
        patch_bbox=p_bbox, patch_seg=p_seg, has_seg=has_seg,
        threshold=np.array(thresh))

    # Report the quantities that decide whether the benchmark is well posed.
    def coverage(patch, sel):
        if sel.sum() == 0:
            return float('nan')
        derived = patch[sel].any(axis=1)         # (m, 20)
        tot = bag[sel].sum()
        return float((bag[sel] & derived).sum() / max(tot, 1))

    allsel = np.ones(n, dtype=bool)
    print(f'\nwrote {out}')
    print(f'  images                       {n}')
    print(f'  with pixel-exact mask        {int(has_seg.sum())}')
    print(f'  mean labels/bag              {bag.sum(1).mean():.2f}')
    print(f'  multi-label bags             {100*(bag.sum(1) >= 2).mean():.1f}%')
    print(f'  OR-coverage (bbox, all)      {100*coverage(p_bbox, allsel):.1f}%')
    print(f'  OR-coverage (seg, seg-only)  {100*coverage(p_seg, has_seg):.1f}%')
    print(f'  labelled patches/bag (bbox)  {p_bbox.any(2).sum(1).mean():.2f} / {PATCHES}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--thresh', type=float, default=0.05)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    build(args.out_dir, args.thresh, args.limit)


if __name__ == '__main__':
    main()
