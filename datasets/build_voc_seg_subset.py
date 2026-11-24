"""Build VOC-Patch-Seg: the segmentation-derived instance-labelled MIML benchmark.

Takes the rows of `voc2012.mat` that have a VOC SegmentationClass mask and
writes them out as a standalone MIML dataset plus a matching patch-label file.

Why only the segmentation subset? At the 0.05 overlap threshold the
bounding-box-derived patch labels have 95.7% recall but only 74.4% precision
against the mask-derived labels on the 2,913 masked images (boxes contain
background, so they over-assign). Over-assignment *inflates* measured
attribution — a wrong guess can land on a spurious witness and score as
correct. The bbox set is therefore sound for negative claims ("this method
fails to localize") but not for positive ones. Every claim that a method *does*
localize has to rest on mask-derived labels, which is this subset.

Outputs (into DATA_DIR):
    voc2012seg.mat        bags (n,1) object array of (16, 2048), labels (n,20) in {-1,+1}
    voc2012seg_10CV.mat   Indexes permutation, same convention as the other datasets
    voc2012seg_patch_gt.npz   patch_seg (n,16,20) bool, bag_labels (n,20) bool, image_ids

Usage:
    python3 -m datasets.build_voc_seg_subset
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat, savemat

from .voc_miml import DEFAULT_OUT_DIR


def build(data_dir: str = DEFAULT_OUT_DIR, seed: int = 42,
          multilabel_only: bool = False, name: str | None = None):
    src_mat = os.path.join(data_dir, 'voc2012.mat')
    src_gt = os.path.join(data_dir, 'voc2012_patch_gt.npz')
    for p in (src_mat, src_gt):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} missing. Run `python3 -m datasets.voc_miml` and '
                '`python3 -m datasets.build_voc_patch_gt` first.')

    m = loadmat(src_mat)
    bags, labels = m['bags'], m['labels']
    z = np.load(src_gt, allow_pickle=True)
    has_seg = z['has_seg']
    patch_seg = z['patch_seg']
    bag_gt = z['bag_labels']
    ids = z['image_ids']

    if bags.shape[0] != has_seg.shape[0]:
        raise RuntimeError(
            f'row mismatch: voc2012.mat has {bags.shape[0]} bags, '
            f'patch GT has {has_seg.shape[0]}')

    # Alignment guard: the .mat labels must agree with the patch-GT bag labels
    # row-for-row, or the patch labels belong to different images.
    if not ((labels > 0) == bag_gt).all():
        raise RuntimeError('bag labels disagree between .mat and patch GT — '
                           'row order is not aligned; refusing to build')

    # Bag labels. VOC-Patch-Seg keeps the image-level VOC annotation (from the
    # XML, `difficult` objects excluded), exactly as the data used in the paper
    # (`voc2012seg.mat` of 2026-08-17): 103 of its 3,095 labels then have no
    # patch witness at tau = 0.05 and are non-scorable, and 109 patch witnesses
    # belong to `difficult` objects absent from the label vector. Its bag labels
    # therefore do not depend on tau.
    # VOC-Patch-Seg-ML (--multilabel-only) redefines bag labels as the OR of the
    # patch labels, which makes OR-consistency hold by construction, and keeps
    # bags with >= 2 derived labels; both its labels and its membership depend
    # on tau.
    # (A refactor in commit c7a4097 applied the OR to both subsets; that version
    # never produced a file used in the paper. Restored 2026-11-24, DMKD
    # revision, defect D3.)
    derived = patch_seg.any(axis=1)                     # (n_all, L) bool

    sel = np.nonzero(has_seg)[0]
    if multilabel_only:
        sel = sel[derived[sel].sum(axis=1) >= 2]
    n = len(sel)

    stem = name or ('voc2012segml' if multilabel_only else 'voc2012seg')

    out_bags = np.empty((n, 1), dtype=object)
    for i, gi in enumerate(sel):
        out_bags[i, 0] = np.asarray(bags[gi, 0], dtype=np.float32)
    # {-1,+1} convention
    if multilabel_only:
        out_labels = np.where(derived[sel], 1, -1).astype(np.int32)
    else:
        out_labels = np.where(labels[sel] > 0, 1, -1).astype(np.int32)

    mat_path = os.path.join(data_dir, f'{stem}.mat')
    savemat(mat_path, {'bags': out_bags, 'labels': out_labels})

    rng = np.random.RandomState(seed)
    perm = (rng.permutation(n) + 1).astype(np.uint16).reshape(1, -1)
    cv_path = os.path.join(data_dir, f'{stem}_10CV.mat')
    savemat(cv_path, {'Indexes': perm})

    gt_path = os.path.join(data_dir, f'{stem}_patch_gt.npz')
    np.savez_compressed(gt_path,
                        patch_seg=patch_seg[sel],
                        bag_labels=derived[sel],
                        image_ids=ids[sel])

    lab = derived[sel]
    # Consistency now holds by construction; assert it rather than trust it.
    assert (patch_seg[sel].any(axis=1) == lab).all(), \
        'derived bag labels disagree with the OR of patch labels'
    print(f'wrote {mat_path}')
    print(f'wrote {cv_path}')
    print(f'wrote {gt_path}')
    print(f'  bags                {n}')
    print(f'  instances/bag       {out_bags[0,0].shape[0]}')
    print(f'  feature dim         {out_bags[0,0].shape[1]}')
    print(f'  labels              {lab.shape[1]}')
    print(f'  mean labels/bag     {lab.sum(1).mean():.2f}')
    print(f'  multi-label bags    {100*(lab.sum(1) >= 2).mean():.1f}%')
    return mat_path, cv_path, gt_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--multilabel-only', action='store_true',
                    help='keep only bags with >=2 derived labels -- the '
                         'population where label-agnostic attention can '
                         'actually be penalised')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    build(args.data_dir, args.seed, args.multilabel_only, args.name)


if __name__ == '__main__':
    main()
