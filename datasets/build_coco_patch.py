"""COCO-Patch-ML: an 8x8-grid, instance-labelled MIML benchmark from COCO 2017 val.

A larger, real-image counterpart of VOC-Patch. Mirrors
`datasets/build_voc_patch_r1.py` (the VOC-Patch-Seg-ML 8x8 builder) with the
dataset swapped; every protocol choice that can be kept identical is kept
identical.

Construction
------------
* Images: all 5,000 images of COCO 2017 **val** (`val2017/`), with the
  instance masks of `annotations/instances_val2017.json` (80 thing classes;
  label index = position of the category id in ascending id order).
* Grid: 8x8, geometry `datasets.voc_patch_labels.patch_bounds` (equal patches,
  last row/column absorbs the remainder), so patch i is row-major i.
* Features: frozen ImageNet ResNet-50 (`IMAGENET1K_V2`), 2048-d avgpool, the
  whole patch resized to 224x224 bilinear with antialias, **no centre crop**,
  ImageNet mean/std. Grayscale images are converted to RGB.
* Overlap fraction (stored as float16, so tau can be re-scored):
      frac[p, c] = |pixels of patch p covered by a non-crowd class-c mask
                    and not covered by any crowd region|
                   / |pixels of patch p not covered by any crowd region|.
  The class-c mask is the union of all non-crowd class-c instance masks
  (pycocotools `annToMask`). This is the COCO analogue of VOC's "non-void
  pixels": COCO has no void label, and crowd (`iscrowd=1`) regions are the
  annotations whose extent is unreliable, so they are excluded from both the
  numerator and the denominator. A patch entirely covered by crowd regions
  gets frac = 0 for every class (it can never be a witness).
* Patch label: frac >= tau, tau = 0.05 (as VOC-Patch).
* Bag labels: OR of the patch labels (as VOC-Patch-Seg-ML). Consequence for
  small objects: a class whose instances cover < 5% of every non-crowd patch
  area (about 240 px for a 640x480 image, i.e. roughly 15x15 px) is NOT a bag
  label of that image. We do not add such objects as label-without-witness
  pairs, because the attribution metrics need >= 1 witness per scorable pair
  and the paper's Seg-ML benchmarks are defined the same way. The number of
  image-level (annotation) labels dropped this way is printed and stored.
  Crowd-only classes likewise never become bag labels.
* Bags kept: those with >= 2 bag labels (as Seg-ML).
* Folds: a fixed permutation, numpy RandomState(42), 1-indexed uint16,
  `Indexes` key (same convention as build_voc_seg_subset.py).

Outputs, in --out-dir (default $COCO_PATCH_DIR or ~/data/coco_patch):
    coco2017val_g8.mat            bags (n,1) of (64, 2048) float32,
                                  labels (n,80) int32 in {-1,+1}
    coco2017val_g8_10CV.mat       Indexes (1,n) uint16
    coco2017val_g8_patch_gt.npz   patch_seg (n,64,80) bool, overlap_frac float16,
                                  image_ids, bag_labels, class names, stats

    python3 -m datasets.build_coco_patch --limit 20 --out-dir /tmp/coco_canary
    python3 -m datasets.build_coco_patch
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.io import savemat

from .voc_patch_labels import patch_bounds

TAU = 0.05
GRID = 8
MIN_LABELS = 2
FOLD_SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
STEM = 'coco2017val_g8'

DEFAULT_COCO_ROOT = os.environ.get('COCO_ROOT',
                                   os.path.expanduser('~/data/coco2017'))
DEFAULT_OUT_DIR = os.environ.get('COCO_PATCH_DIR',
                                 os.path.expanduser('~/data/coco_patch'))


def class_masks(coco, anns, h, w, cat_index):
    """(L, h, w) bool union mask per class over non-crowd anns; (h, w) crowd."""
    L = len(cat_index)
    cls = np.zeros((L, h, w), bool)
    crowd = np.zeros((h, w), bool)
    for a in anns:
        m = coco.annToMask(a).astype(bool)
        if a.get('iscrowd', 0):
            crowd |= m
        else:
            cls[cat_index[a['category_id']]] |= m
    return cls, crowd


def overlap_fractions(cls, crowd, grid=GRID):
    """(grid*grid, L) fraction of each patch's non-crowd pixels per class."""
    L, h, w = cls.shape
    out = np.zeros((grid * grid, L), np.float64)
    valid = ~crowd
    for pi, (pl, pt, pr, pb) in enumerate(patch_bounds(w, h, grid)):
        v = valid[pt:pb, pl:pr]
        nv = int(v.sum())
        if nv == 0:
            continue
        sub = cls[:, pt:pb, pl:pr] & v[None]
        out[pi] = sub.reshape(L, -1).sum(1) / float(nv)
    return out


def _extractor(device):
    import torch
    from torchvision import transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    tf = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR,
                 antialias=True),        # whole patch, no crop
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return net, tf


def build(coco_root, out_dir, device='cpu', limit=None, threads=None):
    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    if threads:
        torch.set_num_threads(threads)
    os.makedirs(out_dir, exist_ok=True)
    coco = COCO(os.path.join(coco_root, 'annotations', 'instances_val2017.json'))
    cat_ids = sorted(coco.getCatIds())
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    names = [coco.loadCats(c)[0]['name'] for c in cat_ids]
    L = len(cat_ids)
    img_ids = sorted(coco.getImgIds())
    if limit:
        img_ids = img_ids[:limit]

    # Pass 1: labels only (cheap), so features are extracted for kept bags only.
    t0 = time.time()
    fracs, keep, n_ann_labels, n_dropped_small = {}, [], 0, 0
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
        cls, crowd = class_masks(coco, anns, info['height'], info['width'],
                                 cat_index)
        fr = overlap_fractions(cls, crowd)
        ann_lab = {cat_index[a['category_id']] for a in anns
                   if not a.get('iscrowd', 0)}
        bag = (fr >= TAU).any(0)
        n_ann_labels += len(ann_lab)
        n_dropped_small += len(ann_lab) - int(bag.sum())
        if bag.sum() >= MIN_LABELS:
            fracs[iid] = fr
            keep.append(iid)
    n = len(keep)
    print(f'[coco] labels pass: {len(img_ids)} images -> {n} bags with '
          f'>= {MIN_LABELS} labels ({(time.time() - t0) / 60:.1f} min); '
          f'{n_dropped_small}/{n_ann_labels} annotated image-labels have no '
          f'patch >= tau', flush=True)

    dev = torch.device(device)
    net, tf = _extractor(dev)
    bags = np.empty((n, 1), dtype=object)
    frac = np.zeros((n, GRID * GRID, L), np.float32)
    t1 = time.time()
    for i, iid in enumerate(keep):
        info = coco.loadImgs(iid)[0]
        frac[i] = fracs[iid]
        with Image.open(os.path.join(coco_root, 'val2017',
                                     info['file_name'])) as im:
            im = im.convert('RGB')
            if im.size != (info['width'], info['height']):
                raise RuntimeError(f'{iid}: image/annotation size mismatch')
            w, h = im.size
            patches = [im.crop(b) for b in patch_bounds(w, h, GRID)]
        batch = torch.stack([tf(p) for p in patches]).to(dev)
        with torch.no_grad():
            f = net(batch).flatten(1).cpu().numpy().astype(np.float32)
        bags[i, 0] = f
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f'[coco]   {i + 1}/{n}  ({(time.time() - t1) / 60:.1f} min)',
                  flush=True)
    feat_min = (time.time() - t1) / 60

    patch_seg = frac >= TAU
    bag_labels = patch_seg.any(axis=1)
    assert (bag_labels.sum(1) >= MIN_LABELS).all()
    labels = np.where(bag_labels, 1, -1).astype(np.int32)

    savemat(os.path.join(out_dir, f'{STEM}.mat'),
            {'bags': bags, 'labels': labels})
    perm = (np.random.RandomState(FOLD_SEED).permutation(n) + 1)
    savemat(os.path.join(out_dir, f'{STEM}_10CV.mat'),
            {'Indexes': perm.astype(np.uint16).reshape(1, -1)})
    stats = dict(images=len(img_ids), bags=n, labels=L, grid=GRID, tau=TAU,
                 annotated_image_labels=n_ann_labels,
                 dropped_small_image_labels=n_dropped_small,
                 labels_per_bag=float(bag_labels.sum(1).mean()),
                 feature_device=str(dev), feature_minutes=feat_min,
                 torch_version=torch.__version__)
    np.savez_compressed(
        os.path.join(out_dir, f'{STEM}_patch_gt.npz'),
        patch_seg=patch_seg, overlap_frac=frac.astype(np.float16),
        image_ids=np.array(keep, np.int64), bag_labels=bag_labels,
        class_names=np.array(names), category_ids=np.array(cat_ids),
        threshold=np.float32(TAU), grid=np.int32(GRID),
        preprocessing=np.array('resize-224x224-bilinear-antialias-no-crop'),
        crowd_rule=np.array('crowd pixels excluded from numerator and denominator'),
        stats_json=np.array(json.dumps(stats)))
    print(f'[coco] wrote {STEM}.mat / _10CV.mat / _patch_gt.npz in {out_dir}; '
          f'{json.dumps(stats)}', flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-root', default=DEFAULT_COCO_ROOT)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    ap.add_argument('--limit', type=int, default=None,
                    help='first N images by id (canary)')
    ap.add_argument('--threads', type=int, default=None)
    a = ap.parse_args()
    build(a.coco_root, a.out_dir, a.device, a.limit, a.threads)


if __name__ == '__main__':
    main()
