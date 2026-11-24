"""VOC-Patch features without the centre crop, at 4x4 or 8x8 (DMKD revision).

Reviewer 1 (pt 4) noted that "standard ImageNet preprocessing" includes a
centre crop: torchvision's IMAGENET1K_V2 transform resizes the short side to
232 and crops 224, so on a non-square patch part of the long side, where a
witness may lie, never reaches the features. This builder re-extracts the
features with the whole patch resized to 224x224 (no crop), keeping
everything else fixed:

  * the same images, in the same row order, as the shipped VOC-Patch-Seg /
    Seg-ML files (read from their *_patch_gt.npz `image_ids`);
  * the same bag labels (Seg: image-level; Seg-ML: OR of 4x4 patch labels at
    tau = 0.05) and the same fold permutation;
  * the same frozen ResNet-50 IMAGENET1K_V2 backbone (2048-d avgpool).

It also stores the raw per-patch, per-class overlap fraction (non-void pixels
of the patch belonging to the class), so patch labels at any tau can be
recomputed without touching the images (reviewer 1 pt 4, reviewer 3 pt 2).

Outputs, next to the shipped files (never overwriting them):
    <stem>.mat            bags (n,1) of (G*G, 2048) float32, labels (n,20) {-1,+1}
    <stem>_10CV.mat       copy of the shipped fold permutation
    <stem>_patch_gt.npz   patch_seg (n,G*G,20) bool at tau=0.05,
                          overlap_frac (n,G*G,20) float16, image_ids, bag_labels
with stem = voc2012seg_nc | voc2012segml_nc (grid 4) or voc2012segml_g8 (grid 8).

Gate (checked at build time for grid 4): patch_seg recomputed from the overlap
fractions at tau = 0.05 must equal the shipped patch_seg exactly.

    python3 -m datasets.build_voc_patch_r1 --grid 4
    python3 -m datasets.build_voc_patch_r1 --grid 8 --subset segml
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
from scipy.io import loadmat, savemat

from .voc_miml import DEFAULT_OUT_DIR, NUM_CLASSES, _voc_root
from .voc_patch_labels import patch_bounds

TAU = 0.05
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _devkit():
    return os.path.join(_voc_root(), 'VOCdevkit', 'VOC2012')


def overlap_fractions(mask, grid):
    """(grid*grid, 20) fraction of each patch's non-void pixels per class."""
    h, w = mask.shape
    out = np.zeros((grid * grid, NUM_CLASSES), np.float64)
    for pi, (pl, pt, pr, pb) in enumerate(patch_bounds(w, h, grid)):
        sub = mask[pt:pb, pl:pr]
        valid = sub[sub != 255]
        if valid.size == 0:
            continue
        cnt = np.bincount(valid.ravel(), minlength=256)[1:NUM_CLASSES + 1]
        out[pi] = cnt / float(valid.size)
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


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build(subset, grid, data_dir=DEFAULT_OUT_DIR):
    import torch
    from PIL import Image

    src = 'voc2012seg' if subset == 'seg' else 'voc2012segml'
    stem = f'{src}_nc' if grid == 4 else f'{src}_g{grid}'
    z = np.load(os.path.join(data_dir, f'{src}_patch_gt.npz'), allow_pickle=True)
    ids = [str(x) for x in z['image_ids']]
    shipped_labels = loadmat(os.path.join(data_dir, f'{src}.mat'),
                             variable_names=['labels'])['labels']
    n = len(ids)
    devkit = _devkit()
    dev = _device()
    net, tf = _extractor(dev)
    print(f'[r1] {stem}: {n} images, grid {grid}, device {dev}', flush=True)

    bags = np.empty((n, 1), dtype=object)
    frac = np.zeros((n, grid * grid, NUM_CLASSES), np.float32)
    t0 = time.time()
    for i, iid in enumerate(ids):
        mask = np.array(Image.open(os.path.join(
            devkit, 'SegmentationClass', f'{iid}.png')))
        frac[i] = overlap_fractions(mask, grid)
        with Image.open(os.path.join(devkit, 'JPEGImages', f'{iid}.jpg')) as im:
            im = im.convert('RGB')
            if im.size != (mask.shape[1], mask.shape[0]):
                raise RuntimeError(f'{iid}: image/mask size mismatch')
            w, h = im.size
            patches = [im.crop(b) for b in patch_bounds(w, h, grid)]
        batch = torch.stack([tf(p) for p in patches]).to(dev)
        with torch.no_grad():
            f = net(batch).flatten(1).cpu().numpy().astype(np.float32)
        bags[i, 0] = f
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f'[r1]   {i + 1}/{n}  ({(time.time() - t0) / 60:.1f} min)',
                  flush=True)

    patch_seg = frac >= TAU
    # Gate: at 4x4 the recomputed labels must equal the shipped ones exactly.
    if grid == 4:
        ok = np.array_equal(patch_seg, z['patch_seg'].astype(bool))
        print(f'[r1] GATE patch_seg == shipped: {ok}', flush=True)
        if not ok:
            diff = int((patch_seg != z['patch_seg'].astype(bool)).sum())
            raise RuntimeError(f'patch labels differ from shipped in {diff} cells')
        labels = shipped_labels
    else:
        # Finer grid: Seg-ML bag labels are the OR of its own patch labels.
        labels = np.where(patch_seg.any(axis=1), 1, -1).astype(np.int32)
        changed = int((labels != shipped_labels).any(axis=1).sum())
        lt2 = int(((labels > 0).sum(axis=1) < 2).sum())
        print(f'[r1] grid {grid}: {changed} bags change label set vs 4x4; '
              f'{lt2} bags have < 2 labels', flush=True)

    savemat(os.path.join(data_dir, f'{stem}.mat'),
            {'bags': bags, 'labels': labels.astype(np.int32)})
    shutil.copy(os.path.join(data_dir, f'{src}_10CV.mat'),
                os.path.join(data_dir, f'{stem}_10CV.mat'))
    np.savez_compressed(
        os.path.join(data_dir, f'{stem}_patch_gt.npz'),
        patch_seg=patch_seg, overlap_frac=frac.astype(np.float16),
        image_ids=np.array(ids), bag_labels=labels > 0,
        threshold=np.float32(TAU), grid=np.int32(grid),
        preprocessing=np.array('resize-224x224-bilinear-antialias-no-crop'))
    print(f'[r1] wrote {stem}.mat / _10CV.mat / _patch_gt.npz', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=4, choices=[4, 8])
    ap.add_argument('--subset', default='both', choices=['seg', 'segml', 'both'])
    ap.add_argument('--data-dir', default=DEFAULT_OUT_DIR)
    a = ap.parse_args()
    subsets = ['seg', 'segml'] if a.subset == 'both' else [a.subset]
    for s in subsets:
        build(s, a.grid, a.data_dir)


if __name__ == '__main__':
    main()
