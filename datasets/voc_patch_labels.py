"""Patch-level (instance-level) ground truth for the VOC-Patch MIML benchmark.

`voc_miml.py` slices each image into a 4x4 grid of patches and treats the 16
patch features as a bag, with the image's 20-dim class vector as the bag label.
The spatial annotations that say *where* each class is are parsed and then
discarded. This module recovers them, giving a per-patch label matrix that can
serve as instance-level ground truth.

Two independent sources, both produced here:

  bbox  — from `Annotations/*.xml` (17,125 images). A patch carries class c if
          the intersection of some non-difficult c-box with the patch covers at
          least `thresh` of the patch area. Approximate: boxes include
          background, so this over-assigns for non-convex or thin objects.

  seg   — from `SegmentationClass/*.png` (2,913 images). A patch carries class c
          if at least `thresh` of its non-void pixels belong to c. Exact up to
          the threshold, since the mask is pixel-accurate.

The seg subset is the gold standard and is also used to quantify how good the
bbox approximation is on the images where both exist — that agreement number is
what justifies using the bbox-derived labels on the full 17k.

Patch geometry here MUST match `voc_miml._patch_grid` exactly: equal-sized
patches with the last row/column absorbing the remainder pixels.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from .voc_miml import VOC_CLASSES, NUM_CLASSES, GRID, _voc_root

__all__ = [
    'patch_bounds',
    'patch_labels_from_bbox',
    'patch_labels_from_segmentation',
    'segmentation_image_ids',
    'bbox_vs_seg_agreement',
]


def patch_bounds(w, h, grid=GRID):
    """Pixel bounds of each patch as (left, top, right, bottom), row-major.

    Mirrors `voc_miml._patch_grid` so patch i here is patch i there.
    """
    pw, ph = w // grid, h // grid
    out = []
    for r in range(grid):
        for c in range(grid):
            left = c * pw
            top = r * ph
            right = (c + 1) * pw if c < grid - 1 else w
            bottom = (r + 1) * ph if r < grid - 1 else h
            out.append((left, top, right, bottom))
    return out


def _devkit(root=None):
    root = root or _voc_root()
    return os.path.join(root, 'VOCdevkit', 'VOC2012')


def patch_labels_from_bbox(image_id, devkit=None, grid=GRID, thresh=0.25,
                           include_difficult=False):
    """(grid*grid, 20) binary patch labels from the XML bounding boxes.

    A patch is labelled c when some c-box covers >= `thresh` of the patch area.
    Returns None if the annotation is unreadable.
    """
    devkit = devkit or _devkit()
    xml_path = os.path.join(devkit, 'Annotations', f'{image_id}.xml')
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None

    size = root.find('size')
    if size is None:
        return None
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bounds = patch_bounds(w, h, grid)
    labels = np.zeros((grid * grid, NUM_CLASSES), dtype=bool)

    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None or name_el.text is None:
            continue
        cls = name_el.text.strip().lower()
        if cls not in VOC_CLASSES:
            continue
        if not include_difficult:
            diff = obj.find('difficult')
            if diff is not None and diff.text and int(diff.text) == 1:
                continue
        bb = obj.find('bndbox')
        if bb is None:
            continue
        # VOC boxes are 1-indexed and inclusive.
        x0 = float(bb.find('xmin').text) - 1.0
        y0 = float(bb.find('ymin').text) - 1.0
        x1 = float(bb.find('xmax').text)
        y1 = float(bb.find('ymax').text)
        ci = VOC_CLASSES.index(cls)

        for pi, (pl, pt, pr, pb) in enumerate(bounds):
            iw = max(0.0, min(x1, pr) - max(x0, pl))
            ih = max(0.0, min(y1, pb) - max(y0, pt))
            if iw <= 0 or ih <= 0:
                continue
            patch_area = float((pr - pl) * (pb - pt))
            if patch_area > 0 and (iw * ih) / patch_area >= thresh:
                labels[pi, ci] = True

    return labels


def segmentation_image_ids(devkit=None):
    """IDs of the images that have a pixel-level SegmentationClass mask."""
    devkit = devkit or _devkit()
    seg_dir = os.path.join(devkit, 'SegmentationClass')
    if not os.path.isdir(seg_dir):
        return []
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(seg_dir) if f.endswith('.png'))


def patch_labels_from_segmentation(image_id, devkit=None, grid=GRID,
                                   thresh=0.25):
    """(grid*grid, 20) binary patch labels from the pixel-exact mask.

    A patch is labelled c when at least `thresh` of its non-void pixels carry
    class c. Palette index 255 (the void/boundary band) is excluded from the
    denominator; background pixels (index 0) are counted in it, so the
    denominator is every non-void pixel of the patch. Returns None if no mask
    exists.
    """
    from PIL import Image

    devkit = devkit or _devkit()
    png = os.path.join(devkit, 'SegmentationClass', f'{image_id}.png')
    if not os.path.exists(png):
        return None

    mask = np.array(Image.open(png))          # palette indices, HxW
    h, w = mask.shape
    bounds = patch_bounds(w, h, grid)
    labels = np.zeros((grid * grid, NUM_CLASSES), dtype=bool)

    for pi, (pl, pt, pr, pb) in enumerate(bounds):
        sub = mask[pt:pb, pl:pr]
        if sub.size == 0:
            continue
        valid = sub[(sub != 255)]             # drop the void band
        if valid.size == 0:
            continue
        denom = float(valid.size)
        for ci in range(NUM_CLASSES):
            # palette index for class ci is ci + 1 (0 == background)
            if (valid == (ci + 1)).sum() / denom >= thresh:
                labels[pi, ci] = True

    return labels


def bbox_vs_seg_agreement(devkit=None, grid=GRID, thresh=0.25, limit=None):
    """Quality check: how well do bbox-derived patch labels match the mask ones?

    Runs over the images that have both. Returns a dict of agreement stats —
    this is the number that justifies extending the bbox-derived labels to the
    full 17k-image set.
    """
    devkit = devkit or _devkit()
    ids = segmentation_image_ids(devkit)
    if limit:
        ids = ids[:limit]

    tp = fp = fn = tn = 0
    exact_patches = 0
    total_patches = 0

    for iid in ids:
        a = patch_labels_from_bbox(iid, devkit, grid, thresh)
        b = patch_labels_from_segmentation(iid, devkit, grid, thresh)
        if a is None or b is None:
            continue
        tp += int((a & b).sum())
        fp += int((a & ~b).sum())
        fn += int((~a & b).sum())
        tn += int((~a & ~b).sum())
        exact_patches += int((a == b).all(axis=1).sum())
        total_patches += a.shape[0]

    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float('nan')
    return {
        'images': len(ids),
        'patches': total_patches,
        'cellwise_accuracy': (tp + tn) / max(tp + tn + fp + fn, 1),
        'exact_patch_match': exact_patches / max(total_patches, 1),
        'precision_vs_seg': prec,
        'recall_vs_seg': rec,
        'f1_vs_seg': f1,
        'threshold': thresh,
    }
