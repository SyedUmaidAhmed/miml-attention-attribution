"""Agreement of bounding-box-derived with segmentation-derived VOC patch labels.

Recall and precision of box-derived patch labels against mask-derived ones
at the default overlap threshold on every VOC 2012 trainval image with a mask
(ImageSets/Segmentation/trainval.txt, 2,913 images). Needs the VOC devkit.

    /opt/anaconda3/bin/python3 analysis/voc_bbox_agreement.py
writes paper_attribution/tables/voc_bbox_agreement.json
"""
import json
import os

import numpy as np

from datasets.voc_patch_labels import (_devkit, patch_labels_from_bbox,
                                       patch_labels_from_segmentation)

OUT = 'paper_attribution/tables/voc_bbox_agreement.json'
THRESH = 0.05


def main():
    dk = _devkit()
    ids = open(os.path.join(dk, 'ImageSets', 'Segmentation', 'trainval.txt')).read().split()
    tp = nb = ns = 0
    for iid in ids:
        b = np.asarray(patch_labels_from_bbox(iid, dk, thresh=THRESH), bool)
        s = np.asarray(patch_labels_from_segmentation(iid, dk, thresh=THRESH), bool)
        tp += int((b & s).sum()); nb += int(b.sum()); ns += int(s.sum())
    out = {'images': len(ids), 'threshold': THRESH, 'recall': tp / ns, 'precision': tp / nb}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)


if __name__ == '__main__':
    main()
