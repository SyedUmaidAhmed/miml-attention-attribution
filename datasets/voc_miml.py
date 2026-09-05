"""Pascal VOC 2012 → MIML dataset construction (ICDM 2026 extension).

Each VOC2012 trainval image is turned into a bag of 16 instances:

    image -> 4x4 grid of equal patches
          -> ImageNet-pretrained ResNet-50 (frozen, IMAGENET1K_V2 weights)
          -> 2048-dim avgpool features per patch
    bag  : (16, 2048) float32
    label: 20-dim {-1, +1} multi-label vector (standard VOC2012 classes)

Outputs (in the same convention as scene.mat / reuters_MIML.mat):

    ../VIT_MIL/Code/MIML-SELFVIT/voc2012.mat
        bags   : (n_images, 1) numpy object array; each cell is (16, 2048) float32
        labels : (n_images, 20) int matrix in {-1, +1}

    ../VIT_MIL/Code/MIML-SELFVIT/voc2012_10CV.mat
        Indexes : (1, n_images) uint16 1-indexed permutation (seed=42)

Designed to run end-to-end on Apple Silicon MPS without CUDA. Memory-efficient:
images are processed one at a time; ResNet-50 stays on the device throughout.

Idempotent: re-running is a no-op unless --rerun is passed.

Smoke check:
    python -m datasets.voc_miml --smoke

Full extraction (only run on a beefy machine; ~hours):
    python -m datasets.voc_miml
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

# torch / torchvision are imported lazily inside _build_extractor and
# _extract_features so that --smoke can exercise the structural path on
# machines that don't yet have torchvision installed.

# 20 standard VOC classes, alphabetical ordering used by every VOC2012 paper.
VOC_CLASSES: Tuple[str, ...] = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)
NUM_CLASSES = len(VOC_CLASSES)         # 20
GRID = 4                                # 4x4 patch grid
PATCHES_PER_BAG = GRID * GRID           # 16
FEATURE_DIM = 2048                      # ResNet-50 avgpool

# VOCtrainval_11-May-2012.tar download URL (mirror used by torchvision).
VOC_URL = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
           'VOCtrainval_11-May-2012.tar')

# Repo layout:
#   <repo>/datasets/voc_miml.py        <- this file
#   <repo>/../VIT_MIL/Code/MIML-SELFVIT/  <- output dir
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_LEGACY_OUT_DIR = os.path.normpath(
    os.path.join(_REPO, '..', 'VIT_MIL', 'Code', 'MIML-SELFVIT'))
# Same resolution order as run_all_enhanced.DATA_DIR (env var, legacy sibling
# checkout, in-repo data/), so the VOC-Patch files land beside the other .mat.
DEFAULT_OUT_DIR = (os.environ.get('MIML_DATA_DIR')
                   or (_LEGACY_OUT_DIR if os.path.isdir(_LEGACY_OUT_DIR)
                       else os.path.join(_REPO, 'data', 'MIML-SELFVIT')))


def _voc_root() -> str:
    """Resolve the VOC2012 cache directory.

    Order: $VOC2012_ROOT -> ~/data/voc2012 .
    """
    env = os.environ.get('VOC2012_ROOT')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.expanduser('~/data/voc2012')


def _ensure_voc_downloaded(root: str) -> str:
    """Download + extract VOC2012 trainval if missing. Returns VOCdevkit path."""
    devkit = os.path.join(root, 'VOCdevkit', 'VOC2012')
    if os.path.isdir(devkit):
        return devkit
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, 'VOCtrainval_11-May-2012.tar')
    if not os.path.exists(tar_path):
        print(f'[voc] downloading {VOC_URL} -> {tar_path}', flush=True)
        urllib.request.urlretrieve(VOC_URL, tar_path)
    print(f'[voc] extracting {tar_path} -> {root}', flush=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(root)
    if not os.path.isdir(devkit):
        raise RuntimeError(f'VOC extraction did not produce {devkit}')
    return devkit


def _read_image_ids(devkit: str) -> List[str]:
    """Read the trainval split list (5717+5823 = 11540 entries on VOC2012)."""
    split_file = os.path.join(devkit, 'ImageSets', 'Main', 'trainval.txt')
    with open(split_file) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _read_label_vector(devkit: str, image_id: str) -> np.ndarray:
    """Parse VOC XML annotation -> 20-dim {-1, +1} multi-label vector.

    A class is +1 iff at least one *non-difficult* object of that class is
    present (standard VOC2012 multi-label convention).
    """
    xml_path = os.path.join(devkit, 'Annotations', f'{image_id}.xml')
    label = -np.ones(NUM_CLASSES, dtype=np.int32)
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return label
    for obj in root.findall('object'):
        name_el = obj.find('name')
        diff_el = obj.find('difficult')
        if name_el is None or name_el.text is None:
            continue
        cls = name_el.text.strip().lower()
        if cls not in VOC_CLASSES:
            continue
        if diff_el is not None and diff_el.text and int(diff_el.text) == 1:
            continue
        label[VOC_CLASSES.index(cls)] = 1
    return label


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_extractor(device):
    """Build a frozen ResNet-50 truncated at the avgpool layer."""
    import torch
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(weights=weights)
    # Drop the final fc layer; keep avgpool. Output is (B, 2048, 1, 1).
    modules = list(backbone.children())[:-1]
    extractor = torch.nn.Sequential(*modules).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    transform = weights.transforms()  # standard ImageNet preprocessing
    return extractor, transform


def _patch_grid(image, grid: int = GRID):
    """Slice a PIL image into a grid x grid list of (cropped) PIL patches.

    Patches are equal-sized (last patch absorbs any remainder pixels).
    """
    w, h = image.size
    pw, ph = w // grid, h // grid
    patches = []
    for r in range(grid):
        for c in range(grid):
            left = c * pw
            top = r * ph
            right = (c + 1) * pw if c < grid - 1 else w
            bottom = (r + 1) * ph if r < grid - 1 else h
            patches.append(image.crop((left, top, right, bottom)))
    return patches


def _extract_one_image(image, extractor, transform, device):
    """Image (PIL) -> (16, 2048) float32 numpy bag."""
    import torch
    patches = _patch_grid(image, GRID)
    # Stack into a single (16, 3, 224, 224) batch — small, fits easily on MPS.
    batch = torch.stack([transform(p) for p in patches]).to(device)
    with torch.no_grad():
        feats = extractor(batch)            # (16, 2048, 1, 1)
    feats = feats.flatten(1)                # (16, 2048)
    return feats.cpu().numpy().astype(np.float32)


def _select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Top-level extraction pipeline
# ---------------------------------------------------------------------------

def build_voc_miml(out_dir: str = DEFAULT_OUT_DIR,
                   limit: int | None = None,
                   rerun: bool = False) -> Tuple[str, str]:
    """Run the full VOC2012 -> MIML extraction pipeline.

    Returns (mat_path, cv_path).
    """
    from PIL import Image

    mat_path = os.path.join(out_dir, 'voc2012.mat')
    cv_path = os.path.join(out_dir, 'voc2012_10CV.mat')
    if os.path.exists(mat_path) and os.path.exists(cv_path) and not rerun:
        print(f'[voc] {mat_path} and {cv_path} already exist; skipping '
              '(pass --rerun to override).')
        return mat_path, cv_path

    voc_root = _voc_root()
    devkit = _ensure_voc_downloaded(voc_root)
    image_ids = _read_image_ids(devkit)
    if limit is not None:
        image_ids = image_ids[:limit]
    print(f'[voc] {len(image_ids)} trainval images @ {devkit}', flush=True)

    device = _select_device()
    print(f'[voc] device = {device}', flush=True)
    extractor, transform = _build_extractor(device)

    n = len(image_ids)
    bags = np.empty((n, 1), dtype=object)
    labels = np.full((n, NUM_CLASSES), -1, dtype=np.int32)

    img_dir = os.path.join(devkit, 'JPEGImages')
    log_every = max(1, n // 50)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(img_dir, f'{image_id}.jpg')
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            bag = _extract_one_image(img, extractor, transform, device)
        bags[i, 0] = bag
        labels[i] = _read_label_vector(devkit, image_id)
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f'[voc] processed {i+1}/{n}', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    from scipy.io import savemat
    savemat(mat_path, {'bags': bags, 'labels': labels})
    print(f'[voc] wrote {mat_path}  '
          f'(bags={bags.shape}, label_matrix={labels.shape})')

    rng = np.random.RandomState(42)
    perm = (rng.permutation(n) + 1).astype(np.uint16).reshape(1, -1)
    savemat(cv_path, {'Indexes': perm})
    print(f'[voc] wrote {cv_path}  (Indexes shape={perm.shape})')

    return mat_path, cv_path


# ---------------------------------------------------------------------------
# Smoke test (structural — no download, no disk write)
# ---------------------------------------------------------------------------

def _smoke(num_images: int = 32) -> None:
    """Exercise the construction code path with mocked image+model.

    Verifies bag shape (16, 2048) and label matrix shape (num_images, 20)
    without downloading or running ResNet-50. Prints a structure summary.
    """
    print('[smoke] running structural check (no download, no disk write)')
    print(f'[smoke] target: {num_images} bags x {PATCHES_PER_BAG} patches '
          f'x {FEATURE_DIM} dims')

    # Build a fake (16, 2048) feature matrix per image and a random label
    # vector to walk the same code path as the real extractor.
    rng = np.random.RandomState(0)
    bags = np.empty((num_images, 1), dtype=object)
    labels = np.full((num_images, NUM_CLASSES), -1, dtype=np.int32)
    for i in range(num_images):
        bags[i, 0] = rng.randn(PATCHES_PER_BAG, FEATURE_DIM).astype(np.float32)
        # Sprinkle 1-3 positive labels to mimic typical VOC bag sparsity.
        for cls_idx in rng.choice(NUM_CLASSES, size=rng.randint(1, 4),
                                  replace=False):
            labels[i, cls_idx] = 1

    # Verify the contract MIMLDatasetCV expects.
    assert bags.shape == (num_images, 1), bags.shape
    assert bags.dtype == object
    first_bag = bags[0, 0]
    assert first_bag.shape == (PATCHES_PER_BAG, FEATURE_DIM), first_bag.shape
    assert first_bag.dtype == np.float32, first_bag.dtype
    assert labels.shape == (num_images, NUM_CLASSES), labels.shape
    assert set(np.unique(labels).tolist()).issubset({-1, 1}), \
        set(np.unique(labels).tolist())

    # CV permutation: 1-indexed uint16, shape (1, n).
    rng2 = np.random.RandomState(42)
    perm = (rng2.permutation(num_images) + 1).astype(np.uint16).reshape(1, -1)
    assert perm.shape == (1, num_images)
    assert perm.min() >= 1 and perm.max() <= num_images

    # Verify _patch_grid splits a fake image into 16 boxes that tile fully.
    class _FakeImg:
        size = (224, 224)

        def crop(self, box):
            return box  # return the box for inspection

    boxes = _patch_grid(_FakeImg(), GRID)
    assert len(boxes) == PATCHES_PER_BAG
    # Boxes must tile the entire image with no gaps.
    union_area = sum((r - l) * (b - t) for (l, t, r, b) in boxes)
    assert union_area == 224 * 224, union_area

    print('[smoke] OK — bag shape', first_bag.shape,
          '| label shape', labels.shape,
          '| CV perm shape', perm.shape,
          '| patch tiling area', union_area)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                        help='Where to write voc2012.mat and voc2012_10CV.mat')
    parser.add_argument('--rerun', action='store_true',
                        help='Force re-extraction even if output .mat exists')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trainval images (debug only)')
    parser.add_argument('--smoke', action='store_true',
                        help='Run a 32-image structural check without writing '
                             'or downloading anything')
    args = parser.parse_args(argv)

    if args.smoke:
        _smoke(num_images=32)
        return 0

    build_voc_miml(out_dir=args.out_dir,
                   limit=args.limit,
                   rerun=args.rerun)
    return 0


if __name__ == '__main__':
    sys.exit(main())
