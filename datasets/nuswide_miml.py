"""NUS-WIDE multi-label MIML construction.

NUS-WIDE has 269,648 images x 81 ground-truth concept labels — exactly the
high-label-count regime where HAP's per-label aggregation adaptivity should
matter most. We construct a MIML version by extracting a fixed 4x4 patch grid
of features per image with a frozen ImageNet-pretrained ResNet-50.

Output schema (matches `datasets/voc_miml.py` / standard MIML format):
  bags        : (n, 1) numpy object array of (16, 2048) float32 matrices
  labels      : (n, 81) int32 with values in {-1, +1}
  ('Indexes' in nuswide_10CV.mat: (1, n) uint16 1-indexed permutation)

Subset support:
  --max-images N   for a budgeted run (default: 50000 — sufficient for headline
                   table, fits comfortably on a 24 GB GPU in <3 hr).
                   Pass --all for the full 269k-image extraction.

Usage:
    python -m datasets.nuswide_miml --smoke
    python -m datasets.nuswide_miml --device cuda --max-images 50000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio


CANONICAL_NUSWIDE_ROOT = os.environ.get(
    "NUSWIDE_ROOT", os.path.expanduser("~/data/nus-wide")
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / ".." / "VIT_MIL" / "Code" / "MIML-SELFVIT"

NUSWIDE_NUM_LABELS = 81  # canonical "Concepts81" label set


def _patch_grid(image_size=224, grid=4):
    """Return list of (y0, x0, h, w) patch boxes covering an image_size x image_size frame."""
    s = image_size // grid
    patches = []
    for i in range(grid):
        for j in range(grid):
            patches.append((i * s, j * s, s, s))
    return patches


def _load_resnet50(device="cuda"):
    """Load a frozen ImageNet-pretrained ResNet-50, return penultimate-feature extractor."""
    import torch
    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    backbone = torchvision.models.resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()  # 2048-dim avgpool output
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, weights.transforms()


def _read_concept_labels(nuswide_root: Path):
    """Read NUS-WIDE 'Concepts81' labels.

    Expects either:
      A. AllLabels/AllLabels<concept>.txt files (one per concept, 0/1 lines).
      B. A single TrainTestLabels matrix.
    Returns a (n_images, 81) int32 ±1 matrix and a list of image filenames.
    """
    img_list_path = nuswide_root / "ImageList" / "Imagelist.txt"
    if not img_list_path.exists():
        raise FileNotFoundError(
            f"Imagelist.txt not found at {img_list_path}. "
            f"Set NUSWIDE_ROOT or pass --root to the NUS-WIDE folder."
        )
    with open(img_list_path) as f:
        files = [l.strip().replace("\\", "/") for l in f if l.strip()]

    concepts_dir = nuswide_root / "AllLabels"
    if not concepts_dir.exists():
        raise FileNotFoundError(f"AllLabels/ not found at {concepts_dir}")

    concepts_list = sorted(p.stem.replace("Labels_", "").lower()
                            for p in concepts_dir.glob("Labels_*.txt"))
    if len(concepts_list) != NUSWIDE_NUM_LABELS:
        warnings.warn(f"found {len(concepts_list)} concepts, expected {NUSWIDE_NUM_LABELS}")

    n = len(files)
    L = len(concepts_list)
    Y = np.zeros((n, L), dtype=np.int32) - 1
    for j, c in enumerate(concepts_list):
        path = concepts_dir / f"Labels_{c}.txt"
        with open(path) as f:
            col = np.fromiter((int(l.strip()) for l in f if l.strip()),
                              dtype=np.int32, count=n)
        # NUS-WIDE label files are 0/1; convert to {-1, +1}.
        Y[col == 1, j] = +1
    return files, concepts_list, Y


def extract_features(nuswide_root: Path, out_mat: Path,
                     max_images: int = 50000, device: str = "cuda",
                     image_size: int = 224, grid: int = 4):
    """Run frozen ResNet-50 over a 4x4 patch grid for every NUS-WIDE image."""
    import torch
    from PIL import Image

    files, concepts, Y = _read_concept_labels(nuswide_root)
    if max_images is not None and max_images < len(files):
        # deterministic seeded subsample
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(files))[:max_images]
        files = [files[i] for i in idx]
        Y = Y[idx]
    n = len(files)
    print(f"[nuswide] extracting features for {n} images x {grid * grid} patches "
          f"-> ({n}, {grid*grid}, 2048)", flush=True)

    backbone, transform = _load_resnet50(device=device)
    patches = _patch_grid(image_size, grid)

    # Storage as object array of (16, 2048) float32 matrices to match the
    # standard MIML schema consumed by miml_clam.data.miml_dataset_cv.
    bags = np.empty((n, 1), dtype=object)
    images_root = nuswide_root / "Flickr"
    if not images_root.exists():
        # Some distributions store images under "images/"
        images_root = nuswide_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(
            f"NUS-WIDE images not found under {nuswide_root}/{{Flickr,images}}/")

    batch_size = 16  # patch batch
    for i, fname in enumerate(files):
        path = images_root / fname
        try:
            img = Image.open(path).convert("RGB").resize((image_size, image_size))
        except Exception as e:
            warnings.warn(f"skipping {fname}: {e}")
            bags[i, 0] = np.zeros((grid * grid, 2048), dtype=np.float32)
            continue
        # Crop the 16 patches and stack as (16, 3, S, S).
        arr = np.array(img)
        crops = []
        for (y0, x0, h, w) in patches:
            patch = arr[y0:y0 + h, x0:x0 + w]
            crops.append(transform(Image.fromarray(patch)))
        x = torch.stack(crops, dim=0).to(device)  # (16, 3, S, S)
        with torch.no_grad():
            f = backbone(x)                       # (16, 2048)
        bags[i, 0] = f.detach().cpu().float().numpy()
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}  ({100*(i+1)/n:.1f}%)", flush=True)

    out_mat.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_mat), {"bags": bags, "labels": Y.astype(np.int32)})
    print(f"[nuswide] wrote {out_mat}")

    # 10-fold CV permutation (1-indexed, shape (1, n)) to match existing convention.
    rng = np.random.RandomState(42)
    perm = rng.permutation(n).astype(np.uint16) + 1
    cv_path = out_mat.parent / "nuswide_10CV.mat"
    sio.savemat(str(cv_path), {"Indexes": perm.reshape(1, -1)})
    print(f"[nuswide] wrote {cv_path}")


def smoke():
    """Structural check — no download, no torch import, no disk I/O."""
    n = 32
    grid = 4
    patches = _patch_grid(224, grid)
    assert len(patches) == grid * grid
    area = sum(h * w for (_, _, h, w) in patches)
    bags = np.empty((n, 1), dtype=object)
    for i in range(n):
        bags[i, 0] = np.zeros((grid * grid, 2048), dtype=np.float32)
    Y = (np.random.RandomState(0).randint(0, 2, size=(n, NUSWIDE_NUM_LABELS)) * 2 - 1
         ).astype(np.int32)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n).astype(np.uint16) + 1
    print(f"[smoke] OK -- bag shape {bags[0,0].shape} | label shape {Y.shape} | "
          f"CV perm shape {(1, n)} | patch tiling area {area}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="structural check only; no download/torch/disk")
    parser.add_argument("--root", type=str, default=CANONICAL_NUSWIDE_ROOT,
                        help="NUS-WIDE root (must contain ImageList/, AllLabels/, "
                             "Flickr/ or images/ subdirs)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT / "nuswide.mat"))
    parser.add_argument("--max-images", type=int, default=50000,
                        help="cap on number of images (--all to use full set)")
    parser.add_argument("--all", action="store_true",
                        help="extract ALL ~270k images (slow; ~5-8 hr on a 24 GB GPU)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--rerun", action="store_true",
                        help="re-extract even if the .mat already exists")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return

    out_mat = Path(args.out).resolve()
    if out_mat.exists() and not args.rerun:
        print(f"[nuswide] {out_mat} already exists; pass --rerun to overwrite.")
        return

    cap = None if args.all else args.max_images
    extract_features(Path(args.root).expanduser(), out_mat,
                     max_images=cap, device=args.device)


if __name__ == "__main__":
    main()
