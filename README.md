# Do Attention-Based MIML Models Know Where Their Labels Are?

Code, benchmarks and stored results for

> Syed Umaid Ahmed, Muhammad Atif Tahir, Muhammad Waqas.
> *Do Attention-Based Multi-Instance Multi-Label Models Know Where Their Labels
> Are? An Attribution Protocol, a Ceiling Theorem, and a Segmentation-Derived Benchmark.*
> Manuscript under review, 2026.

**Release v1.1** accompanies the revised manuscript. Every table and figure is
regenerated from the JSON result files in `results/instance_level_r1/`, and
`analysis/verify_paper_claims_r1.py` re-derives every numerical claim of the
text from them (97 checks). Release v1.0 and `results/instance_level/` are the
submitted version, kept for the record; `paper_attribution/REVISION_CHANGELOG.md`
lists what changed and why.

## What is here

| Path | Contents |
|---|---|
| `miml_clam/` | Model package, including **hybrid attention pooling (HAP)** and `metrics/instance_metrics.py`: the attribution protocol (P@k, chance rate, attribution AUC, pooled AUC, the label-agnostic ceiling and the pooled-AUC bounds). |
| `baselines/` | AB-MIL, Gated AB-MIL, TransMIL (lite), DSMIL in its original form (`dsmil_orig.py`: one query network shared with the critical instance), DSMIL-sep (`dsmil.py`: separate query and key projections), ML-Decoder, Q2L; one forward contract, each exposing the attention tensor it pools with. |
| `datasets/` | VOC-Patch construction. `build_voc_patch_r1.py` builds the revision's feature files (whole patch resized to 224×224, no centre crop; 4×4 and 8×8 grids; per-patch overlap fractions). Older builders and the HJA Bird Song conversion are kept. |
| `run_instance_attribution.py` | Trains each (architecture, benchmark) cell and, on the same trained models, scores attribution, the additive read-out of linear-head models, a deletion test, and per-model bag metrics; saves per-model arrays. |
| `gpu_run_pack/cell_queue.py` | Restart-safe parallel queue of cells (one process per cell). |
| `results/instance_level_r1/` | The 80 cells of the revision (2,480 trained models, all on CPU): 8 architectures × 6 benchmarks, the HAP ablation rows, the 8×8 grid and the cropped-feature variant of VOC-Patch-Seg-ML. |
| `results/instance_level_r1_colabA/` | Six VOC-Patch-Seg cells retrained on a CUDA GPU as a reproduction check; not used in any result. |
| `results/instance_level/` | The 42 cells of the submitted version (v1.0). |
| `analysis/` | Generators for every table and figure, `attribution_stats.py` (fold-level corrected resampled t-test, Holm), `attribution_extras.py` (post-hoc analyses that re-score every saved model), and the verifiers. |
| `data/MIML-SELFVIT/` | The four classical benchmarks, and for VOC-Patch the fold permutations and patch labels with overlap fractions. Feature files are release assets. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch ≥ 2.0 (CPU, CUDA or Apple MPS). Every reported model was trained on CPU.

## Data

The classical benchmarks ship in `data/MIML-SELFVIT/`. Download the VOC-Patch
feature files from the releases into `data/MIML-SELFVIT/`:

| Asset | Release | Size | Used for |
|---|---|---|---|
| `voc2012seg_nc.mat` | v1.1 | 298 MB | VOC-Patch-Seg (no crop) |
| `voc2012segml_nc.mat` | v1.1 | 87 MB | VOC-Patch-Seg-ML (no crop) |
| `voc2012segml_g8.mat` | v1.1 | 346 MB | VOC-Patch-Seg-ML, 8×8 grid |
| `voc2012segml.mat` | v1.0 | 87 MB | cropped features (`voc2012segml_crop*` are symlinks to it) |
| `voc2012seg.mat` | v1.0 | 298 MB | cropped features (checked by the verifier) |
| `instance_level_r1_arrays.zip` | v1.1 | 239 MB | per-model attention, deletion and prediction arrays; unzip into `results/instance_level_r1/arrays/` |

The arrays are needed only by `analysis/attribution_extras.py` (miss
decomposition, multi-label subsets, τ sweep, deletion against present labels);
everything else reads the JSON files. To rebuild the feature files from
PASCAL VOC 2012 instead: `python3 -m datasets.build_voc_patch_r1 --grid 4` and
`--grid 8 --subset segml` (needs the VOC 2012 images; see the docstring). Set `MIML_DATA_DIR` to use a data directory elsewhere.

## Reproducing the paper

No training; a few minutes:

```bash
M=attnmil,gated_attn,transmil_lite,dsmil_orig,dsmil,ml_decoder,q2l,hap_only
T=paper_attribution/tables
python3 analysis/attribution_stats.py --results results/instance_level_r1 --methods $M --out $T/attribution_stats_r1.json
python3 analysis/attribution_stats.py --results results/instance_level_r1 --methods $M --metric BagAveragePrecision --out $T/bagap_stats_r1.json
python3 analysis/attribution_extras.py --out $T/attribution_extras.json      # needs the arrays
python3 analysis/make_attribution_tables.py --rev r1 --outdir $T
python3 analysis/make_revision_tables.py --outdir $T
python3 analysis/make_setup_tables.py --rev r1 --stats $T/attribution_stats_r1.json --outdir $T
python3 analysis/make_attribution_figures.py --rev r1 --outdir paper_attribution/figures
python3 analysis/make_overview_figure.py --rev r1 --outdir paper_attribution/figures
python3 analysis/verify_paper_claims_r1.py        # must print "all checks passed"
cd paper_attribution && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Retrain (cells are idempotent; `--rerun` overwrites):

```bash
# classical benchmarks: 5 folds x 7 seeds = 35 models per cell (defaults)
python3 run_instance_attribution.py --methods $M \
    --datasets letter_frost,letter_carroll,mscv2,hja_birdsong --device cpu --outdir results/instance_level_r1
# VOC-Patch: 10 folds x 3 seeds, batch 8 (defaults for the _nc datasets)
python3 run_instance_attribution.py --methods $M --datasets voc2012seg_nc,voc2012segml_nc --device cpu --outdir results/instance_level_r1
# 8x8 grid: 10 folds, one model per fold
python3 run_instance_attribution.py --methods $M --datasets voc2012segml_g8 --no-ensemble --device cpu --outdir results/instance_level_r1
# several cells in parallel
python3 gpu_run_pack/cell_queue.py --methods $M --datasets voc2012seg_nc --outdir results/instance_level_r1 --workers 3 --threads 2
```

HAP ablation rows: `hap_soft_only`, `hap_hard_only`, `hap_k1`, `hap_k5`.

## The protocol on your own model

`miml_clam/metrics/instance_metrics.py` needs only NumPy:

```python
from miml_clam.metrics.instance_metrics import all_instance_metrics
m = all_instance_metrics(attn, inst_labels, bag_labels, mask=mask)
# attn        (B, L, N)  per-label attention over instances (need not be normalised)
# inst_labels (B, N, L)  per-instance ground truth
# bag_labels  (B, L)     bag ground truth
# mask        (B, N)     True at padded positions
```

`label_agnostic_ceiling` and `label_agnostic_auc_bounds` compute, for any
dataset with instance labels and before any model is trained, the bounds on
attribution precision and pooled AUC that no label-agnostic attention can
exceed.

## License

MIT. The datasets in `data/` retain the terms of their original distributors
(MSRC v2: Microsoft Research; Letter Frost / Letter Carroll: as distributed with
the MIML literature; VOC-Patch is derived from PASCAL VOC 2012).
