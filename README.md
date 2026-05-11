# UGSRPE: Uncertainty-Guided Self-Referring Prompt Evolution for One-Shot Polyp Segmentation with SAM

This repository contains the official implementation of **UGSRPE**, a training-free framework for one-shot polyp segmentation that combines frozen DINOv2 and SAM2 with a hybrid correlation-based prior, scale-cascaded fusion, uncertainty-guided multi-prompt evolution, and adaptive recovery mechanisms.

Given a single annotated support image, UGSRPE automatically segments polyps in unseen query images without any training or fine-tuning.

For a step-by-step explanation of how the framework works, see [FRAMEWORK_WALKTHROUGH.md](./FRAMEWORK_WALKTHROUGH.md).

## Results

UGSRPE outperforms the previous state-of-the-art one-shot method OP-SAM (ICCV 2025) on every IoU comparison across four public benchmarks, with fixed hyperparameters and no dataset-specific tuning.

| Dataset | Queries | Mean IoU (%) | Mean Dice (%) | OP-SAM IoU (%) |
|---|---|---|---|---|
| Kvasir-SEG | 999 | **77.09** | 83.91 | 76.93 |
| CVC-ClinicDB | 611 | **69.75** | 78.16 | 67.75 |
| CVC-ColonDB | 379 | **64.62** | 72.07 | 60.81 |
| PolypGen C1 | 251 | **70.22** | 77.65 | 68.68 |
| PolypGen C2 | 269 | **72.96** | 79.73 | 68.65 |
| PolypGen C3 | 455 | **74.73** | 81.65 | 71.44 |

PolypGen Centres 4–6 and sequence frames (1,710 additional images) are also evaluated in the corresponding notebook; see the paper for the full per-centre breakdown.

## Repository Contents

| Notebook | Dataset | Description |
|---|---|---|
| `ugsrpe-kvasir-seg.ipynb` | Kvasir-SEG | 999 query images |
| `ugsrpe-clinicdb.ipynb` | CVC-ClinicDB | 611 query images (PNG version) |
| `cvc-colondb.ipynb` | CVC-ColonDB | 379 query images |
| `ugsrpe_polypgen.ipynb` | PolypGen | 3,120 query images across six centres + sequence frames |

Each notebook is self-contained and follows the same 13-section structure: environment setup, model loading, dataset preprocessing, feature extraction, the five core modules (CPG, SPF, UGMPE, adaptive fallback, grid-proposal discovery), the full pipeline, evaluation metrics, support selection, prior diagnostics, and full-dataset evaluation.

## Method Overview

The pipeline consists of five sequential modules:

1. **Correlation-based Prior Generation (CPG)** — Hybrid cross-correlation with prototype-residual gating and self-correlation refinement.
2. **Scale-Cascaded Prior Fusion (SPF)** — Three-scale CPG priors fused via confidence-weighted reverse transfer.
3. **Uncertainty-Guided Multi-Prompt Evolution (UGMPE)** — Seven diverse prompts per iteration, inter-prompt agreement as uncertainty, self-referring prior updates.
4. **Adaptive Fallback Recovery** — Three-tier strategy that progressively relaxes thresholds to recover from empty predictions.
5. **Grid-Proposal Discovery** — SAM as an object discovery engine when the prior-based pipeline drifts, with top-K frequency voting.

See the paper for full methodology details.

## Environment

All experiments were run on **Kaggle T4 ×2 GPUs with Internet enabled**. The notebooks expect this environment and automatically download the SAM2 checkpoint on first run.

### Models

- **DINOv2 ViT-L/14** (frozen) — loaded via `torch.hub`
- **SAM2 hiera-large** (frozen) — `sam2.1_hiera_large.pt` from the official Meta release

### Hyperparameters

All hyperparameters are fixed across all four datasets:

| Parameter | Value | Module |
|---|---|---|
| DINO image size | 560 × 560 | Feature extraction |
| SAM2 image size | 1024 × 1024 | Mask generation |
| Residual gate weight (w) | 0.5 | CPG |
| Self-correlation temperature (τ) | 0.1 | CPG |
| Self-correlation rounds (ρ) | 1 | CPG |
| Top percentile | 30% | CPG |
| SPF scales | original, ×1.5, ×0.5 | SPF |
| Number of prompts | 7 | UGMPE |
| Max iterations | 3 | UGMPE |
| Prior update momentum (α) | 0.70 | UGMPE |
| Convergence tolerance | 0.10 | UGMPE |
| Grid density | 8 × 8 | Grid fallback |
| Top-K proposals | 10 | Grid fallback |
| Area filter | 0.3%–35% | Grid fallback |

## How to Run

1. **Open the notebook on Kaggle** with a T4 ×2 GPU accelerator and Internet enabled.
2. **Attach the corresponding dataset** (see Datasets section below).
3. **Run all cells in order.** The notebook will:
   - Install SAM2 from the official GitHub repo.
   - Download the SAM2 checkpoint (~860 MB).
   - Load DINOv2 via `torch.hub`.
   - Automatically select a support image via prototype search (sample size 1000).
   - Run a 50-image quick validation.
   - Run the full-dataset evaluation (approximately 17 minutes for ColonDB, 23 minutes for ClinicDB, 41 minutes for Kvasir-SEG, 142 minutes for PolypGen).
4. **Results** are saved as CSV files alongside the notebook, and a summary table is printed at the end.

The notebooks can also be run locally with appropriate adjustments to the dataset paths in cell 8 of each notebook.

## Datasets

The notebooks expect Kaggle dataset paths under `/kaggle/input/`. Specific versions used:

- **Kvasir-SEG** — auto-detected from any Kaggle dataset with `images/` and `masks/` subdirectories.
- **CVC-ClinicDB** — `balraj98/cvcclinicdb` (PNG version). The PNG format is required to avoid colour-channel corruption from TIFF loading.
- **CVC-ColonDB** — `longvil/cvc-colondb` (or any Kaggle upload with `images/` and `masks/` subdirectories).
- **PolypGen** — Dataset Ninja JSON-annotation format with `ds/img/` and `ds/ann/` subdirectories. Masks are decoded on the fly from base64-encoded bitmap annotations.

Each notebook's dataset-loading cell prints the resolved paths and image counts; verify these match expectations before running the full evaluation.

## Citation

If you use this work, please cite:

```bibtex
@article{roy2026ugsrpe,
  title   = {Uncertainty-Guided Self-Referring Prompt Evolution for One-Shot Polyp Segmentation with SAM},
  author  = {Roy, Arnab Satyam and Mim, Dolon Akter and Mridha, Muhammad Firoz},
  year    = {2026}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors:

- Arnab Satyam Roy — arnabsroy9@gmail.com
- Dolon Akter Mim — d.mim.edu@gmail.com
- Muhammad Firoz Mridha — firoz.mridha@aiub.edu

American International University–Bangladesh (AIUB), Dhaka, Bangladesh.

## Acknowledgements

This work builds on the publicly released DINOv2 (Meta AI) and SAM2 (Meta AI) foundation models, and uses the publicly available Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, and PolypGen datasets.
