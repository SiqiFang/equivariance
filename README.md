# Moment Kernels

## Repository structure

```
.
├── moment_kernels.py                      # Core equivariant building blocks (shared library)
├── requirements.txt                       # Python dependencies
├── classification/                        # 2D classification: PCam + MedMNIST (Table 1, Appendix E)
│   ├── benchmark_train.py                 # Train any classification model
│   ├── pcam_dataset.py                    # PatchCamelyon loader (MedMNIST-shaped interface)
│   ├── benchmark_eval.py                  # 8-orientation worst-case eval / rotation sweep
│   ├── train_sreconv_published.sh         # Reproduce published SRE-ResNet18 (3.99M)
│   ├── eval_sreconv_published.py          # Eval that checkpoint under our protocol
│   └── models/
│       ├── benchmark_models2.py           # Model zoo: VanillaCNN, ESCNN, SRE-Conv,
│       │                                  #   MatrixMomentOriginal, ablation variants
│       └── E2CNN_models.py                # ResNet-18 variants
└── registration/                           # 3D ADNI affine regression (Table 2, Appendices F–I)
    ├── mri_train.py                       # Training
    ├── mri_eval.py                        # 48-orientation evaluation (rot/refl split, ED2, WC)
    ├── mri_common.py                      # MODELS dict + apply_affine + predict_affine
    └── models/
        ├── escnn3d.py                     # ESCNN 3D variants (A–H from Appendix I)
        └── mri_synthmorph.py              # SynthMorph-style encoder (default 5.4M + matched 175k)
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt

# SRE-Conv baseline only: the kernel-substitution rows need the PyPI package,
# and the published SRE-ResNet18 script clones the authors' repository itself.
pip install SRE-Conv
```

The classification scripts expect a CUDA-capable GPU (RTX 4090 or equivalent for the timings reported in §4.1).

## Running the code

`moment_kernels.py` (the shared equivariant-layer library) lives at the repository root and is imported by every other script via a `sys.path` insertion of the parent directory. **Run all commands from the repository root** so this resolves correctly. For example:

```bash
cd /path/to/this/release
python classification/benchmark_train.py [args...]
```

Equivalently, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

All scripts take `--help` for the full argument list. The commands below show the configurations for each experiment; per-augmentation and per-model variants follow the same pattern (just change `--augment` and `--model`).

Trained checkpoints land under `results/<dataset>/run<N>/`, where `--run` is just an integer that names the output directory (default `0`).

## Reproducing the paper results

### Experiment 1: 2D classification — PatchCamelyon (Table 1, primary benchmark)

PCam auto-downloads on first run (via the HuggingFace `1aurent/PatchCamelyon` mirror,
falling back to the original Google Drive h5 files). Optionally set `PCAM_ROOT` and
`HF_HOME` to keep the ~8 GB dataset and cache off the system disk.

**Moment kernel** (Table 1 headline row — 60 epochs, cosine LR, batch 128, no augmentation):

```bash
python classification/benchmark_train.py \
    --model MatrixMomentOriginal --data_flag pcam \
    --n_channels 4 --num_layers 8 --epochs 60 --batch_size 128 \
    --lr 1e-3 --lr_schedule cosine --augment none \
    --no_normalize --save_on_best auc

python classification/benchmark_eval.py \
    --model MatrixMomentOriginal --data_flag pcam \
    --n_channels 4 --num_layers 8 --no_normalize --mode standard \
    --checkpoint results/pcam/run0/best_auc_MatrixMomentOriginal_pcam_augnone_run0.pt
```

**Baselines** (same protocol; initial LR 1e-4, their protocol value):

```bash
# Standard CNN (196k) — also run with --augment continuous for the second Table 1 row
python classification/benchmark_train.py \
    --model VanillaCNNOriginal --data_flag pcam \
    --n_channels 13 --num_layers 8 --epochs 60 --batch_size 128 \
    --lr 1e-4 --lr_schedule cosine --augment none --no_normalize --save_on_best auc

# ESCNN (185k) with downsampling matched to the moment kernel
python classification/benchmark_train.py \
    --model TrivialIrrepMatrixECNN --data_flag pcam \
    --n_channels 13 --num_layers 8 --epochs 60 --batch_size 128 \
    --lr 1e-4 --lr_schedule cosine --augment none --escnn_pool --save_on_best auc

# SRE-Conv (204k)
python classification/benchmark_train.py \
    --model SREVanillaCNN --data_flag pcam \
    --n_channels 23 --num_layers 8 --epochs 60 --batch_size 128 \
    --lr 1e-4 --lr_schedule cosine --augment none --no_normalize --save_on_best auc
```

At 96×96 input all downsampled feature maps remain even-sized, so no `--resize` is
needed on PCam (the resize is a DermaMNIST-only fix, see below).

### Experiment 1b: 2D classification — DermaMNIST and BloodMNIST (Appendix E)

**Moment kernel** (the proposed model — main moment-kernel row of Table 4, Appendix E.1, DermaMNIST classification):

```bash
python classification/benchmark_train.py \
    --model MatrixMomentOriginal --data_flag dermamnist \
    --n_channels 4 --num_layers 8 --epochs 1000 \
    --augment continuous --no_normalize

python classification/benchmark_eval.py \
    --model MatrixMomentOriginal --data_flag dermamnist \
    --n_channels 4 --num_layers 8 --no_normalize --mode standard \
    --checkpoint results/dermamnist/run0/best_auc_MatrixMomentOriginal_dermamnist_augcontinuous_run0.pt
```

**SRE-Conv kernel at our 8-layer 210k backbone** — two equivalent configurations, both giving exact $D_4$ equivariance (see Appendix E.3 for ablations):

```bash
# Configuration A: resize input 28→32 so all intermediate spatial sizes are even
python classification/benchmark_train.py \
    --model SREVanillaCNN --data_flag dermamnist \
    --n_channels 23 --num_layers 8 --epochs 1000 \
    --augment none --no_normalize --resize 32

# Configuration B: keep native 28×28; replace AvgPool2d with parity-aware mk.Downsample
python classification/benchmark_train.py \
    --model SREVanillaCNN --data_flag dermamnist \
    --n_channels 23 --num_layers 8 --epochs 1000 \
    --augment none --no_normalize --mk_downsample
```

**Standard CNN baseline** (parameter-matched at 200k):

```bash
python classification/benchmark_train.py \
    --model VanillaCNNOriginal --data_flag dermamnist \
    --n_channels 13 --num_layers 8 --epochs 1000 \
    --augment continuous --no_normalize
```

**ESCNN baseline** (185k):

```bash
python classification/benchmark_train.py \
    --model TrivialIrrepMatrixECNN --data_flag dermamnist \
    --n_channels 13 --num_layers 8 --epochs 1000 --augment discrete
```

**Published SRE-ResNet18** (3.99M, Appendix E.3 bottom row):

```bash
bash classification/train_sreconv_published.sh
python classification/eval_sreconv_published.py \
    --checkpoint <path printed at the end of the train script>
```

For BloodMNIST (Appendix E.2), pass `--data_flag bloodmnist`. For other augmentation conditions, pass `--augment {none, discrete, continuous}`.

### Experiment 2: 3D affine registration (Table 2)

> **Note: the registration code can be examined for inspection but cannot be re-run end-to-end without access to the ADNI dataset and our preprocessing.**
>
> The paper reports (§3.2): *"We use 3,745 MRI volumes from the Alzheimer's Disease Neuroimaging Initiative (ADNI), aligned to an ICBM atlas [Fonov et al., 2009] using the pipeline of [Puglisi et al., 2024] and resampled to $80\times80\times80$ voxels."*
>
> ADNI is a controlled-access dataset: reviewers wishing to reproduce these numbers must register at [adni.loni.usc.edu](https://adni.loni.usc.edu/) for access, then run the preprocessing pipeline of Puglisi et al. (2024) to produce ICBM-aligned $80\times80\times80$ volumes. The registration scripts expect data in this format and require the path to be passed via `--data_dir`.
>
> **References:**
> - Fonov, V., et al. (2009). *Unbiased nonlinear average age-appropriate brain templates from birth to adulthood.* NeuroImage.
> - Puglisi, L., et al. (2024). *Enhancing spatiotemporal disease progression models via latent diffusion and prior knowledge.*

With ADNI data preprocessed and available, point `--data_dir` at the directory containing `adni_affine_processed/`. The headline commands are:

```bash
# Table 2 main rows — ADNI registration (200 epochs each)
python registration/mri_train.py --model equivariant --data_dir <path/to/adni_root> --epochs 200
python registration/mri_train.py --model standard    --data_dir <path/to/adni_root> --epochs 200
python registration/mri_train.py --model synthmorph  --data_dir <path/to/adni_root> --epochs 200

# ESCNN variants A–H: each is a combination of --gspace, --bn_type / --no_bn, --base_mult.
# See Appendix I for the full variant table. Example (variant F: no-BN, SO(3)):
python registration/mri_train.py --model escnn --data_dir <path/to/adni_root> \
    --gspace rot3dOnR3 --no_bn --epochs 200

python registration/mri_eval.py --model equivariant --data_dir <path/to/adni_root> --checkpoint <path>
```

## Datasets

- **PatchCamelyon** — auto-downloads on first run (HuggingFace mirror, Google Drive fallback); reviewers can fully reproduce Table 1.
- **MedMNIST (DermaMNIST, BloodMNIST)** — auto-downloads via the `medmnist` pip package on first run; reviewers can fully reproduce Appendix E.
- **ADNI (3D MRI)** — controlled-access; not redistributed. See note above for access and preprocessing details.
