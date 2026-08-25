"""PatchCamelyon (PCam) with a MedMNIST-compatible interface.

Why this exists
---------------
`benchmark_train.py` builds its dataloaders around the MedMNIST API:

    info       = INFO[data_flag]                       # metadata dict
    DataClass  = getattr(medmnist, info['python_class'])
    dataset    = DataClass(split=..., transform=..., download=...)

and the training loop does `labels.to(device).squeeze(1)`, which assumes labels
come back shaped (1,) as MedMNIST returns them. `torchvision.datasets.PCAM`
returns a plain int instead, so `squeeze(1)` would raise IndexError.

This module supplies:
  * PCAM_INFO      -- a metadata dict shaped like an entry of medmnist.INFO
  * PCamDataset    -- a wrapper exposing the MedMNIST constructor signature and
                      returning labels as a LongTensor of shape (1,)

so PCam can be dropped into the existing pipeline with a ~6-line change (see
`patch_instructions()` below) and no edits to the training loop itself.

Dataset facts (Veeling et al., "Rotation Equivariant CNNs for Digital
Pathology", MICCAI 2018):
  * 327,680 patches from Camelyon16, 96x96 RGB, 10x magnification
  * 80 / 10 / 10 train/val/test split (262,144 / 32,768 / 32,768)
  * binary label: tumour present in the central 32x32 region
  * published baselines are parameter-comparable to ours: P4M-DenseNet 119K
    params at 89.8% acc / 96.3 AUC; DenseNet 128K at 87.6% / 95.5

Requirements
------------
    pip install h5py gdown

  * `h5py` -- PCam ships as HDF5; torchvision raises RuntimeError without it,
    and it checks this BEFORE attempting any download.
  * `gdown` -- torchvision fetches PCam from Google Drive.

Download note
-------------
Google Drive is slow or blocked from some networks; if the download stalls,
fetch from HuggingFace (`1aurent/PatchCamelyon`) instead and point `root` at
the extracted files.

Disk: the full dataset is ~8 GB; set `PCAM_ROOT` to keep it off a small
system disk.
"""
import os
import torch

# Mirrors the shape of an entry in medmnist.INFO, so callers can treat it the same.
PCAM_INFO = {
    'python_class': 'PCamDataset',
    'task': 'binary-class',
    'n_channels': 3,
    'label': {'0': 'no_tumor', '1': 'tumor'},
    'n_samples': {'train': 262144, 'val': 32768, 'test': 32768},
}

# Where PCam lives. Override with PCAM_ROOT -- useful on machines where the
# default lands on the small system disk instead of the data disk, e.g.
#   export PCAM_ROOT=/path/to/pcam_cache
DEFAULT_ROOT = os.environ.get('PCAM_ROOT', os.path.expanduser('~/.pcam'))


_HF_REPO = '1aurent/PatchCamelyon'
# The repo names its validation split "valid"; other mirrors/versions use
# "validation". Try candidates in order rather than hard-coding one.
_HF_SPLIT_CANDIDATES = {
    'train': ['train'],
    'val': ['valid', 'validation', 'val'],
    'test': ['test'],
}
_HF_SPLIT = {'train': 'train', 'val': 'valid', 'test': 'test'}
# torchvision stores {root}/pcam/camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5
_TV_SPLIT = {'train': 'train', 'val': 'valid', 'test': 'test'}


def _h5_files_present(root, split):
    """True if torchvision's expected .h5 pair already exists locally."""
    s = _TV_SPLIT[split]
    base = os.path.join(root, 'pcam')
    return all(os.path.exists(os.path.join(
        base, f'camelyonpatch_level_2_split_{s}_{xy}.h5')) for xy in ('x', 'y'))


class PCamHFDataset(torch.utils.data.Dataset):
    """PCam via the HuggingFace mirror (`1aurent/PatchCamelyon`).

    Use when Google Drive is unreachable (e.g. from mainland China), which is
    torchvision's only download source. Splits are 262,144 / 32,768 / 32,768,
    matching the official release.

        pip install datasets
        export HF_ENDPOINT=https://hf-mirror.com     # if huggingface.co is slow

    Set `cache_dir` to a data-disk path -- the dataset is ~8 GB.
    """

    def __init__(self, split='train', transform=None, cache_dir=None, **kwargs):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace backend needs: pip install datasets"
            ) from None
        if split not in _HF_SPLIT_CANDIDATES:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.ds, last_err = None, None
        for name in _HF_SPLIT_CANDIDATES[split]:
            try:
                self.ds = load_dataset(_HF_REPO, split=name,
                                       cache_dir=cache_dir)
                self.hf_split = name
                break
            except ValueError as e:
                last_err = e  # wrong split name for this repo; try the next
        if self.ds is None:
            raise ValueError(
                f"could not resolve split {split!r} in {_HF_REPO}; "
                f"tried {_HF_SPLIT_CANDIDATES[split]}. Last error: {last_err}"
            )
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        rec = self.ds[index]
        img = rec['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([int(rec['label'])], dtype=torch.long)


class PCamDataset(torch.utils.data.Dataset):
    """PCam with the MedMNIST constructor signature, backend auto-selected.

    Args mirror MedMNIST: split in {'train','val','test'}, transform, download.
    __getitem__ returns (image, LongTensor([label])) so the training loop's
    `labels.squeeze(1)` behaves exactly as for MedMNIST datasets.

    backend:
      'auto' (default) -- use torchvision if the .h5 files are already on disk,
                          otherwise fall back to the HuggingFace mirror.
      'torchvision'    -- force torchvision (downloads from Google Drive).
      'hf'             -- force the HuggingFace mirror.

    Override with the PCAM_BACKEND environment variable.
    """

    def __init__(self, split='train', transform=None, download=False,
                 root=DEFAULT_ROOT, backend=None, cache_dir=None, **kwargs):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        backend = backend or os.environ.get('PCAM_BACKEND', 'auto')
        os.makedirs(root, exist_ok=True)

        if backend == 'auto':
            backend = 'torchvision' if _h5_files_present(root, split) else 'hf'

        if backend == 'hf':
            self._impl = PCamHFDataset(
                split=split, transform=transform,
                cache_dir=cache_dir or os.path.join(root, 'hf_cache'))
        elif backend == 'torchvision':
            try:
                import h5py  # noqa: F401
            except ImportError:
                raise ImportError(
                    "PCam .h5 files need h5py: pip install h5py"
                ) from None
            from torchvision.datasets import PCAM
            self._impl = _TVWrapper(
                PCAM(root=root, split=split, transform=transform,
                     download=download))
        else:
            raise ValueError(f"backend must be auto/torchvision/hf, got {backend!r}")
        self.backend = backend
        self.split = split

    def __len__(self):
        return len(self._impl)

    def __getitem__(self, index):
        return self._impl[index]


class _TVWrapper(torch.utils.data.Dataset):
    """Adapts torchvision PCAM's int label to MedMNIST's (1,) tensor."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, target = self.ds[index]
        return img, torch.tensor([int(target)], dtype=torch.long)


class SubsetEpochDataset(torch.utils.data.Dataset):
    """Fixed-size random view of a dataset, for cheaper 'epochs'.

    Veeling et al. define an epoch as 312 batches of 64 (~20k images) rather
    than a full pass over 245k. Wrapping the training set in this reproduces
    that convention and makes PCam runs affordable:

        train = SubsetEpochDataset(PCamDataset('train', ...), 19968, seed=0)

    Note the subset is fixed at construction, so every epoch sees the same
    images. For genuine resampling per epoch, prefer a RandomSampler with
    `num_samples` on the DataLoader instead.
    """

    def __init__(self, base, n, seed=0):
        self.base = base
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(len(base), generator=g)[:n].tolist()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def patch_instructions():
    """Print the minimal change needed in benchmark_train.py / benchmark_eval.py."""
    print(__doc__)
    print("""
Minimal integration (benchmark_train.py, in the dataloader setup around L105):

    # --- replace ---
    info = INFO[data_flag]
    ...
    DataClass = getattr(medmnist, info['python_class'])

    # --- with ---
    if data_flag == 'pcam':
        from pcam_dataset import PCAM_INFO, PCamDataset
        info, DataClass = PCAM_INFO, PCamDataset
    else:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])

Everything downstream is unchanged: `info['n_channels']` (3) and
`len(info['label'])` (2) are read the same way, the DataClass call signature
matches, and labels arrive shaped (1,) so `squeeze(1)` still applies.

Apply the same 4-line branch in benchmark_eval.py (it does the same INFO lookup).

Checks worth running before a long job:
  * python3 -c "from pcam_dataset import PCamDataset; d=PCamDataset('val', download=True); print(len(d), d[0][0].size, d[0][1])"
  * python3 check_param_counts.py --data_flag pcam --img_channels 3   # needs the branch above
  * confirm SRE-Conv's pooling chain stays even: 96 -> 48 -> 24 -> 12 -> 6 (fine);
    avoid configurations that pool an odd size, which breaks D4 equivariance.
""")


if __name__ == '__main__':
    patch_instructions()
