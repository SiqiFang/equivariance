"""
Shared dataset, models, and helpers for mri_train.py and mri_eval.py.
"""
import os, sys
import numpy as np
import torch
import nibabel as nib
from glob import glob
from itertools import permutations, product
from os.path import join, basename
from scipy.linalg import expm

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir   = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)                       # for moment_kernels
sys.path.insert(0, script_dir)                     # for mri_common
sys.path.insert(0, join(script_dir, 'models'))     # for escnn3d
import moment_kernels as mk

dtype = torch.float32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BADFILES_ADNI = [
    'ADNI_018_S_0450_MR_MPRAGE_br_raw_20061208163123098_86_S23719_I32421.nii.gz',
    'ADNI_018_S_0425_MR_MPRAGE_br_raw_20061212165306445_86_S23892_I32853.nii.gz',  # contains NaN voxels
]
PIXEL_SIGMA  = 1.9448502071922995   # RMS of training data, computed once
DATA_SUBDIR  = 'adni_affine_processed'

# 48 axis-aligned orientations: full octahedral group with reflections
# (6 axis permutations × 8 sign combinations = all signed 3×3 permutation matrices)
_ORIENT_NP = []
for _perm in permutations(range(3)):
    _P = np.eye(3)[list(_perm)]
    for _signs in product((1, -1), repeat=3):
        _ORIENT_NP.append(np.diag(_signs) @ _P)
ORIENTATIONS_NP = _ORIENT_NP   # list of 48 (3,3) numpy arrays


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        files = glob(join(data_dir, DATA_SUBDIR, '*'))
        self.filesgood = [f for f in files
                          if not any(b in f for b in BADFILES_ADNI)]

        ids = np.unique([basename(f)[5:15] for f in self.filesgood])
        state = np.random.get_state()
        np.random.seed(0)
        ids = np.random.permutation(ids)
        ntrain = int(len(ids) * 0.8)
        nval   = int(len(ids) * 0.1)
        np.random.set_state(state)

        if split == 'train':
            ids = ids[:ntrain]
        elif split == 'val':
            ids = ids[ntrain:ntrain + nval]
        elif split == 'test':
            ids = ids[ntrain + nval:-1]
        else:
            raise ValueError(f'split must be train/val/test, got {split}')

        self.filesgood = [f for f in self.filesgood
                          if basename(f)[5:15] in ids]

    def __len__(self):
        return len(self.filesgood)

    def __getitem__(self, i):
        vol = nib.load(join(self.filesgood[i], 'processed_80.nii.gz'))
        return torch.tensor(
            vol.get_fdata()[None] / PIXEL_SIGMA, dtype=dtype
        )[..., 1:, 1:, 1:]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Net0(torch.nn.Module):
    """Standard 3D CNN baseline.
    ch0=6 → 139,710 params (original)
    ch0=7 → 186,996 params (parameter-matched to Net, default)
    """
    def __init__(self, ch0=7):
        super().__init__()
        chin, chout = 1, 12
        pm = 'reflect'
        bn = lambda c: torch.nn.BatchNorm3d(c, track_running_stats=True)

        self.c0  = torch.nn.Conv3d(chin,  ch0,   3, 1, 1, padding_mode=pm)
        self.b0  = bn(ch0)
        self.c0a = torch.nn.Conv3d(ch0,   ch0,   3, 1, 1, padding_mode=pm)
        self.b0a = bn(ch0)

        self.c1  = torch.nn.Conv3d(ch0,   ch0*2, 3, 2, 1, padding_mode=pm)
        self.b1  = bn(ch0*2)
        self.c1a = torch.nn.Conv3d(ch0*2, ch0*2, 3, 1, 1, padding_mode=pm)
        self.b1a = bn(ch0*2)

        self.c2  = torch.nn.Conv3d(ch0*2, ch0*4, 3, 2, 1, padding_mode=pm)
        self.b2  = bn(ch0*4)
        self.c2a = torch.nn.Conv3d(ch0*4, ch0*4, 3, 1, 1, padding_mode=pm)
        self.b2a = bn(ch0*4)

        self.c3  = torch.nn.Conv3d(ch0*4, ch0*8, 3, 2, 1, padding_mode=pm)
        self.b3  = bn(ch0*8)
        self.c3a = torch.nn.Conv3d(ch0*8, ch0*8, 3, 1, 1, padding_mode=pm)
        self.b3a = bn(ch0*8)

        self.c4  = torch.nn.Conv3d(ch0*8, chout, 3, 1, 1, padding_mode=pm)

    def forward(self, x):
        x = torch.relu(self.b0(self.c0(x)));   x = torch.relu(self.b0a(self.c0a(x)))
        x = torch.relu(self.b1(self.c1(x)));   x = torch.relu(self.b1a(self.c1a(x)))
        x = torch.relu(self.b2(self.c2(x)));   x = torch.relu(self.b2a(self.c2a(x)))
        x = torch.relu(self.b3(self.c3(x)));   x = torch.relu(self.b3a(self.c3a(x)))
        return self.c4(x).mean((-1, -2, -3))


class Net(torch.nn.Module):
    """Equivariant moment-kernel model. 174,384 parameters."""
    def __init__(self):
        super().__init__()
        chin, ch0 = 1, 4
        chout = [0, 4]   # four vectors stacked into the 3×4 affine output
        k, dim, p, o = 3, 3, 1, 3
        pm = 'reflect'
        bnkw = {'track_running_stats': True}
        skw  = {'subtract_one': False}

        self.down = mk.Downsample()

        def blk(ci, co):
            return (mk.OEConvBlock(ci, co, k, dim, padding=p, padding_mode=pm),
                    mk.OEBatchnormBlock(co, dim, **bnkw),
                    mk.OESigmoidBlock(co, dim, **skw))

        self.c0,  self.b0,  self.s0  = blk([chin],    [ch0]*o)
        self.c0a, self.b0a, self.s0a = blk([ch0]*o,   [ch0]*o)
        self.c1,  self.b1,  self.s1  = blk([ch0]*o,   [ch0*2]*o)
        self.c1a, self.b1a, self.s1a = blk([ch0*2]*o, [ch0*2]*o)
        self.c2,  self.b2,  self.s2  = blk([ch0*2]*o, [ch0*4]*o)
        self.c2a, self.b2a, self.s2a = blk([ch0*4]*o, [ch0*4]*o)
        self.c3,  self.b3,  self.s3  = blk([ch0*4]*o, [ch0*8]*o)
        self.c3a, self.b3a, self.s3a = blk([ch0*8]*o, [ch0*8]*o)
        self.c4 = mk.OEConvBlock([ch0*8]*o, chout, k, dim, padding=p, padding_mode=pm)

    def forward(self, x):
        x = self.s0(self.b0(self.down(self.c0(x)))); x = self.s0a(self.b0a(self.c0a(x)))
        x = self.s1(self.b1(self.down(self.c1(x)))); x = self.s1a(self.b1a(self.c1a(x)))
        x = self.s2(self.b2(self.down(self.c2(x)))); x = self.s2a(self.b2a(self.c2a(x)))
        x = self.s3(self.b3(self.down(self.c3(x)))); x = self.s3a(self.b3a(self.c3a(x)))
        return self.c4(x).mean((-1, -2, -3))


MODELS = {'standard': Net0, 'equivariant': Net}

try:
    from escnn3d import Net_ESCNN
    MODELS['escnn'] = Net_ESCNN
except ImportError:
    pass

from mri_synthmorph import Net_SynthMorph
MODELS['synthmorph'] = Net_SynthMorph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_affine(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run net forward and reshape output to (B, 3, 4) affine matrices."""
    return net(x).reshape(-1, 4, 3).transpose(-1, -2)


def sample_R():
    """Sample a uniformly random rotation matrix via matrix exponential."""
    w = np.random.randn(3)
    W = np.array([[0, w[0], w[1]], [-w[0], 0, w[2]], [-w[1], -w[2], 0]])
    return expm(W)


# Cache coordinate grids by volume shape to avoid rebuilding per call.
# Key: (D, H, W) tuple.  Value: (D,H,W,3) float32 tensor on CPU.
_GRID_CACHE: dict = {}


def apply_affine(x: torch.Tensor, A: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply a batch of 4×4 affine matrices A (B,4,4) to volume batch x (B,C,D,H,W)."""
    shape = tuple(x.shape[-3:])
    if shape not in _GRID_CACHE:
        xi = [torch.arange(n) - (n - 1) / 2 for n in shape]
        _GRID_CACHE[shape] = torch.stack(torch.meshgrid(xi, indexing='ij'), -1)
    X = _GRID_CACHE[shape].to(device)

    # pinv degrades gracefully on singular A (can occur with poorly-trained models).
    Ai = torch.linalg.pinv(A)
    Xs = (Ai[:, None, None, None, :3, :3] @ X[None, ..., None])[..., 0] \
         + Ai[:, None, None, None, :3, -1]
    Xs = (Xs - X[0, 0, 0]) / (X[-1, -1, -1] - X[0, 0, 0]) * 2 - 1
    return torch.nn.functional.grid_sample(
        x, Xs.flip(-1), align_corners=True, padding_mode='border')


def sample_batch_affines(batch_size: int, device: torch.device,
                         aug: str = 'so3') -> torch.Tensor:
    """Sample a batch of random affine matrices (B,4,4).

    aug:
      'none' — scale (0.95–1.05) + translation only; no rotation.
               Tests pure equivariance: model never sees rotations during training.
      'so3'  — random SO(3) rotation × scale × SO(3) + translation (default).
               Current standard setup; no reflections.
      'o3'   — SO(3) × scale × SO(3) + random axis reflection (50% prob) + translation.
               Full O(3) augmentation; gives standard CNN a fair chance on reflections.
    """
    if aug not in ('none', 'so3', 'o3'):
        raise ValueError(f"aug must be 'none', 'so3', or 'o3', got '{aug}'")
    A_ = []
    for _ in range(batch_size):
        scale = np.diag(np.random.rand(3) * 0.1 + 0.95)
        T = np.random.randn(3) * 3
        if aug == 'none':
            L = scale
        elif aug == 'so3':
            L = sample_R() @ scale @ sample_R()
        else:  # o3
            L = sample_R() @ scale @ sample_R()
            if np.random.rand() < 0.5:
                axis = np.random.randint(3)   # pick x, y, or z to reflect
                flip = np.eye(3)
                flip[axis, axis] = -1         # det = -1, proper reflection
                L = flip @ L
        A = np.concatenate((L, T[:, None]), axis=-1)
        A = np.concatenate((A, np.array([[0, 0, 0, 1]])))
        A_.append(A)
    return torch.tensor(np.array(A_), device=device, dtype=dtype)
