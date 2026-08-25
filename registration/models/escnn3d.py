"""ESCNN equivariant 3D CNN for affine registration.

Follows Rezvani et al. 2026 (arXiv 2604.08034) recipe:
- Group: configurable via `gspace_name`. 'rot3d' = SO(3) (continuous rotations,
  no reflections; matches paper's SE(3) setup); 'fliprot3d' = O(3) (rotations
  + reflections; matches moment kernel's o=3 architectural symmetry).
- Irrep mix at deeper layers: 5:2:1 scalar:vector:rank-2-tensor
  (5×1 + 2×3 + 1×5 = 16 dims per "unit" — paper Table 3 best config).
- First stage restricted to 4:4 scalar:vector with no l=2 (paper §2.4).
- Weiler-style gated sigmoid activation (`GatedNonLinearity1`).
- Strided R3Conv (stride=2) at level-entry stages for downsampling.

BN choice via `bn_type`:
- 'iid'   — IIDBatchNorm3d (per-channel running stats; default for our earlier
            runs A--E; not equivariance-preserving for non-trivial irreps).
- 'gnorm' — GNormBatchNorm (per-irrep-multiplet normalisation; equivariance-
            preserving; matches the 2D classification setup in this project's
            benchmark_models2.py and aligns with the recommendation implicit
            in escnn's library design).
- 'none'  — no BN (run27 setup; clean diagnostic baseline).

Channel schedule (base_mult=1, exact paper recipe):
  stage 0 (first, restricted): 4:4:0 scalar:vector:tensor (data dim 16)
  stage 1: 5:2:1 (data dim 16)
  stages 2-3: 10:4:2 (data dim 32)
  stages 4-5: 20:8:4 (data dim 64)
  stages 6-7: 40:16:8 (data dim 128)

Matches output contract of Net/Net0 in mri_common.py: (B, 1, D, H, W) input,
(B, 12) output → 3x4 affine via predict_affine.
"""
import torch
import torch.nn as tnn

import escnn.nn as enn
import escnn.gspaces as gspaces


def _get_reps(gs, gspace_name):
    """Return (trivial, vector, tensor) representations for the given gspace.

    For SO(3): irreps are indexed by l only.
    For O(3):  irreps are indexed by (parity, l). The 3D polar vector is
               obtained from `standard_representation()` (transforms as R·v
               under R∈O(3), including reflections); the rank-2 symmetric
               traceless tensor is `irrep(0, 2)` (parity-even, l=2).
    """
    trivial = gs.trivial_repr
    if gspace_name == 'rot3d':
        vector = gs.irrep(1)
        tensor = gs.irrep(2)
    elif gspace_name == 'fliprot3d':
        vector = gs.fibergroup.standard_representation()
        tensor = gs.irrep(0, 2)
    else:
        raise ValueError(
            f"gspace_name must be 'rot3d' or 'fliprot3d' (got {gspace_name!r})")
    return trivial, vector, tensor


class _GatedConv3dBlock(enn.EquivariantModule):
    """R3Conv + (optional batch norm) + GatedNonLinearity1.

    Pre-nonlinearity layout: `(n_s + n_v + n_t)` trivial gates, then n_s
    trivial + n_v vector + n_t tensor data reps in that order.
    GatedNonLinearity1 pairs the i-th gate with the i-th data rep and drops
    the gates after sigmoid-multiplication.
    """
    def __init__(self, gs, in_type, n_s, n_v, n_t, reps,
                 stride=1, kernel_size=3, bn_type='iid'):
        super().__init__()
        self.in_type = in_type
        trivial, vector, tensor = reps

        data_reps = n_s * [trivial] + n_v * [vector] + n_t * [tensor]
        n_data    = n_s + n_v + n_t
        gate_reps = n_data * [trivial]

        pre_nl_type  = enn.FieldType(gs, gate_reps + data_reps)
        post_nl_type = enn.FieldType(gs, data_reps)
        gates_list   = n_data * ['gate'] + n_data * ['gated']

        self.out_type = post_nl_type

        modules = [
            enn.R3Conv(in_type, pre_nl_type, kernel_size=kernel_size,
                       padding=kernel_size // 2, stride=stride),
        ]
        if bn_type == 'iid':
            modules.append(enn.IIDBatchNorm3d(pre_nl_type))
        elif bn_type == 'gnorm':
            modules.append(enn.GNormBatchNorm(pre_nl_type))
        elif bn_type == 'none':
            pass
        else:
            raise ValueError(f"bn_type must be 'iid', 'gnorm', or 'none' (got {bn_type!r})")
        modules.append(enn.GatedNonLinearity1(pre_nl_type, gates=gates_list))
        self.block = enn.SequentialModule(*modules)

    def forward(self, x):
        return self.block(x)

    def evaluate_output_shape(self, input_shape):
        return input_shape


class Net_ESCNN(tnn.Module):
    """ESCNN equivariant 3D CNN matching Rezvani et al. 2026 recipe.

    Stage 0 restricted to scalar+vector (4:4 ratio, no l=2). Stages 1-7 use
    5:2:1 scalar:vector:tensor. Channel multiplier doubles between levels.
    Strided R3Conv (stride=2) at stages 0, 2, 4, 6 for downsampling.

    base_mult=3 gives ~248k params, base_mult=4 gives ~439k params.
    bn_type ∈ {'iid', 'gnorm', 'none'} controls batch normalisation.
    gspace_name ∈ {'rot3d', 'fliprot3d'} controls the architectural symmetry
    group: 'rot3d' = SO(3) (run25--run28 default); 'fliprot3d' = O(3), which
    additionally provides architectural reflection equivariance and matches
    moment kernel's o=3 setup.
    """
    def __init__(self, base_mult=3, bn_type='iid', gspace_name='rot3d'):
        super().__init__()
        if gspace_name == 'rot3d':
            self.gs = gspaces.rot3dOnR3()
        elif gspace_name == 'fliprot3d':
            self.gs = gspaces.flipRot3dOnR3()
        else:
            raise ValueError(
                f"gspace_name must be 'rot3d' or 'fliprot3d' (got {gspace_name!r})")
        self.gspace_name = gspace_name

        reps = _get_reps(self.gs, gspace_name)
        trivial, vector, _ = reps

        self.in_type = enn.FieldType(self.gs, [trivial])

        def cfg(mult, restricted=False):
            if restricted:
                return 4 * mult, 4 * mult, 0
            return 5 * mult, 2 * mult, 1 * mult

        stage_cfgs = [
            cfg(1 * base_mult, restricted=True),   # stage 0: 4:4:0
            cfg(1 * base_mult),                    # stage 1: 5:2:1
            cfg(2 * base_mult),                    # stage 2: 10:4:2
            cfg(2 * base_mult),                    # stage 3
            cfg(4 * base_mult),                    # stage 4: 20:8:4
            cfg(4 * base_mult),                    # stage 5
            cfg(8 * base_mult),                    # stage 6: 40:16:8
            cfg(8 * base_mult),                    # stage 7
        ]

        layers = []
        prev_type = self.in_type
        for i, (n_s, n_v, n_t) in enumerate(stage_cfgs):
            stride = 2 if (i % 2 == 0) else 1
            blk = _GatedConv3dBlock(self.gs, prev_type, n_s, n_v, n_t, reps,
                                    stride=stride, bn_type=bn_type)
            layers.append(blk)
            prev_type = blk.out_type
        self.body = enn.SequentialModule(*layers)

        self.out_type = enn.FieldType(self.gs, 4 * [vector])
        self.head = enn.R3Conv(prev_type, self.out_type,
                               kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.body(x)
        x = self.head(x)
        return x.tensor.mean((-1, -2, -3))
