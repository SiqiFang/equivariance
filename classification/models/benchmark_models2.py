import os
import sys
import importlib

_models_dir = os.path.dirname(os.path.realpath(__file__))
_root_dir   = os.path.dirname(os.path.dirname(_models_dir))
sys.path.insert(0, _root_dir)

import moment_kernels as mk
importlib.reload(mk)

import torch # type: ignore
import torch.nn as tnn # type: ignore
import torch.nn.functional as F # type: ignore

try:
    import escnn.nn as enn # type: ignore
    import escnn.gspaces as gspaces # type: ignore
    _ESCNN_AVAILABLE = True
except Exception:
    _ESCNN_AVAILABLE = False
    # Stubs so ECNN class definitions still parse (the inheritance line
    # `class ECNNConvBlock(enn.EquivariantModule)` is evaluated at import).
    # Any actual ECNN instantiation raises a clear ImportError.
    def _missing_escnn(*args, **kwargs):
        raise ImportError(
            "escnn is required for ECNN models (TrivialECNN, TrivialIrrepECNN, "
            "TrivialIrrepMatrixECNN, RegularECNN). Install via `pip install escnn`."
        )
    _StubEquivariantModule = type('_StubEquivariantModule', (object,), {})
    class _Stub:
        def __init__(self, **fixed):
            for k, v in fixed.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return _missing_escnn
    enn = _Stub(EquivariantModule=_StubEquivariantModule)
    gspaces = _Stub()

try:
    from SRE_Conv import SRE_Conv2d as _SRE_Conv2d
    _SRE_AVAILABLE = True
except ImportError:
    _SRE_AVAILABLE = False

class MomentConvBlock(tnn.Module):
    def __init__(self, in_scalars : int, in_vectors : int, out_scalars : int, out_vectors : int, kernel_size, padding, map_type = 'scalar_vector_to_scalar_vector'):
        '''
        in_type: FieldType
            The input type of the layer
        out_type: FieldType
            The output type of the layer
        kernel_size: int
            The size of the kernel
        padding: int
            The padding to use
        '''
        super(MomentConvBlock, self).__init__()
        self.layers_list = tnn.ModuleList()

        self.CONV_MAPPINGS = {
            'scalar_vector_to_scalar_vector': mk.ScalarVectorToScalarVector(in_scalars, in_vectors, out_scalars, out_vectors, kernel_size, padding),
            'scalar_to_scalar': mk.ScalarToScalar(in_scalars, out_scalars, kernel_size, padding),
            'scalar_to_vector': mk.ScalarToVector(in_scalars, out_vectors, kernel_size, padding),
            'vector_to_vector': mk.VectorToVector(in_vectors, out_vectors, kernel_size, padding),
            'vector_to_scalar': mk.VectorToScalar(in_vectors, out_scalars, kernel_size, padding),
        }
        self.BN_MAPPINGS = {
            'scalar_vector_to_scalar_vector': mk.ScalarVectorBatchnorm(out_scalars, out_vectors),
            'scalar_to_scalar': mk.ScalarBatchnorm(out_scalars),
            'scalar_to_vector': mk.VectorBatchnorm(out_vectors),
            'vector_to_vector': mk.VectorBatchnorm(out_vectors),
            'vector_to_scalar': mk.ScalarBatchnorm(out_scalars),
        }
        self.SIG_MAPPINGS = {
            'scalar_vector_to_scalar_vector': mk.ScalarVectorSigmoid(out_scalars),
            'scalar_to_scalar': mk.ScalarSigmoid(),
            'scalar_to_vector': mk.VectorSigmoid(),
            'vector_to_vector': mk.VectorSigmoid(),
            'vector_to_scalar': mk.ScalarSigmoid(),
        }

        # if the input and output scalars and vectors are the same, then we do not need to downsample
        self.layers_list.append(self.CONV_MAPPINGS[map_type])
        if in_scalars != out_scalars or in_vectors != out_vectors:
            self.layers_list.append(mk.Downsample())
        self._add_bn_sigmoid(map_type, out_scalars, out_vectors)
        self.block = tnn.Sequential(*self.layers_list)

    def _add_bn_sigmoid(self, map_type, out_scalars, out_vectors):
        bn_mapping = self.BN_MAPPINGS[map_type]
        sig_mapping = self.SIG_MAPPINGS[map_type]

        # We need this conditional for situations of scalar_vector_to_scalar_vector where the output is either all scalars or all vectors
        # such that we can use the appropriate batchnorm and sigmoid functions.
        if map_type == 'scalar_vector_to_scalar_vector':
            if out_vectors == 0:
                bn_mapping = self.BN_MAPPINGS['scalar_to_scalar']
                sig_mapping = self.SIG_MAPPINGS['scalar_to_scalar']
            elif out_scalars == 0:
                bn_mapping = self.BN_MAPPINGS['vector_to_vector']
                sig_mapping = self.SIG_MAPPINGS['vector_to_vector']

        self.layers_list.append(bn_mapping)
        self.layers_list.append(sig_mapping)

    def forward(self, x):
        return self.block(x)

class TorchConvBlock(tnn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size, padding):
        super(TorchConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # If the input and output channels are the same, then the stride is 1, do not downsample
        if in_channels == out_channels:
            stride = 1
        else:
            stride = 2

        self.block = tnn.Sequential(
            tnn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride),
            tnn.BatchNorm2d(out_channels),
            tnn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class VanillaCNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 5):
        super(VanillaCNN, self).__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Channels double every 2 layers (as per paper: "after every second layer")
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()

        # Create the layers (according to the number of layers specified)
        for i in range(num_layers):
            self.layers_list.append(TorchConvBlock(self.stages[i], self.stages[i + 1], kernel_size, padding))

        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class SREConvBlock(tnn.Module):
    """SRE-Conv block matching Du et al.'s recipe for D_4 equivariance.

    AvgPool2d(2, 2) is exactly D_4-equivariant only when the input feature
    map has even spatial size; on odd-sized features it drops the rightmost
    column and bottom row asymmetrically and breaks D_4. Setting
    ``use_mk_downsample=True`` swaps in the parity-aware ``mk.Downsample``,
    which subsamples on odd dimensions and averages pairs on even ones —
    exactly D_4-equivariant on either parity.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding,
                 use_mk_downsample: bool = False):
        super().__init__()
        if not _SRE_AVAILABLE:
            raise ImportError("SRE-Conv not installed. Run: pip install SRE-Conv")
        downsample = (in_channels != out_channels)
        # sre_k: number of radial bands. Paper formula: k//2 + 2.
        sre_k = kernel_size // 2 + 2
        layers = [
            _SRE_Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        padding=padding, stride=1, sre_k=sre_k),
            tnn.BatchNorm2d(out_channels),
            tnn.ReLU(),
        ]
        if downsample:
            layers.append(mk.Downsample() if use_mk_downsample
                          else tnn.AvgPool2d(kernel_size=2, stride=2))
        self.block = tnn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SREVanillaCNN(tnn.Module):
    """VanillaCNN with nn.Conv2d replaced by SRE_Conv2d (Du et al., 2025).

    Identical architecture to VanillaCNN (same channel schedule, same downsampling)
    so parameter counts are comparable. SRE_Conv2d uses radially symmetric band
    kernels to achieve approximate equivariance at all rotation angles.

    Requires: pip install SRE-Conv
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int,
                 kernel_size=3, padding=1, num_layers: int = 8,
                 use_mk_downsample: bool = False):
        super().__init__()
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()
        for i in range(num_layers):
            self.layers_list.append(
                SREConvBlock(self.stages[i], self.stages[i + 1], kernel_size, padding,
                             use_mk_downsample=use_mk_downsample))
        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ECNNConvBlock(enn.EquivariantModule):
    def __init__(self, in_type : enn.FieldType, out_type : enn.FieldType, group, kernel_size, padding, stride=None):
        super(ECNNConvBlock, self).__init__()
        self.in_type = in_type
        self.out_type = out_type

        if stride is not None:
            stride = stride
        elif in_type.size == out_type.size:
            stride = 1
        else:
            stride = 2

        # Per ESCNN paper (Weiler & Cesa): trivial (scalar) representations must use a
        # pointwise nonlinearity; non-trivial representations use NormNonLinearity.
        # Using NormNonLinearity on scalars computes f(|x|)/|x| * x which is numerically
        # unstable near zero and can break equivariance in practice.
        trivial_id = out_type.gspace.trivial_repr.id
        all_trivial = all(r.id == trivial_id for r in out_type.representations)

        # PointwiseBatchNorm for scalar-only layers; GNormBatchNorm for layers with
        # non-trivial irreps. GNormBatchNorm normalises by group-orbit norms which is
        # incorrect for trivial representations and breaks equivariance in practice.
        if all_trivial:
            self.norm = enn.InnerBatchNorm(out_type)
            self.nonlinearity = enn.PointwiseNonLinearity(out_type, function='p_relu')
        else:
            # track_running_stats=False: always use batch statistics rather than
            # accumulated running stats. Running stats can be corrupted by near-zero
            # norm activations during training (via NormNonLinearity instability),
            # causing explosion in eval mode. Batch stats are equivariant when the
            # batch is large enough and avoid this corruption.
            self.norm = enn.GNormBatchNorm(out_type)
            self.nonlinearity = enn.NormNonLinearity(out_type)

        self.block = enn.SequentialModule(enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, padding = padding, stride = stride),
                                          self.norm,
                                          self.nonlinearity
                                          )

    def forward(self, x):
        return self.block(x)

    def evaluate_output_shape(self, input_shape: tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

class TrivialECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = -1):
        super(TrivialECNN, self).__init__()
        self.in_type = None
        self.out_type = None
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = []

        # O(2): equivariant to all continuous rotations AND reflections (N=-1 = continuous)
        self.r2_act = gspaces.flipRot2dOnR2(N = N_equivariance)

        for i in range(num_layers):

            # On the very first layer, we must specify that we want to use the trivial representation as the input * img_channels
            # On subsequent layers, the input type is simply the output of the previous layer
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class TrivialIrrepECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = -1):
        super(TrivialIrrepECNN, self).__init__()
        self.in_type = None
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = []

        # O(2): equivariant to all continuous rotations AND reflections (N=-1 = continuous)
        # irrep(1, 1) = standard 2D vector representation at frequency 1
        self.r2_act = gspaces.flipRot2dOnR2(N = N_equivariance)

        for i in range(num_layers):

            # The input of first layer should be trivial only. The output of last layer should be trivial only. Intermediate layers should be both trivial and irrep
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, (self.stages[i + 1] // 2) * [self.r2_act.trivial_repr] + (self.stages[i + 1] // 2) * [self.r2_act.irrep(1, 1)])
            elif i == num_layers - 1:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, (self.stages[i + 1] // 2) * [self.r2_act.trivial_repr] + (self.stages[i + 1] // 2) * [self.r2_act.irrep(1, 1)])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def _escnn_pool(out_type, pool_type='box', sigma=0.66, stride=2):
    """Equivariance-preserving downsampling for escnn field types.

    'box' -- 2x2 average with stride 2, i.e. out[k] = (x[2k] + x[2k+1]) / 2. This is
    the same operator as mk.Downsample and is the one to use. For even spatial size the
    2x2 blocks tile the grid and a 90 degree rotation maps blocks onto blocks, so the
    average commutes with the rotation exactly.

    'antialiased' -- escnn's PointwiseAvgPoolAntialiased: Gaussian blur followed by
    subsampling of the even sublattice {0, 2, 4, ...}. MEASURED TO BREAK D4: under the
    grid rotation i -> N-1-i an even sublattice maps to the odd one, so the sampled
    points do not correspond. A per-layer check at 96x96 gives ~1e0 error at the first
    pool versus ~1e-7 without pooling. Kept only so the finding can be reproduced.

    Averaging is a channel-independent linear spatial filter, so unlike max pooling it
    is safe for non-trivial irreps as well as for scalar fields.
    """
    if pool_type == 'box':
        cls = getattr(enn, 'PointwiseAvgPool', None) or getattr(enn, 'PointwiseAvgPool2D', None)
        if cls is None:
            raise RuntimeError("escnn build exposes neither PointwiseAvgPool nor PointwiseAvgPool2D")
        return cls(out_type, kernel_size=2, stride=stride)
    elif pool_type == 'antialiased':
        cls = getattr(enn, 'PointwiseAvgPoolAntialiased', None) or getattr(enn, 'PointwiseAvgPoolAntialiased2D', None)
        if cls is None:
            raise RuntimeError("escnn build exposes no PointwiseAvgPoolAntialiased")
        return cls(out_type, sigma=sigma, stride=stride)
    raise ValueError(f"unknown pool_type {pool_type!r}, expected 'box' or 'antialiased'")


class TrivialIrrepMatrixECNN(tnn.Module):
    """ESCNN model with scalar + vector + matrix channels under O(2).
    O(2) (N=-1, continuous rotations + reflections) supports:
      - irrep(0, 0) = trivial (scalar)
      - irrep(1, 1) = frequency-1 (vector)
      - irrep(1, 2) = frequency-2 (matrix / traceless symmetric tensor)
    This matches the scalar+vector+matrix decomposition used by MatrixMomentV2.
    n0 = channels per type at the first layer."""
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 8, N_equivariance : int = -1,
                 use_pool : bool = False, pool_type : str = 'box', pool_sigma : float = 0.66):
        super(TrivialIrrepMatrixECNN, self).__init__()
        self.in_type = None
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_pool = use_pool

        # n0 channels per type, doubling every 2 layers
        self.stages = [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = []

        # O(2): equivariant to all continuous rotations AND reflections.
        self.r2_act = gspaces.flipRot2dOnR2(N = N_equivariance)

        for i in range(num_layers):
            n = self.stages[i]
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, n * [self.r2_act.trivial_repr] + n * [self.r2_act.irrep(1, 1)] + n * [self.r2_act.irrep(1, 2)])
            elif i == num_layers - 1:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, n * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, n * [self.r2_act.trivial_repr] + n * [self.r2_act.irrep(1, 1)] + n * [self.r2_act.irrep(1, 2)])
            # stride=1 everywhere. This mirrors MatrixMomentV2, which also does not
            # downsample, and avoids stride-2 R2Conv, which is numerically unstable for
            # small images (28x28) with O(2) and breaks equivariance during training.
            #
            # NOTE: MatrixMomentOriginal — the moment-kernel model used for the paper
            # table and for PCam — DOES downsample 3x (96->48->24->12 at 96x96). So with
            # the default below this baseline runs at full resolution through all layers
            # while its moment-kernel counterpart does not, which inflates ESCNN's memory
            # and per-epoch time for reasons unrelated to the kernel parameterization.
            # Pass use_pool=True to match the moment kernel's downsampling. See
            # _escnn_pool above: use the default 'box' type, which is the same operator
            # as mk.Downsample. Default False so every previously-published run
            # reproduces exactly.
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding, stride=1))

            # Downsample wherever the next stage widens, giving 3 downsamples at
            # num_layers=8 and matching MatrixMomentOriginal's 3 (96->48->24->12).
            if use_pool and i + 1 < num_layers and self.stages[i + 1] != n:
                self.layers_list.append(_escnn_pool(out_type, pool_type=pool_type, sigma=pool_sigma))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class RegularECNN(tnn.Module):
    def __init__(self, img_channels : int, n0 : int, num_classes : int, kernel_size = 3, padding = 1, num_layers : int = 4, N_equivariance : int = 4):
        super(RegularECNN, self).__init__()
        self.in_type = None
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = []

        # D4: equivariant to 90° rotations AND reflections (matches the 8-orientation worst-case test)
        self.r2_act = gspaces.flipRot2dOnR2(N = N_equivariance)

        for i in range(num_layers):
            if i == 0:
                in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
                self.in_type = in_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.regular_repr])
            elif i == num_layers - 1:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.trivial_repr])
            else:
                in_type = out_type
                out_type = enn.FieldType(self.r2_act, self.stages[i + 1] * [self.r2_act.regular_repr])
            self.layers_list.append(ECNNConvBlock(in_type, out_type, self.r2_act, kernel_size, padding))

        self.layers = enn.SequentialModule(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        x = x.tensor
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class TrivialMoment(tnn.Module):
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size = 3, padding = 1, num_layers: int = 4):
        super(TrivialMoment, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()

        for i in range(num_layers):
            self.layers_list.append(MomentConvBlock(self.stages[i], 0, self.stages[i + 1], 0, kernel_size, padding, map_type = 'scalar_to_scalar'))

        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class TrivialIrrepMoment(tnn.Module):
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size = 3, padding = 1, num_layers: int = 4):
        super(TrivialIrrepMoment, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # The input channels + number of channels at each stage of the network should double
        self.stages = [img_channels] + [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()

        # The input of first layer should be trivial only. The output of last layer should be trivial only. Intermediate layers should be both trivial and irrep.
        for i in range(num_layers):
            if i == 0:
                self.layers_list.append(MomentConvBlock(self.stages[i], 0, self.stages[i + 1] // 2, self.stages[i + 1] // 2, kernel_size, padding, map_type = 'scalar_vector_to_scalar_vector'))
            elif i == num_layers - 1:
                self.layers_list.append(MomentConvBlock(self.stages[i] // 2, self.stages[i] // 2, self.stages[i + 1], 0, kernel_size, padding, map_type = 'scalar_vector_to_scalar_vector'))
            else:
                self.layers_list.append(MomentConvBlock(self.stages[i] // 2, self.stages[i] // 2, self.stages[i + 1] // 2, self.stages[i + 1] // 2, kernel_size, padding, map_type = 'scalar_vector_to_scalar_vector'))

        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class MatrixMoment(tnn.Module):
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8):
        super(MatrixMoment, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # channels per type (scalar, vector, matrix) at each stage; double every 2 layers
        self.stages = [n0 * (2 ** (i // 2)) for i in range(num_layers)]
        self.layers_list = tnn.ModuleList()

        dimension = 2

        for i in range(num_layers):
            if i == 0:
                in_ch = [img_channels]
                out_ch = [self.stages[i]] * 3
            elif i == num_layers - 1:
                in_ch = [self.stages[i - 1]] * 3
                out_ch = [self.stages[i]]
            else:
                in_ch = [self.stages[i - 1]] * 3
                out_ch = [self.stages[i]] * 3

            self.layers_list.append(mk.OEConvBlock(in_ch, out_ch, kernel_size, dimension, padding=padding))
            self.layers_list.append(mk.Downsample())
            self.layers_list.append(mk.OEBatchnormBlock(out_ch, dimension))
            self.layers_list.append(mk.OESigmoidBlock(out_ch, dimension))

        self.layers = tnn.Sequential(*self.layers_list)
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MatrixMomentV2(tnn.Module):
    """MatrixMoment rebuilt from advisor's Net2 template.

    All layers output scalar+vector+matrix. After global avg pool,
    only scalar channels feed the linear classifier (invariant output).
    No downsampling — global avg pool handles spatial aggregation.
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8):
        super(MatrixMomentV2, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        dimension = 2

        # channels per type double every 2 layers
        self.stages = [n0 * (2 ** (i // 2)) for i in range(num_layers)]

        self.convs = tnn.ModuleList()
        self.norms = tnn.ModuleList()
        self.acts = tnn.ModuleList()

        for i in range(num_layers):
            in_ch = [img_channels] if i == 0 else [self.stages[i - 1]] * 3
            out_ch = [self.stages[i]] * 3  # all layers output all three types

            self.convs.append(mk.OEConvBlock(in_ch, out_ch, kernel_size, dimension, padding=padding))
            self.norms.append(mk.OEBatchnormBlock(out_ch, dimension))
            self.acts.append(mk.OESigmoidBlock(out_ch, dimension))

        # only scalar channels (first stages[-1]) feed the classifier
        self.linear = tnn.Linear(self.stages[-1], num_classes)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.norms[i](x)
            x = self.acts[i](x)
        x = x.mean((-1, -2))           # global average pool
        x = x[:, :self.stages[-1]]     # take scalar channels only
        x = self.linear(x)
        return x


class MatrixMomentOriginal(tnn.Module):
    """Exact reproduction of Net from NeurIPS 2025 supplementary notebook.

    Matches notebook: ch0=4, 3 downsamples (after c1/c2/c3), last layer scalar only,
    no BN/sigmoid on last layer, padding_mode='reflect', subtract_one=False.

    Use with: lr=1e-3, batch_size=64, epochs=1000, --no_normalize.
    Gives 174,487 params at n0=4.
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8,
                 interp_mode: str = 'linear', sigma_factor: float = 1.0):
        super().__init__()
        dimension = 2
        o = 3
        padding_mode = 'reflect'
        bnkwargs = {'track_running_stats': True}
        skwargs = {'subtract_one': False}
        ch = n0  # n0=4 matches paper
        ckw = {'interp_mode': interp_mode, 'sigma_factor': sigma_factor}

        self.down = mk.Downsample()

        self.c0  = mk.OEConvBlock([img_channels], [ch]*o,   kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b0  = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0  = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)
        self.c0a = mk.OEConvBlock([ch]*o, [ch]*o,           kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b0a = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0a = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)

        self.c1  = mk.OEConvBlock([ch]*o,   [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b1  = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1  = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)
        self.c1a = mk.OEConvBlock([ch*2]*o, [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b1a = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1a = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)

        self.c2  = mk.OEConvBlock([ch*2]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b2  = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2  = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)
        self.c2a = mk.OEConvBlock([ch*4]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b2a = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2a = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)

        self.c3  = mk.OEConvBlock([ch*4]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b3  = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3  = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)
        self.c3a = mk.OEConvBlock([ch*8]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)
        self.b3a = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3a = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)

        # last layer: all types → scalar only (num_classes), no BN/sigmoid
        self.c4 = mk.OEConvBlock([ch*8]*o, [num_classes], kernel_size, dimension, padding=padding, padding_mode=padding_mode, **ckw)

    def forward(self, x):
        x = self.s0(self.b0(self.c0(x)))
        x = self.s0a(self.b0a(self.c0a(x)))

        x = self.s1(self.b1(self.down(self.c1(x))))
        x = self.s1a(self.b1a(self.c1a(x)))

        x = self.s2(self.b2(self.down(self.c2(x))))
        x = self.s2a(self.b2a(self.c2a(x)))

        x = self.s3(self.b3(self.down(self.c3(x))))
        x = self.s3a(self.b3a(self.c3a(x)))

        x = self.c4(x)
        return x.mean((-1, -2))


class TrivialMomentOriginal(tnn.Module):
    """Rank-0 (scalar only) ablation variant matching MatrixMomentOriginal architecture.

    Same paired-block + 3-downsample + reflect-padding structure as MatrixMomentOriginal,
    but all layers use scalar channels only (o=1).

    Use with: lr=1e-3, batch_size=64, epochs=1000, --no_normalize.
    Find n0 for ~174k params: TrivialMomentOriginal(3, n0, 7).
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8):
        super().__init__()
        dimension = 2
        o = 1
        padding_mode = 'reflect'
        bnkwargs = {'track_running_stats': True}
        skwargs = {'subtract_one': False}
        ch = n0

        self.down = mk.Downsample()

        self.c0  = mk.OEConvBlock([img_channels], [ch]*o,   kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b0  = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0  = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)
        self.c0a = mk.OEConvBlock([ch]*o, [ch]*o,           kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b0a = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0a = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)

        self.c1  = mk.OEConvBlock([ch]*o,   [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b1  = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1  = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)
        self.c1a = mk.OEConvBlock([ch*2]*o, [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b1a = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1a = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)

        self.c2  = mk.OEConvBlock([ch*2]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b2  = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2  = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)
        self.c2a = mk.OEConvBlock([ch*4]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b2a = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2a = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)

        self.c3  = mk.OEConvBlock([ch*4]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b3  = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3  = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)
        self.c3a = mk.OEConvBlock([ch*8]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b3a = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3a = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)

        self.c4 = mk.OEConvBlock([ch*8]*o, [num_classes], kernel_size, dimension, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        x = self.s0(self.b0(self.c0(x)))
        x = self.s0a(self.b0a(self.c0a(x)))

        x = self.s1(self.b1(self.down(self.c1(x))))
        x = self.s1a(self.b1a(self.c1a(x)))

        x = self.s2(self.b2(self.down(self.c2(x))))
        x = self.s2a(self.b2a(self.c2a(x)))

        x = self.s3(self.b3(self.down(self.c3(x))))
        x = self.s3a(self.b3a(self.c3a(x)))

        x = self.c4(x)
        return x.mean((-1, -2))


class TrivialIrrepMomentOriginal(tnn.Module):
    """Rank-2 (scalar+vector) ablation variant matching MatrixMomentOriginal architecture.

    Same paired-block + 3-downsample + reflect-padding structure as MatrixMomentOriginal,
    but all layers use scalar+vector channels (o=2).

    Use with: lr=1e-3, batch_size=64, epochs=1000, --no_normalize.
    Find n0 for ~174k params: TrivialIrrepMomentOriginal(3, n0, 7).
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8):
        super().__init__()
        dimension = 2
        o = 2
        padding_mode = 'reflect'
        bnkwargs = {'track_running_stats': True}
        skwargs = {'subtract_one': False}
        ch = n0

        self.down = mk.Downsample()

        self.c0  = mk.OEConvBlock([img_channels], [ch]*o,   kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b0  = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0  = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)
        self.c0a = mk.OEConvBlock([ch]*o, [ch]*o,           kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b0a = mk.OEBatchnormBlock([ch]*o,  dimension, **bnkwargs)
        self.s0a = mk.OESigmoidBlock([ch]*o,    dimension, **skwargs)

        self.c1  = mk.OEConvBlock([ch]*o,   [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b1  = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1  = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)
        self.c1a = mk.OEConvBlock([ch*2]*o, [ch*2]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b1a = mk.OEBatchnormBlock([ch*2]*o, dimension, **bnkwargs)
        self.s1a = mk.OESigmoidBlock([ch*2]*o,   dimension, **skwargs)

        self.c2  = mk.OEConvBlock([ch*2]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b2  = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2  = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)
        self.c2a = mk.OEConvBlock([ch*4]*o, [ch*4]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b2a = mk.OEBatchnormBlock([ch*4]*o, dimension, **bnkwargs)
        self.s2a = mk.OESigmoidBlock([ch*4]*o,   dimension, **skwargs)

        self.c3  = mk.OEConvBlock([ch*4]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b3  = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3  = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)
        self.c3a = mk.OEConvBlock([ch*8]*o, [ch*8]*o, kernel_size, dimension, padding=padding, padding_mode=padding_mode)
        self.b3a = mk.OEBatchnormBlock([ch*8]*o, dimension, **bnkwargs)
        self.s3a = mk.OESigmoidBlock([ch*8]*o,   dimension, **skwargs)

        self.c4 = mk.OEConvBlock([ch*8]*o, [num_classes], kernel_size, dimension, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        x = self.s0(self.b0(self.c0(x)))
        x = self.s0a(self.b0a(self.c0a(x)))

        x = self.s1(self.b1(self.down(self.c1(x))))
        x = self.s1a(self.b1a(self.c1a(x)))

        x = self.s2(self.b2(self.down(self.c2(x))))
        x = self.s2a(self.b2a(self.c2a(x)))

        x = self.s3(self.b3(self.down(self.c3(x))))
        x = self.s3a(self.b3a(self.c3a(x)))

        x = self.c4(x)
        return x.mean((-1, -2))


class VanillaCNNOriginal(tnn.Module):
    """Exact reproduction of Net0 (standard baseline) from NeurIPS 2025 supplementary notebook.

    Matches notebook: ch0=12, stride-2 Conv2d for downsampling, ReLU, standard BatchNorm2d.
    Use with: lr=1e-3, batch_size=64, epochs=1000, --no_normalize.
    """
    def __init__(self, img_channels: int, n0: int, num_classes: int, kernel_size=3, padding=1, num_layers: int = 8):
        super().__init__()
        ch = n0  # n0=12 matches paper
        padding_mode = 'reflect'

        self.c0  = tnn.Conv2d(img_channels, ch,   3, 1, 1, padding_mode=padding_mode)
        self.b0  = tnn.BatchNorm2d(ch,  track_running_stats=True)
        self.c0a = tnn.Conv2d(ch,   ch,   3, 1, 1, padding_mode=padding_mode)
        self.b0a = tnn.BatchNorm2d(ch,  track_running_stats=True)

        self.c1  = tnn.Conv2d(ch,   ch*2, 3, 2, 1, padding_mode=padding_mode)  # stride 2
        self.b1  = tnn.BatchNorm2d(ch*2, track_running_stats=True)
        self.c1a = tnn.Conv2d(ch*2, ch*2, 3, 1, 1, padding_mode=padding_mode)
        self.b1a = tnn.BatchNorm2d(ch*2, track_running_stats=True)

        self.c2  = tnn.Conv2d(ch*2, ch*4, 3, 2, 1, padding_mode=padding_mode)  # stride 2
        self.b2  = tnn.BatchNorm2d(ch*4, track_running_stats=True)
        self.c2a = tnn.Conv2d(ch*4, ch*4, 3, 1, 1, padding_mode=padding_mode)
        self.b2a = tnn.BatchNorm2d(ch*4, track_running_stats=True)

        self.c3  = tnn.Conv2d(ch*4, ch*8, 3, 2, 1, padding_mode=padding_mode)  # stride 2
        self.b3  = tnn.BatchNorm2d(ch*8, track_running_stats=True)
        self.c3a = tnn.Conv2d(ch*8, ch*8, 3, 1, 1, padding_mode=padding_mode)
        self.b3a = tnn.BatchNorm2d(ch*8, track_running_stats=True)

        self.c4  = tnn.Conv2d(ch*8, num_classes, 3, 1, 1, padding_mode=padding_mode)

    def forward(self, x):
        x = F.relu(self.b0(self.c0(x)))
        x = F.relu(self.b0a(self.c0a(x)))
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b1a(self.c1a(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b2a(self.c2a(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b3a(self.c3a(x)))
        x = self.c4(x)
        return x.mean((-1, -2))


# count number of parameters
def count_parameters(model):
    model.train()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model(model: torch.nn.Module, device: torch.device, input_shape: tuple):
    model.to(device).eval()
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        y = model(x)
    return y
