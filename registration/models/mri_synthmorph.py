"""SynthMorph-encoder architecture baseline for affine registration.

**Important: this is NOT the SynthMorph method.** Published SynthMorph
(Hoffmann et al., 2021, "SynthMorph: learning contrast-invariant registration
without acquired images") is a *pairwise* registration network: it takes a
moving and fixed image as input, outputs a transform/warp, and is trained
with image-similarity losses on synthesised moving/fixed pairs. SynthMorph's
key contributions are (i) the synthesis pipeline that generates label-driven
training pairs, and (ii) the unsupervised training objective.

What this file implements is **the encoder-and-head architecture used in
SynthMorph's affine sub-network**, adapted for our setup:

  - Input: single warped volume (B, 1, D, H, W) — not a pair
  - Output: 12 affine parameters
  - Training loss: parameter-MSE on the known ground-truth A — not image NCC
  - No synthesis pipeline; we use ADNI volumes warped by our own random A

Use this baseline to ask one specific architectural question: *if we replace
our standard CNN encoder with SynthMorph's wider encoder*, does the affine
parameter regression improve? It does NOT answer "does SynthMorph as a
method beat moment kernels". For that you would need a separate pairwise
NCC-trained variant evaluated on image-reconstruction metrics (Dice, surface
distance), which is out of scope here.

In the paper, name this baseline as:
  - **"SynthMorph encoder (Hoffmann et al. 2021), supervised on affine target"**
  - or  **"SynthMorph-architecture baseline"**
  - NOT "SynthMorph baseline" (overclaims what is being compared)

---

SynthMorph's published affine encoder (default config):
  - 4-level Conv3D, stride-2 downsampling
  - `nb_enc_features = [256, 256, 256, 256]`
  - `nb_dense_features = [256, 64]` post-pool dense head, then linear → 12
  - LeakyReLU(0.2) activations, no batch normalization

Two presets:
  - `default`: paper-faithful (256,256,256,256)+(256,64) → ~5.4M params
  - `matched`: shrunk to ~175k params for parameter-matched comparison with
               moment kernel (174k) and standard CNN (187k)

Output contract matches Net/Net0 in mri_common.py: (B, 1, D, H, W) input,
(B, 12) output → reshaped by predict_affine into 3x4 affine matrix.
"""
import torch
import torch.nn as tnn


# (enc_features, dense_features) per preset.
PRESETS = {
    'default':  ((256, 256, 256, 256), (256, 64)),    # ~5.4M params (paper-faithful)
    'matched':  ((12, 24, 48, 96),     (64, 64)),     # ~175k params (matches moment kernel)
}


class _ConvLeaky3d(tnn.Module):
    """SynthMorph's base block: Conv3D + LeakyReLU(0.2). No batchnorm."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = tnn.Conv3d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding)
        self.act = tnn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class Net_SynthMorphArch(tnn.Module):
    """SynthMorph-architecture affine regressor (encoder + global pool + dense).

    Adapted from SynthMorph (Hoffmann et al. 2021) for single-image
    parameter-supervised affine regression. See module docstring for the
    distinction between this architecture-only adaptation and the full
    SynthMorph registration method.

    Args:
        enc_features:   channel count at each encoder stage (each layer
                        downsamples 2x via stride-2 conv).
        dense_features: hidden widths for the post-pool dense head, before
                        the final linear-to-12 projection.
        preset:         shorthand for (enc_features, dense_features); see
                        PRESETS at module top. When provided, overrides the
                        explicit feature args.
    """
    def __init__(self, enc_features=(256, 256, 256, 256),
                 dense_features=(256, 64),
                 preset=None):
        super().__init__()
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset {preset!r}. Valid: {list(PRESETS)}")
            enc_features, dense_features = PRESETS[preset]

        encoders = []
        in_ch = 1
        for out_ch in enc_features:
            encoders.append(_ConvLeaky3d(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.encoder = tnn.Sequential(*encoders)

        head = []
        prev = enc_features[-1]
        for n in dense_features:
            head.append(tnn.Linear(prev, n))
            head.append(tnn.LeakyReLU(0.2))
            prev = n
        head.append(tnn.Linear(prev, 12))
        self.head = tnn.Sequential(*head)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=(-1, -2, -3))      # global average pool over spatial dims
        return self.head(x)


# Back-compat alias: previous code imported `Net_SynthMorph`.
Net_SynthMorph = Net_SynthMorphArch
