#!/usr/bin/env python3
"""
Evaluation script for 3D MRI affine registration.

Modes:
    standard       -- MSE, worst-case MSE, inter-orientation consistency,
                      per-volume MSE histogram, qualitative registration figure
    rotation_sweep -- MSE vs rotation angle (mirrors DermaMNIST sweep figure)
    ants           -- ANTs affine baseline (requires antspy)

Usage:
    python mri_eval.py --model equivariant --checkpoint results/adni/run1/best_equivariant_run1.pt \
                       --data_dir /path/to/adni --mode standard

    python mri_eval.py --model equivariant --checkpoint results/adni/run1/best_equivariant_run1.pt \
                       --data_dir /path/to/adni --mode rotation_sweep --angle_step 10

    python mri_eval.py --mode ants --data_dir /path/to/adni
"""
import os, json, argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from os.path import join

from mri_common import Dataset, MODELS, apply_affine, sample_batch_affines, predict_affine, dtype, ORIENTATIONS_NP


# ---------------------------------------------------------------------------
# Helpers (eval-only)
# ---------------------------------------------------------------------------

def rotation_matrix_z(angle_deg):
    """3×3 rotation matrix around z-axis."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_affine4(R3):
    """Embed 3×3 rotation into 4×4 homogeneous matrix (no translation)."""
    A = np.eye(4)
    A[:3, :3] = R3
    return A


def load_model(args, device):
    kwargs = {}
    if args.model == 'standard' and args.ch0 is not None:
        kwargs['ch0'] = args.ch0
    if args.model == 'escnn':
        if args.base_mult is not None:
            kwargs['base_mult'] = args.base_mult
        if args.bn_type is not None:
            kwargs['bn_type'] = args.bn_type
        elif args.no_bn:
            kwargs['bn_type'] = 'none'
        if args.gspace is not None:
            kwargs['gspace_name'] = args.gspace
    if args.model == 'synthmorph':
        kwargs['preset'] = args.synth_preset
    net = MODELS[args.model](**kwargs).to(device)
    # ESCNN requires model to be in eval mode BEFORE loading state_dict —
    # the filter buffer is rebuilt from raw weights on the first eval() call.
    # Loading in train mode then calling eval() can use stale cached filters.
    net.eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    net.load_state_dict(state_dict)
    print(f'Loaded: {args.checkpoint}')
    return net


# ---------------------------------------------------------------------------
# Mode: standard
# Computes MSE, worst-case MSE, inter-orientation consistency (ED2)
# Uses 48 axis-aligned orientations (full octahedral group with reflections)
# ---------------------------------------------------------------------------

def eval_standard(args, net, loader, device, out_dir, stem):
    """Evaluate MSE, worst-case MSE, inter-orientation consistency.

    Also saves:
      - per-volume MSE histogram
      - qualitative registration figure (original / GT-warped / pred-registered slices)
    """
    orientations_t = [torch.tensor(R, device=device, dtype=dtype) for R in ORIENTATIONS_NP]
    # Index orientations by determinant sign — rotations (det=+1) vs reflections (det=−1).
    # An SO(3)-equivariant model has architectural support only for the 24 rotation
    # orientations; reflections are out-of-distribution. Splitting metrics by det makes
    # the architectural-vs-augmentation source of any gap explicit.
    import numpy as _np
    rot_idx  = [i for i, R in enumerate(ORIENTATIONS_NP) if _np.linalg.det(R) > 0]
    refl_idx = [i for i, R in enumerate(ORIENTATIONS_NP) if _np.linalg.det(R) < 0]

    # Fix the affine draw so a rerun reproduces the same evaluation. Without this
    # the ground-truth affines are redrawn every invocation: the aggregate MSE/ED2
    # are stable over 412 volumes, but the qualitative figure picks a single volume
    # per percentile and so changes subject entirely between runs. eval_rotation_sweep
    # already seeds with 42; matching it keeps the two modes on the same draw.
    _np.random.seed(args.seed)

    MSEsave, MSEworstsave, ED2save = [], [], []
    per_vol_mse = []
    # Collect all identity-orientation candidates for qualitative selection
    qual_candidates = []   # list of (mse_per_vol, x_cpu, A_cpu, Ahat4_cpu)

    # Precompute the orientation matrices and their inverses once (not per batch).
    R_mats     = []
    R_inv_mats = []
    for R0 in orientations_t:
        R = torch.eye(4, device=device, dtype=dtype)
        R[:3, :3] = R0
        R_mats.append(R)
        R_inv_mats.append(torch.linalg.inv(R))

    # The qualitative figure only ever uses orientation 0 (the identity): cases are
    # both ranked and rendered from it. Restricting the loop to that orientation
    # therefore produces a bit-identical figure for 1/48 of the compute, which is
    # what makes this feasible on CPU. Metrics need all 48 and are skipped below.
    if getattr(args, 'qual_only', False):
        assert _np.allclose(ORIENTATIONS_NP[0], _np.eye(3)), \
            'qual_only assumes orientation 0 is the identity'
        R_mats, R_inv_mats = R_mats[:1], R_inv_mats[:1]

    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            A = sample_batch_affines(x.shape[0], device, aug=args.aug)

            Ahatsave = []
            Esave = []
            for orient_i, (R, R_inv) in enumerate(zip(R_mats, R_inv_mats)):
                RA = R @ A
                Ax = apply_affine(x, RA, device)
                Ahat = predict_affine(net, Ax)
                E = torch.sum((Ahat - RA[:, :3]) ** 2, (-1, -2))

                Ahat4 = torch.eye(4, device=device, dtype=dtype)[None].repeat(x.shape[0], 1, 1)
                Ahat4[:, :3] = Ahat
                Ahatsave.append(R_inv @ Ahat4)
                Esave.append(E)
                per_vol_mse.extend(E.cpu().tolist())

                # Collect identity-orientation candidates for qualitative figure
                if orient_i == 0:
                    for i in range(x.shape[0]):
                        qual_candidates.append((
                            E[i].item(),
                            x[i:i+1].cpu(), A[i:i+1].cpu(), Ahat4[i:i+1].cpu()
                        ))

            Ahatsave = torch.stack(Ahatsave)   # (N_orient, B, 4, 4)
            ED2 = torch.mean(
                torch.sum((Ahatsave[None] - Ahatsave[:, None]) ** 2, (-1, -2)), (0, 1))
            ED2save.append(ED2.cpu())
            MSEsave.append(torch.stack(Esave).cpu())
            MSEworstsave.append(torch.amax(torch.stack(Esave), 0).cpu())

    # Select qualitative examples at 25th / 50th / 75th percentile MSE
    # so the figure honestly shows good, typical, and hard cases.
    qual_candidates.sort(key=lambda t: t[0])
    n_cand = len(qual_candidates)
    pick_indices = [
        int(n_cand * 0.25),   # 25th percentile — good case
        int(n_cand * 0.50),   # 50th percentile — typical case
        int(n_cand * 0.75),   # 75th percentile — hard case
    ]
    selected = [qual_candidates[i] for i in pick_indices]
    qual_examples = (
        torch.cat([s[1] for s in selected]),
        torch.cat([s[2] for s in selected]),
        torch.cat([s[3] for s in selected]),
    )

    if getattr(args, 'qual_only', False):
        plot_qualitative(qual_examples,
                         os.path.join(out_dir, f'qualitative_{stem}.png'))
        print('\n--qual_only: qualitative figure written from the identity orientation.')
        print('  MSE / worst-case MSE / ED2 were NOT computed (they need all 48 '
              'orientations); no eval JSON is written, so existing metrics are preserved.')
        return {'model': args.model, 'checkpoint': args.checkpoint, 'aug': args.aug,
                'qual_only': True, 'n_orient': 1, 'n_test': len(per_vol_mse)}

    mse      = torch.cat([m.reshape(-1) for m in MSEsave]).float()
    mse_wc   = torch.cat(MSEworstsave).float()
    ed2      = torch.cat(ED2save).float()

    # Split metrics by orientation determinant sign for fair comparison
    # against SO(3)-equivariant models that lack reflection support.
    # MSEsave stack shape: each entry (n_orient, batch); concatenated along batch -> (n_orient, n_test_total)
    mse_per_orient = torch.cat([m for m in MSEsave], dim=1).float()  # (n_orient, n_test)
    mse_rot   = mse_per_orient[rot_idx].reshape(-1)
    mse_refl  = mse_per_orient[refl_idx].reshape(-1)
    mse_wc_rot  = torch.amax(mse_per_orient[rot_idx],  dim=0)
    mse_wc_refl = torch.amax(mse_per_orient[refl_idx], dim=0)

    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'aug': args.aug,
        'n_orient': len(ORIENTATIONS_NP),
        'n_test': len(mse),
        'mse_mean':    float(mse.mean()),
        'mse_std':     float(mse.std()),
        'mse_wc_mean': float(mse_wc.mean()),
        'mse_wc_std':  float(mse_wc.std()),
        'ed2_mean':    float(ed2.mean()),
        'ed2_std':     float(ed2.std()),
        # Split by orientation determinant: 24 rotations (det=+1) vs 24 reflections (det=−1).
        'mse_rot_mean':     float(mse_rot.mean()),
        'mse_rot_std':      float(mse_rot.std()),
        'mse_refl_mean':    float(mse_refl.mean()),
        'mse_refl_std':     float(mse_refl.std()),
        'mse_wc_rot_mean':  float(mse_wc_rot.mean()),
        'mse_wc_rot_std':   float(mse_wc_rot.std()),
        'mse_wc_refl_mean': float(mse_wc_refl.mean()),
        'mse_wc_refl_std':  float(mse_wc_refl.std()),
    }

    print(f'\nStandard evaluation — {args.model}')
    print(f'  MSE:                     {results["mse_mean"]:.4f} ± {results["mse_std"]:.4f}')
    print(f'    rotations only (24):   {results["mse_rot_mean"]:.4f} ± {results["mse_rot_std"]:.4f}')
    print(f'    reflections only (24): {results["mse_refl_mean"]:.4f} ± {results["mse_refl_std"]:.4f}')
    print(f'  Worst-case MSE:          {results["mse_wc_mean"]:.4f} ± {results["mse_wc_std"]:.4f}')
    print(f'    rotations WC:          {results["mse_wc_rot_mean"]:.4f} ± {results["mse_wc_rot_std"]:.4f}')
    print(f'    reflections WC:        {results["mse_wc_refl_mean"]:.4f} ± {results["mse_wc_refl_std"]:.4f}')
    print(f'  Inter-orient. dist (ED2):{results["ed2_mean"]:.4f} ± {results["ed2_std"]:.4f}')

    plot_mse_histogram(per_vol_mse, args.model,
                       os.path.join(out_dir, f'mse_histogram_{stem}.png'))
    plot_qualitative(qual_examples,
                     os.path.join(out_dir, f'qualitative_{stem}.png'))

    return results


# ---------------------------------------------------------------------------
# Mode: rotation_sweep
# MSE vs rotation angle around z-axis (mirrors DermaMNIST sweep figure)
# ---------------------------------------------------------------------------

def eval_rotation_sweep(args, net, loader, device):
    """Sweep rotation angle around z-axis and measure MSE per angle."""
    angles = list(range(0, 360, args.angle_step))
    mse_per_angle = []

    # Pre-load all test volumes to avoid repeated I/O
    all_x, all_A = [], []
    np.random.seed(42)  # fixed seed for reproducible affine transforms
    for x in loader:
        x = x.to(device=device, dtype=dtype)
        A = sample_batch_affines(x.shape[0], device, aug=args.aug)
        all_x.append(x)
        all_A.append(A)

    with torch.no_grad():
        for angle in angles:
            R0 = rotation_matrix_z(angle)
            R  = torch.tensor(make_affine4(R0), device=device, dtype=dtype)

            batch_mse = []
            for x, A in zip(all_x, all_A):
                RA = R @ A
                Ahat = predict_affine(net, apply_affine(x, RA, device))
                mse = torch.mean((Ahat - RA[:, :3]) ** 2).item()
                batch_mse.append(mse)
            mse_per_angle.append(float(np.mean(batch_mse)))
            print(f'  {angle:3d}°  MSE={mse_per_angle[-1]:.6f}')

    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'angle_step': args.angle_step,
        'angles': angles,
        'mse_per_angle': mse_per_angle,
        'avg_mse': float(np.mean(mse_per_angle)),
        'mse_range': float(max(mse_per_angle) - min(mse_per_angle)),
    }

    print(f'\nRotation sweep — {args.model}')
    print(f'  Avg MSE:   {results["avg_mse"]:.6f}')
    print(f'  MSE range: {results["mse_range"]:.6f}  (0 = perfectly equivariant)')

    return results


# ---------------------------------------------------------------------------
# Mode: ants
# Run ANTs affine registration on test set as published baseline
# ---------------------------------------------------------------------------

def eval_ants(args, device):
    """Run ANTs affine registration on test set. Requires antspy."""
    try:
        import ants
    except ImportError:
        raise ImportError('antspy is required for ANTs mode. Install with: pip install antspyx')

    dataset = Dataset(args.data_dir, split='test')
    # Use first test volume as fixed (standard orientation template)
    fixed_np = nib.load(join(dataset.filesgood[0], 'processed_80.nii.gz')).get_fdata()[1:, 1:, 1:]
    fixed_ants = ants.from_numpy(fixed_np.astype(np.float32))

    mse_list = []
    print('Running ANTs affine registration on test set...')
    for i, f in enumerate(dataset.filesgood[1:], 1):
        moving_np = nib.load(join(f, 'processed_80.nii.gz')).get_fdata()[1:, 1:, 1:]
        moving_ants = ants.from_numpy(moving_np.astype(np.float32))

        result = ants.registration(
            fixed=fixed_ants, moving=moving_ants,
            type_of_transform='Affine'
        )
        tx = ants.read_transform(result['fwdtransforms'][0])
        params = tx.parameters
        A_pred = np.array(params[:9]).reshape(3, 3)
        T_pred = np.array(params[9:12])
        Ahat = np.concatenate([A_pred, T_pred[:, None]], axis=1)  # 3x4

        # Measure how close to identity the recovered transform is
        mse = float(np.mean((Ahat - np.eye(3, 4)) ** 2))
        mse_list.append(mse)

        if i % 50 == 0:
            print(f'  {i}/{len(dataset.filesgood)-1}  MSE={np.mean(mse_list):.6f}')

    results = {
        'model': 'ants_affine',
        'n_test': len(mse_list),
        'mse_mean': float(np.mean(mse_list)),
        'mse_std':  float(np.std(mse_list)),
    }
    print(f'\nANTs affine baseline:')
    print(f'  MSE: {results["mse_mean"]:.4f} ± {results["mse_std"]:.4f}')
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_mse_histogram(per_vol_mse, model_name, out_path):
    """Histogram of per-volume MSE values to reveal outliers."""
    vals = np.array(per_vol_mse)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vals, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(vals.mean(), color='crimson', linewidth=1.4, linestyle='--',
               label=f'mean={vals.mean():.4f}')
    ax.set_xlabel('Per-volume MSE')
    ax.set_ylabel('Count')
    ax.set_title(f'MSE distribution — {model_name}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Histogram saved: {out_path}')


def _mid_slice(vol, axis):
    """Return the central 2D slice along `axis` of a 3D numpy array."""
    return np.take(vol, vol.shape[axis] // 2, axis=axis)


def _corresponding_slice(vol, axis, A4, forward=True, A_gt=None):
    """Return the slice of vol that corresponds to the center of the original volume.

    Coordinates are centered (apply_affine builds its grid as arange(n)-(n-1)/2),
    so the original center is the homogeneous origin e = (0, 0, 0, 1).

    apply_affine(x, A) samples: warped[y] = x[A⁻¹ @ y]
    So the original center appears in the warped volume where A⁻¹ y = e, i.e.
    y = A @ e = A[:, 3]. Pass forward=True with A4 = A.

    For pred_warped = apply_affine(gt_warped, pinv(Ahat)):
      pred[y] = gt_warped[Ahat @ y] = x[A⁻¹ @ Ahat @ y]
      The center appears where A⁻¹ Ahat y = e, i.e. y = Ahat⁻¹ @ A @ e
                                                     = Ahat⁻¹ @ A[:, 3].
      Pass forward=False with A4 = Ahat and A_gt = A.

    Note the ground-truth affine is required in the backward case. Using
    Ahat⁻¹ @ e instead (i.e. ignoring A's translation) is only correct when A
    has zero translation; since sample_batch_affines draws translations from
    randn(3)*3, that shortcut offsets the displayed slice by ~2.4 voxels on
    average even for a perfect prediction, which is unrelated to model quality.
    """
    if forward:
        center_coord = A4[:3, 3]              # for gt_warped: use A directly
    else:
        if A_gt is None:
            raise ValueError(
                '_corresponding_slice(forward=False) needs A_gt, the ground-truth '
                'affine, to locate the original center in the predicted frame.')
        center_coord = (np.linalg.inv(A4) @ A_gt[:, 3])[:3]
    n = vol.shape[axis]
    idx = int(round(center_coord[axis] + (n - 1) / 2))
    idx = max(0, min(n - 1, idx))
    return np.take(vol, idx, axis=axis)


def plot_qualitative(qual_examples, out_path, n_cases=3,
                     case_labels=('25th pct (good)', '50th pct (typical)', '75th pct (hard)')):
    """
    For each selected volume, show three columns per anatomical plane:
      - Original (unwarped) volume
      - Warped by ground-truth affine (input to the network)
      - Pred-registered: inv(Ahat) applied to warped — should recover original

    Cases are selected at 25th / 50th / 75th percentile MSE so the figure
    honestly represents good, typical, and hard predictions.
    Rows: axial / coronal / sagittal for each case.
    """
    x_cpu, A_cpu, Ahat4_cpu = qual_examples
    n_cases = min(n_cases, x_cpu.shape[0])

    planes  = ['axial', 'coronal', 'sagittal']
    ax_dims = [2, 1, 0]   # volume axis for each plane
    cols    = ['Original', 'GT-warped (input)', 'Pred-registered']

    x_batch    = x_cpu[:n_cases]       # (n_cases, 1, D, H, W)
    A_batch    = A_cpu[:n_cases]       # (n_cases, 4, 4)
    Ahat_batch = Ahat4_cpu[:n_cases]   # (n_cases, 4, 4)

    with torch.no_grad():
        gt_warped_tensor  = apply_affine(x_batch, A_batch, 'cpu')          # (B,1,D,H,W)
        gt_warped_batch   = gt_warped_tensor[:, 0].numpy()
        # pinv instead of inv: a poorly-trained model can predict near-singular
        # affines, which crash inv; pinv degrades gracefully.
        pred_warped_batch = apply_affine(
            gt_warped_tensor, torch.linalg.pinv(Ahat_batch), 'cpu'
        )[:, 0].numpy()

    originals = x_batch[:, 0].numpy()

    n_rows = n_cases * len(planes)
    fig, axes = plt.subplots(n_rows, len(cols),
                             figsize=(len(cols) * 2.2, n_rows * 2.0))

    for case_i in range(n_cases):
        original   = originals[case_i]
        gt_warped  = gt_warped_batch[case_i]
        pred_warped = pred_warped_batch[case_i]

        vmin = float(np.percentile(original, 1))
        vmax = float(np.percentile(original, 99))

        A4_np    = A_batch[case_i].numpy()
        Ahat4_np = Ahat_batch[case_i].numpy()

        for plane_i, (plane, axis) in enumerate(zip(planes, ax_dims)):
            row = case_i * len(planes) + plane_i
            ax_orig, ax_gt, ax_pred = axes[row]

            ax_orig.imshow(_mid_slice(original, axis), cmap='gray', vmin=vmin, vmax=vmax)
            ax_gt.imshow(  _corresponding_slice(gt_warped,   axis, A4_np,    forward=True),  cmap='gray', vmin=vmin, vmax=vmax)
            ax_pred.imshow(_corresponding_slice(pred_warped, axis, Ahat4_np, forward=False, A_gt=A4_np), cmap='gray', vmin=vmin, vmax=vmax)

            for ax in (ax_orig, ax_gt, ax_pred):
                ax.axis('off')
            label = case_labels[case_i] if case_i < len(case_labels) else f'Case {case_i+1}'
            ax_orig.set_ylabel(f'{label}\n{plane}', fontsize=7,
                               rotation=90, labelpad=2)

    for col_i, label in enumerate(cols):
        axes[0, col_i].set_title(label, fontsize=8)

    fig.suptitle('Qualitative registration results\n(rows: 25th / 50th / 75th percentile MSE)',
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Qualitative figure saved: {out_path}')


def plot_rotation_sweep(sweep_results, out_path):
    """Plot MSE vs rotation angle, one line per model."""
    fig, ax = plt.subplots(figsize=(6, 3.2))

    colors = {'standard': '#d62728', 'equivariant': '#1f77b4', 'ants_affine': '#2ca02c'}
    labels = {'standard': 'Standard CNN', 'equivariant': 'Moment kernels (ours)', 'ants_affine': 'ANTs affine'}

    for res in sweep_results:
        name = res['model']
        angles = res['angles'] + [360]
        mse    = res['mse_per_angle'] + [res['mse_per_angle'][0]]
        ax.plot(angles, mse, color=colors.get(name, 'grey'),
                label=labels.get(name, name), linewidth=1.6)

    for x in [90, 180, 270]:
        ax.axvline(x, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)

    ax.set_xlabel('Rotation angle (degrees)')
    ax.set_ylabel('Test MSE')
    ax.set_xlim(0, 360)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(90))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(45))
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.28),
              ncol=3, framealpha=0.9)
    ax.set_title('MSE vs. rotation angle — ADNI affine registration', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Figure saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    _script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.mode == 'ants' or args.checkpoint is None:
        out_dir = os.path.join(_script_dir, args.save_dir, args.data_flag)
    else:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'ants':
        results = eval_ants(args, device)
        out_json = os.path.join(out_dir, 'eval_ants_affine.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved: {out_json}')
        return

    net = load_model(args, device)
    dataset = Dataset(args.data_dir, split='test')
    loader  = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    print(f'Test set: {len(dataset)} volumes')

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    aug_suffix = '' if args.aug == 'so3' else f'_aug{args.aug}'

    if args.mode == 'standard':
        results = eval_standard(args, net, loader, device, out_dir, stem + aug_suffix)
        if args.qual_only:
            # Deliberately no JSON: a stub here would overwrite the real metrics.
            pass
        else:
            out_json = os.path.join(out_dir, f'eval_{stem}{aug_suffix}.json')
            with open(out_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'Results saved: {out_json}')

    elif args.mode == 'rotation_sweep':
        print(f'Running rotation sweep (step={args.angle_step}°)...')
        results = eval_rotation_sweep(args, net, loader, device)
        out_json = os.path.join(out_dir, f'rotation_sweep_{stem}.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        out_png = os.path.join(out_dir, f'rotation_sweep_{stem}.png')
        plot_rotation_sweep([results], out_png)
        print(f'Results saved: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MRI affine registration models')
    parser.add_argument('--model',      type=str, default='equivariant',
                        choices=['standard', 'equivariant', 'escnn', 'synthmorph'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt checkpoint (not needed for --mode ants)')
    parser.add_argument('--data_dir',   type=str, default=None,
                        help='Root directory containing adni_affine_processed/. '
                             'If omitted, read from the config JSON next to the checkpoint.')
    parser.add_argument('--mode',       type=str, default='standard',
                        choices=['standard', 'rotation_sweep', 'ants'])
    parser.add_argument('--angle_step', type=int, default=10,
                        help='Degree increment for rotation_sweep (default 10)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--qual_only',  action='store_true', default=False,
                        help='standard mode: render only the qualitative figure, using '
                             'the identity orientation. The figure is identical (cases are '
                             'ranked and drawn from that orientation) but costs 1/48 the '
                             'compute, which makes a CPU run practical. Skips MSE/WC/ED2 '
                             'and writes no eval JSON, so existing metrics are preserved.')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Seed for the ground-truth affine draw, so reruns are '
                             'reproducible. Matters most for the qualitative figure, '
                             'which shows one volume per percentile.')
    parser.add_argument('--save_dir',   type=str, default='results',
                        help='Output directory (relative to script location)')
    parser.add_argument('--data_flag',  type=str, default='adni',
                        help='Subdirectory name under save_dir')
    parser.add_argument('--aug',        type=str, default='so3',
                        choices=['none', 'so3', 'o3'],
                        help='Base affine distribution for eval (default so3 for all models)')
    parser.add_argument('--ch0',        type=int, default=None,
                        help='Base channels for standard Net0 (default 7). Use 6 for run9 checkpoint.')
    parser.add_argument('--base_mult',  type=int, default=None,
                        help='Width multiplier for ESCNN (defaults to model default). Match training value.')
    parser.add_argument('--no_bn',      action='store_true', default=False,
                        help='ESCNN only (legacy): equivalent to --bn_type none.')
    parser.add_argument('--bn_type',    type=str, default=None,
                        choices=['iid', 'gnorm', 'none'],
                        help='ESCNN only: batch norm type to match training.')
    parser.add_argument('--gspace',     type=str, default=None,
                        choices=['rot3d', 'fliprot3d'],
                        help='ESCNN only: gspace to match training.')
    parser.add_argument('--synth_preset', type=str, default='default',
                        choices=['default', 'matched'],
                        help='SynthMorph-architecture baseline only (NOT SynthMorph '
                             'method): preset to match training. "default" ~5.4M, '
                             '"matched" ~175k.')
    args = parser.parse_args()

    if args.mode != 'ants' and args.checkpoint is None:
        parser.error('--checkpoint is required for modes standard and rotation_sweep')

    # Infer model, data_dir, and ch0 from checkpoint config if not explicitly provided
    if args.checkpoint is not None:
        import glob as _glob
        config_candidates = _glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'config_*.json'))
        if config_candidates:
            with open(config_candidates[0]) as _f:
                _cfg = json.load(_f)
            # Only override defaults, not explicit user args
            _defaults = {'model': 'equivariant', 'data_dir': None, 'ch0': None}
            if args.model == _defaults['model'] and 'model' in _cfg:
                args.model = _cfg['model']
                print(f'model inferred from config: {args.model}')
            if args.data_dir is None:
                args.data_dir = _cfg['data_dir']
                print(f'data_dir inferred from config: {args.data_dir}')
            if args.ch0 is None and 'ch0' in _cfg:
                args.ch0 = _cfg['ch0']
                print(f'ch0 inferred from config: {args.ch0}')
            if args.base_mult is None and 'base_mult' in _cfg:
                args.base_mult = _cfg['base_mult']
                print(f'base_mult inferred from config: {args.base_mult}')
            if not args.no_bn and _cfg.get('no_bn', False):
                args.no_bn = True
                print(f'no_bn inferred from config: True')
            if args.bn_type is None and 'bn_type' in _cfg and _cfg['bn_type'] is not None:
                args.bn_type = _cfg['bn_type']
                print(f'bn_type inferred from config: {args.bn_type}')
            if args.gspace is None and 'gspace' in _cfg and _cfg['gspace'] is not None:
                args.gspace = _cfg['gspace']
                print(f'gspace inferred from config: {args.gspace}')
            if args.synth_preset == 'default' and 'synth_preset' in _cfg:
                args.synth_preset = _cfg['synth_preset']
                print(f'synth_preset inferred from config: {args.synth_preset}')
        elif args.data_dir is None:
            parser.error('--data_dir not provided and no config JSON found next to checkpoint')

    main(args)
