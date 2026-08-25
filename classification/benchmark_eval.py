import sys
import os

_script_dir = os.path.dirname(os.path.realpath(__file__))
_root_dir   = os.path.dirname(_script_dir)
sys.path.insert(0, _root_dir)    # for moment_kernels
sys.path.insert(0, _script_dir)  # for models.benchmark_models2

from models.benchmark_models2 import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

PRESETS = {
    "VanillaCNN": VanillaCNN,
    "SREVanillaCNN": SREVanillaCNN,
    "TrivialECNN": TrivialECNN,
    "TrivialIrrepECNN": TrivialIrrepECNN,
    "TrivialIrrepMatrixECNN": TrivialIrrepMatrixECNN,
    "RegularECNN": RegularECNN,
    "TrivialMoment": TrivialMoment,
    "TrivialIrrepMoment": TrivialIrrepMoment,
    "MatrixMoment": MatrixMoment,
    "MatrixMomentV2": MatrixMomentV2,
    "MatrixMomentOriginal": MatrixMomentOriginal,
    "VanillaCNNOriginal": VanillaCNNOriginal,
    "TrivialMomentOriginal": TrivialMomentOriginal,
    "TrivialIrrepMomentOriginal": TrivialIrrepMomentOriginal,
}
import torch.utils.data as data
import torch
from medmnist import INFO
import medmnist
import argparse
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassAUROC


def get_8_orientations(images):
    """Return list of 8 versions of each image: 4 rotations x 2 reflections."""
    orientations = []
    for k in range(4):
        rotated = torch.rot90(images, k, dims=[2, 3])
        orientations.append(rotated)
        orientations.append(torch.flip(rotated, dims=[3]))
    return orientations  # list of 8 tensors, each (B, C, H, W)


def rotate_batch(images, angle_deg, pad_mode='zeros'):
    """Rotate a batch of images by angle_deg using bilinear interpolation.

    pad_mode: how to handle corners at non-90° angles.
      'zeros'     — fill corners with black (default eval behaviour)
      'reflect'   — mirror image at boundary before rotating
      'replicate' — extend edge pixels outward before rotating
    """
    if pad_mode in ('reflect', 'replicate'):
        pad = images.shape[-1] // 2
        h, w = images.shape[-2], images.shape[-1]
        images = torch.nn.functional.pad(images, [pad] * 4, mode=pad_mode)
        rotated = TF.rotate(images, angle_deg, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        return TF.center_crop(rotated, [h, w])
    return TF.rotate(images, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)


def evaluate(model, loader, device, num_classes):
    model.eval()

    all_correct = []
    all_worst_case = []
    auroc = MulticlassAUROC(num_classes=num_classes, device=device)
    orient_aurocs = [MulticlassAUROC(num_classes=num_classes, device=device) for _ in range(8)]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            B = images.size(0)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels)
            auroc.update(torch.softmax(outputs, dim=1), labels)

            orientations = get_8_orientations(images)
            worst_case = torch.ones(B, dtype=torch.bool, device=device)
            for k, oriented in enumerate(orientations):
                out = model(oriented)
                worst_case &= (out.argmax(dim=1) == labels)
                orient_aurocs[k].update(torch.softmax(out, dim=1), labels)

            all_correct.append(correct.cpu())
            all_worst_case.append(worst_case.cpu())

    accuracy = torch.cat(all_correct).float().mean().item()
    worst_case_accuracy = torch.cat(all_worst_case).float().mean().item()
    auc = auroc.compute().item()
    worst_case_auc = min(a.compute().item() for a in orient_aurocs)

    return accuracy, worst_case_accuracy, auc, worst_case_auc


def evaluate_rotation_sweep(model, loader, device, num_classes, angle_step=10, pad_mode='zeros', batch_size=128):
    """Evaluate accuracy at each rotation angle (SRE-Conv protocol).

    Rotates the test set at `angle_step` degree increments from 0 to 360.
    Also evaluates horizontal and vertical reflection.
    Reports per-angle accuracy and average across all angles (matching SRE-Conv Table 1).
    """
    model.eval()
    angles = list(range(0, 360, angle_step))

    # Collect all images and labels first to avoid reloading per angle
    all_images, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_images.append(images)
            all_labels.append(labels.squeeze(1))
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    acc_per_angle = []
    with torch.no_grad():
        for angle in angles:
            rotated = rotate_batch(all_images, angle, pad_mode=pad_mode).to(device)
            labels = all_labels.to(device)
            correct = []
            # Process in batches to avoid OOM
            for i in range(0, len(rotated), batch_size):
                imgs_b = rotated[i:i+batch_size]
                labs_b = labels[i:i+batch_size]
                out = model(imgs_b)
                preds = out.argmax(dim=1)
                correct.append((preds == labs_b).cpu())
            acc = torch.cat(correct).float().mean().item()
            acc_per_angle.append(acc)

        labels_dev = all_labels.to(device)
        for flip_dims, attr in [([3], 'hflip'), ([2], 'vflip')]:
            flipped = torch.flip(all_images, dims=flip_dims).to(device)
            correct_f = []
            for i in range(0, len(flipped), batch_size):
                out = model(flipped[i:i+batch_size])
                correct_f.append((out.argmax(dim=1) == labels_dev[i:i+batch_size]).cpu())
            if attr == 'hflip':
                acc_hflip = torch.cat(correct_f).float().mean().item()
            else:
                acc_vflip = torch.cat(correct_f).float().mean().item()

    avg_rot_acc = float(np.mean(acc_per_angle))
    avg_ref_acc = float(np.mean([acc_hflip, acc_vflip]))

    return {
        'angles': angles,
        'acc_per_angle': acc_per_angle,
        'avg_rotation_accuracy': avg_rot_acc,
        'acc_hflip': acc_hflip,
        'acc_vflip': acc_vflip,
        'avg_reflection_accuracy': avg_ref_acc,
        'orig_accuracy': acc_per_angle[0],  # angle=0 is the original
    }


def evaluate_equivariance_error(model, loader, device, angle_step=5, pad_mode='replicate', batch_size=128):
    """Compute activation-level equivariance error as a function of rotation angle.

    For each angle theta, computes the mean relative L2 distance between the model's
    pre-softmax logits on the original image and the rotated image:

        eps(theta) = E_x [ ||f(R_theta * x) - f(x)||_2 / ||f(x)||_2 ]

    For a perfectly invariant network, eps(theta) = 0 for all theta.
    This metric isolates equivariance independently of classification accuracy.
    """
    model.eval()
    angles = list(range(0, 360, angle_step))

    # Collect all images (no labels needed)
    all_images = []
    with torch.no_grad():
        for images, _ in loader:
            all_images.append(images)
    all_images = torch.cat(all_images)  # (N, C, H, W)

    # Compute baseline logits f(x) at 0 degrees
    baseline_logits = []
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            imgs_b = all_images[i:i + batch_size].to(device)
            out = model(imgs_b)
            baseline_logits.append(out.cpu())
    baseline_logits = torch.cat(baseline_logits)  # (N, num_classes)

    # For each angle, compute per-image relative equivariance error
    err_per_angle = []
    with torch.no_grad():
        for angle in angles:
            rotated = rotate_batch(all_images, angle, pad_mode=pad_mode)
            errors = []
            for i in range(0, len(rotated), batch_size):
                imgs_b = rotated[i:i + batch_size].to(device)
                out = model(imgs_b)
                base = baseline_logits[i:i + batch_size].to(device)
                diff_norm = (out - base).norm(dim=1)          # (B,)
                base_norm = base.norm(dim=1).clamp(min=1e-6)  # (B,)
                errors.append((diff_norm / base_norm).cpu())
            err_per_angle.append(torch.cat(errors).mean().item())

    return {
        'angles': angles,
        'err_per_angle': err_per_angle,
        'avg_err': float(np.mean(err_per_angle)),
        'max_err': float(max(err_per_angle)),
        'pad_mode': pad_mode,
    }


def main(args):
    data_flag = args.data_flag
    # PCam is not part of MedMNIST; see pcam_dataset for the compatible shim.
    if data_flag == 'pcam':
        from pcam_dataset import PCAM_INFO, PCamDataset
        info, DataClass = PCAM_INFO, PCamDataset
    else:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
    img_channels = info['n_channels']
    n_classes = len(info['label'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = [] if args.no_normalize else [transforms.Normalize(mean=[0.5], std=[0.5])]
    resize = [transforms.Resize((args.resize, args.resize))] if args.resize else []
    test_transforms = transforms.Compose([
        *resize,
        transforms.ToTensor(),
        *normalize,
    ])
    test_dataset = DataClass(split='test', transform=test_transforms, download=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    import inspect
    model_kwargs = {}
    model_params = inspect.signature(PRESETS[args.model]).parameters
    if args.mk_downsample and 'use_mk_downsample' in model_params:
        model_kwargs['use_mk_downsample'] = True
    if args.escnn_pool and 'use_pool' in model_params:
        model_kwargs['use_pool'] = True
        model_kwargs['pool_type'] = args.escnn_pool_type
    if 'interp_mode' in model_params and args.interp_mode != 'linear':
        model_kwargs['interp_mode'] = args.interp_mode
        model_kwargs['sigma_factor'] = args.sigma_factor
    model = PRESETS[args.model](img_channels, args.n_channels, n_classes,
                                args.kernel_size, args.padding,
                                num_layers=args.num_layers, **model_kwargs).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {args.checkpoint}')

    checkpoint_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if args.mode == 'rotation_sweep' and args.pad_mode != 'zeros':
        checkpoint_stem += f'_{args.pad_mode}pad'
    out_dir = os.path.dirname(args.checkpoint)

    if args.mode == 'standard':
        accuracy, worst_case_accuracy, auc, worst_case_auc = evaluate(model, test_loader, device, n_classes)

        results = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'accuracy': accuracy,
            'worst_case_accuracy': worst_case_accuracy,
            'auc': auc,
            'worst_case_auc': worst_case_auc,
        }

        print(f'\nResults for {args.model}:')
        print(f'  Accuracy:            {accuracy:.4f}')
        print(f'  Worst-case accuracy: {worst_case_accuracy:.4f}')
        print(f'  AUC:                 {auc:.4f}')
        print(f'  Worst-case AUC:      {worst_case_auc:.4f}')

        out_path = os.path.join(out_dir, f'eval_{checkpoint_stem}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved to {out_path}')

    elif args.mode == 'rotation_sweep':
        print(f'Running rotation sweep (step={args.angle_step}°, pad_mode={args.pad_mode})...')
        sweep = evaluate_rotation_sweep(model, test_loader, device, n_classes,
                                        angle_step=args.angle_step,
                                        pad_mode=args.pad_mode,
                                        batch_size=args.batch_size)

        print(f'\nRotation sweep results for {args.model}:')
        print(f'  Orig. accuracy (0°):       {sweep["orig_accuracy"]:.4f}')
        print(f'  Avg. rotated accuracy:     {sweep["avg_rotation_accuracy"]:.4f}')
        print(f'  Rotation drop:             {sweep["orig_accuracy"] - sweep["avg_rotation_accuracy"]:+.4f}')
        print(f'  H-flip accuracy:           {sweep["acc_hflip"]:.4f}')
        print(f'  V-flip accuracy:           {sweep["acc_vflip"]:.4f}')
        print(f'  Avg. reflection accuracy:  {sweep["avg_reflection_accuracy"]:.4f}')
        print(f'  Reflection drop:           {sweep["orig_accuracy"] - sweep["avg_reflection_accuracy"]:+.4f}')

        # Plot accuracy vs angle
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sweep['angles'], [a * 100 for a in sweep['acc_per_angle']],
                marker='o', markersize=3, linewidth=1.5, label='Accuracy')
        ax.axhline(sweep['orig_accuracy'] * 100, color='gray', linestyle='--',
                   linewidth=1, label=f'Orig. ({sweep["orig_accuracy"]*100:.1f}%)')
        ax.axhline(sweep['avg_rotation_accuracy'] * 100, color='red', linestyle='--',
                   linewidth=1, label=f'Avg. rotated ({sweep["avg_rotation_accuracy"]*100:.1f}%)')
        ax.set_xlabel('Rotation angle (degrees)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{args.model} — accuracy vs rotation angle')
        ax.set_xticks(range(0, 361, 30))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'rotation_sweep_{checkpoint_stem}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Plot saved to {plot_path}')

        results = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'angle_step': args.angle_step,
            **sweep,
        }
        out_path = os.path.join(out_dir, f'rotation_sweep_{checkpoint_stem}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {out_path}')

    elif args.mode == 'equivariance_error':
        print(f'Running equivariance error sweep (step={args.angle_step}°, pad_mode={args.pad_mode})...')
        eq = evaluate_equivariance_error(model, test_loader, device,
                                         angle_step=args.angle_step,
                                         pad_mode=args.pad_mode,
                                         batch_size=args.batch_size)

        print(f'\nEquivariance error for {args.model}:')
        print(f'  Average eps:  {eq["avg_err"]:.4f}')
        print(f'  Max eps:      {eq["max_err"]:.4f}')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eq['angles'], [e * 100 for e in eq['err_per_angle']],
                marker='o', markersize=3, linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        for q in range(0, 361, 90):
            ax.axvline(q, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Rotation angle (degrees)')
        ax.set_ylabel(r'Equivariance error $\varepsilon(\theta)$ (%)')
        ax.set_title(f'{args.model} — equivariance error vs rotation angle\n'
                     r'$\varepsilon(\theta) = \mathbb{E}_x[\|f(R_\theta x)-f(x)\|_2\,/\,\|f(x)\|_2]$')
        ax.set_xticks(range(0, 361, 30))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        suffix = f'_{args.pad_mode}pad' if args.pad_mode != 'zeros' else ''
        plot_path = os.path.join(out_dir, f'equivariance_error_{checkpoint_stem}{suffix}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Plot saved to {plot_path}')

        results = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'angle_step': args.angle_step,
            **eq,
        }
        out_path = os.path.join(out_dir, f'equivariance_error_{checkpoint_stem}{suffix}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--data_flag', type=str, default='dermamnist')
    parser.add_argument('--n_channels', type=int, required=True,
                        help='n0 used during training')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Conv kernel size (must match training)')
    parser.add_argument('--padding', type=int, default=1,
                        help='Conv padding (must match training; typically (kernel_size-1)//2)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no_normalize', action='store_true', default=False,
                        help='Skip input normalization (must match training setting)')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'rotation_sweep', 'equivariance_error'],
                        help='standard: 8-orientation WC eval; rotation_sweep: SRE-Conv protocol; '
                             'equivariance_error: activation-level eps(theta) vs angle')
    parser.add_argument('--angle_step', type=int, default=10,
                        help='Degree increment for rotation_sweep mode (default 10, matching SRE-Conv)')
    parser.add_argument('--pad_mode', type=str, default='zeros',
                        choices=['zeros', 'reflect', 'replicate'],
                        help='Padding mode before rotating: zeros (default), reflect, or replicate — rotation_sweep only')
    parser.add_argument('--resize', type=int, default=None,
                        help='If set, resize images to (resize, resize) before evaluation. Must match the value used at training (use --resize 32 for SRE-Conv on DermaMNIST 28x28).')
    parser.add_argument('--escnn_pool_type', type=str, default='box', choices=['box', 'antialiased'],
                        help='Must match training. See benchmark_train.py for why the default is box.')
    parser.add_argument('--escnn_pool', action='store_true',
                        help='(TrivialIrrepMatrixECNN only) Must match the flag used at training time, otherwise the architecture and checkpoint will not correspond.')
    parser.add_argument('--mk_downsample', action='store_true',
                        help='(SREVanillaCNN only) Use parity-aware mk.Downsample in place of AvgPool2d(2,2). Must match the training-time setting.')
    parser.add_argument('--interp_mode', type=str, default='linear', choices=['linear', 'cubic', 'gaussian'],
                        help="(Moment-kernel models only) Radial-profile parameterization. Must match the training-time setting.")
    parser.add_argument('--sigma_factor', type=float, default=1.0,
                        help="Only used with --interp_mode gaussian. Must match training.")
    args = parser.parse_args()

    main(args)
