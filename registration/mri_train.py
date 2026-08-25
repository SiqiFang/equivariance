#!/usr/bin/env python3
"""
Training script for 3D MRI affine registration.

Usage:
    python mri_train.py --model standard   --data_dir /path/to/adni --epochs 200 --run 1
    python mri_train.py --model equivariant --data_dir /path/to/adni --epochs 200 --run 1
"""
import os, json, time, argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mri_common import Dataset, MODELS, apply_affine, sample_batch_affines, predict_affine, dtype


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    _script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(_script_dir, args.save_dir, args.data_flag)

    if args.run is None:
        # Auto-select the next unused run number
        existing = [
            int(d[3:]) for d in os.listdir(base_dir)
            if d.startswith('run') and d[3:].isdigit()
        ] if os.path.isdir(base_dir) else []
        args.run = max(existing, default=0) + 1
        print(f'Auto-selected run {args.run}')

    save_dir = os.path.join(base_dir, f'run{args.run}')
    if os.path.exists(save_dir) and not args.resume:
        raise RuntimeError(
            f'run{args.run} already exists at {save_dir}. '
            'Use --resume to continue it or omit --run to auto-select a new one.'
        )
    os.makedirs(save_dir, exist_ok=True)

    stem = f'{args.model}_run{args.run}'
    config_path  = os.path.join(save_dir, f'config_{stem}.json')
    ckpt_path    = os.path.join(save_dir, f'best_{stem}.pt')
    resume_path  = os.path.join(save_dir, f'resume_{stem}.pt')
    metrics_path = os.path.join(save_dir, f'metrics_{stem}.json')
    plot_path    = os.path.join(save_dir, f'loss_curve_{stem}.png')

    if not args.resume:
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    dataset     = Dataset(args.data_dir, split='train')
    dataset_val = Dataset(args.data_dir, split='val')
    print(f'Train: {len(dataset)}  Val: {len(dataset_val)}')

    loader     = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, timeout=120)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        num_workers=args.num_workers, timeout=120)

    model_kwargs = {}
    if args.model == 'escnn':
        if args.base_mult is not None:
            model_kwargs['base_mult'] = args.base_mult
        # Resolve bn_type with priority: explicit --bn_type > legacy --no_bn > default 'iid'.
        if args.bn_type is not None:
            model_kwargs['bn_type'] = args.bn_type
        elif args.no_bn:
            model_kwargs['bn_type'] = 'none'
        if args.gspace is not None:
            model_kwargs['gspace_name'] = args.gspace
    elif args.model == 'synthmorph':
        model_kwargs['preset'] = args.synth_preset
    net = MODELS[args.model](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    extras = []
    if 'base_mult' in model_kwargs:
        extras.append(f'base_mult={args.base_mult}')
    if 'bn_type' in model_kwargs:
        extras.append(f'bn_type={model_kwargs["bn_type"]}')
    if 'gspace_name' in model_kwargs:
        extras.append(f'gspace={model_kwargs["gspace_name"]}')
    if 'preset' in model_kwargs:
        extras.append(f'preset={model_kwargs["preset"]}')
    extras_str = '  ' + '  '.join(extras) if extras else ''
    print(f'Model: {args.model}  Parameters: {n_params:,}{extras_str}')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_losses, val_losses, epoch_times = [], [], []
    best_val = float('inf')
    start_epoch = 0

    if args.resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        if args.reset_optimizer:
            print('  --reset_optimizer set: Adam state restarted fresh.')
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                # Sanity-check the loaded optimizer state matches current parameter shapes.
                # PyTorch's Adam doesn't validate at load time — invalid state surfaces at
                # the first .step() call. Validate by running step on zero gradients.
                for p in net.parameters():
                    p.grad = torch.zeros_like(p)
                optimizer.step()
                optimizer.zero_grad()
            except (RuntimeError, ValueError) as e:
                print(f'  Warning: optimizer state incompatible ({type(e).__name__}: {e}). Resetting.')
                # Re-initialize optimizer from scratch
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
                # Clear any spurious gradients from the validation step
                for p in net.parameters():
                    p.grad = None
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses', [])
        epoch_times  = ckpt.get('epoch_times', [])
        best_val     = ckpt.get('best_val', float('inf'))
        start_epoch  = ckpt.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    # Pre-sample val affines once so the val loss is comparable across epochs.
    # loader_val has shuffle=False so batch order is deterministic.
    print('Pre-sampling val affines...')
    rng_state = np.random.get_state()
    np.random.seed(0)
    val_affines = [sample_batch_affines(x.shape[0], device, aug=args.aug) for x in loader_val]
    np.random.set_state(rng_state)
    print(f'  {len(val_affines)} val batches fixed.')

    pbar = tqdm(range(start_epoch, args.epochs), desc=args.model, unit='epoch')
    for epoch in pbar:
        net.train()
        t0 = time.time()
        batch_losses = []

        for x in tqdm(loader, desc='  train', leave=False):
            x = x.to(device=device, dtype=dtype)
            A = sample_batch_affines(x.shape[0], device, aug=args.aug)
            Ahat = predict_affine(net, apply_affine(x, A, device))
            loss = torch.mean((Ahat - A[:, :3]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        epoch_time = time.time() - t0

        net.eval()
        val_batch_losses = []
        with torch.no_grad():
            for x, A in zip(tqdm(loader_val, desc='  val  ', leave=False), val_affines):
                x = x.to(device=device, dtype=dtype)
                Ahat = predict_affine(net, apply_affine(x, A, device))
                loss = torch.mean((Ahat - A[:, :3]) ** 2)
                val_batch_losses.append(loss.item())
        val_loss = float(np.mean(val_batch_losses))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        # ETA uses rolling average of recent epochs for stability
        eta_h = (args.epochs - epoch - 1) * np.mean(epoch_times[-5:]) / 3600
        pbar.set_postfix(
            train=f'{train_loss:.5f}',
            val=f'{val_loss:.5f}',
            best=f'{best_val:.5f}',
            ETA=f'{eta_h:.1f}h',
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label='train')
        ax.plot(val_losses,   label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss')
        ax.set_title(f'{args.model} run{args.run}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), ckpt_path)
            tqdm.write(f'  → New best checkpoint at epoch {epoch+1} (val={best_val:.6f})')

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch_times': epoch_times,
                'best_val': best_val,
            }, resume_path)

    with open(metrics_path, 'w') as f:
        json.dump({
            'model': args.model,
            'n_params': n_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'best_val': best_val,
            'total_time_hours': sum(epoch_times) / 3600,
        }, f, indent=2)
    print(f'\nMetrics saved to {metrics_path}')
    print(f'Total training time: {sum(epoch_times)/3600:.2f} hours')
    print(f'Best val MSE: {best_val:.6f}')
    print(f'Best checkpoint: {ckpt_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train standard or equivariant MRI affine registration model')
    parser.add_argument('--model',      type=str, required=True,
                        choices=['standard', 'equivariant', 'escnn', 'synthmorph'])
    parser.add_argument('--data_dir',   type=str, required=True,
                        help='Root directory containing adni_affine_processed/')
    parser.add_argument('--epochs',     type=int, default=120)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--run',        type=int, default=None,
                        help='Run number (auto-increments if omitted)')
    parser.add_argument('--save_dir',   type=str, default='results')
    parser.add_argument('--data_flag',  type=str, default='adni',
                        help='Subdirectory name under save_dir')
    parser.add_argument('--resume',      action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--aug',         type=str, default='so3',
                        choices=['none', 'so3', 'o3'],
                        help='Augmentation: none=scale+translate only, '
                             'so3=random SO(3) rotations (default), '
                             'o3=SO(3)+random reflections')
    parser.add_argument('--base_mult',   type=int, default=None,
                        help='Width multiplier for ESCNN model (default: model default). '
                             '3 → ~250k params, 4 → ~440k params.')
    parser.add_argument('--no_bn',       action='store_true', default=False,
                        help='ESCNN only (legacy): equivalent to --bn_type none.')
    parser.add_argument('--bn_type',     type=str, default=None,
                        choices=['iid', 'gnorm', 'none'],
                        help='ESCNN only: batch norm type. "iid" = IIDBatchNorm3d '
                             '(default), "gnorm" = GNormBatchNorm (per-irrep, '
                             'equivariance-preserving; matches 2D classification setup), '
                             '"none" = no BN.')
    parser.add_argument('--gspace',      type=str, default=None,
                        choices=['rot3d', 'fliprot3d'],
                        help='ESCNN only: architectural symmetry group. "rot3d" = SO(3) '
                             '(default; runs 25--28 setup); "fliprot3d" = O(3), adds '
                             'reflection equivariance (matches moment kernel o=3).')
    parser.add_argument('--synth_preset', type=str, default='default',
                        choices=['default', 'matched'],
                        help='SynthMorph-architecture baseline only. NOTE: this is the '
                             'SynthMorph encoder architecture supervised on our affine '
                             'target, NOT the full SynthMorph method (which is pairwise '
                             '+ unsupervised). Presets: "default" = paper-faithful '
                             '(256,256,256,256)+(256,64), ~5.4M params; "matched" = '
                             '(12,24,48,96)+(64,64), ~175k params matches moment kernel.')
    parser.add_argument('--reset_optimizer', action='store_true', default=False,
                        help='With --resume: skip loading optimizer state and start '
                             'Adam fresh. Use when model architecture changed slightly '
                             'between save and resume.')
    args = parser.parse_args()
    train(args)
