import sys
import os

_script_dir = os.path.dirname(os.path.realpath(__file__))
_root_dir   = os.path.dirname(_script_dir)
sys.path.insert(0, _root_dir)    # for moment_kernels
sys.path.insert(0, _script_dir)  # for models.benchmark_models2

from models.benchmark_models2 import *
import torchvision.transforms as transforms # type: ignore
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC # type: ignore
import torch.utils.data as data # type: ignore
import torch # type: ignore
from medmnist import INFO # type: ignore
import medmnist # type: ignore
import argparse
import random
import numpy as np # type: ignore
import time
from tqdm import tqdm # type: ignore
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import profiler # type: ignore

PRESETS ={
    "VanillaCNN" : VanillaCNN,
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


def calculate_metrics(model, loader, device, num_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    batch_losses = []
    acc = MulticlassAccuracy(num_classes=num_classes, device=device)
    precision = MulticlassPrecision(num_classes=num_classes, device=device)
    recall = MulticlassRecall(num_classes=num_classes, device=device)
    auroc = MulticlassAUROC(num_classes=num_classes, device=device)
    n_batches = len(loader)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            acc.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            auroc.update(outputs, labels)

    valid_loss = np.mean(batch_losses)
    valid_acc = acc.compute().item()
    valid_precision = precision.compute().item()
    valid_recall = recall.compute().item()
    valid_auroc = auroc.compute().item()

    acc.reset()
    precision.reset()
    recall.reset()
    auroc.reset()

    return valid_loss, valid_acc, valid_precision, valid_recall, valid_auroc

class CircleCrop(object):
    def __init__(self, h, w, center=None, radius=None):
        self.h = h
        self.w = w
        self.center = center
        self.radius = radius

    def __call__(self, img):
        if self.center is None:
            self.center = (self.h // 2, self.w // 2)
        if self.radius is None:
            self.radius = min(self.h, self.w) // 2

        y, x = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        mask = dist_from_center <= self.radius
        img[:, ~mask] = 0
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(h={self.h}, w={self.w}, center={self.center}, radius={self.radius})"

def load_data(args):

    data_flag = args.data_flag
    download = True
    # PCam is not part of MedMNIST; pcam_dataset supplies a MedMNIST-shaped
    # info dict and a DataClass with the same constructor signature.
    if data_flag == 'pcam':
        from pcam_dataset import PCAM_INFO, PCamDataset
        info, DataClass = PCAM_INFO, PCamDataset
    else:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
    task = info['task']
    img_channels = info['n_channels']
    n_classes = len(info['label'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms to convert from image to normalized tensor (or more if augmentation, but not for testing purposes)
    normalize = [] if args.no_normalize else [transforms.Normalize(mean=[0.5], std=[0.5])]
    resize = [transforms.Resize((args.resize, args.resize))] if args.resize else []

    if args.augment == 'discrete':
        train_transforms = transforms.Compose([
            *resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
                transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
                transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
            ]),
            transforms.ToTensor(),
            *normalize,
        ])
    elif args.augment == 'continuous':
        train_transforms = transforms.Compose([
            *resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            *normalize,
        ])
    else:
        train_transforms = transforms.Compose([
            *resize,
            transforms.ToTensor(),
            *normalize,
        ])

    # separate transforms for test
    test_transforms = transforms.Compose([
        *resize,
        transforms.ToTensor(),
        #CircleCrop(28, 28, radius = 14, center = (13.5, 13.5)),
        *normalize,
    ])

    train_dataset = DataClass(split="train", transform=train_transforms, download=download)
    valid_dataset = DataClass(split="val", transform=test_transforms, download=download)
    test_dataset = DataClass(split="test", transform=test_transforms, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    return train_loader, valid_loader, test_loader, img_channels, n_classes, device

class Metrics():
    def __init__(self):
        self.metrics = {}

    def update(self, **kwargs):
        '''Updates for a single value

        Args:
            **kwargs: key-value pairs to update
        '''

        for key, value in kwargs.items():
            self.metrics[key] = value

    def append(self, **kwargs):
        '''Appends for a list of values

        Args:
            **kwargs: key-value pairs to append
        '''

        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = [value]
            else:
                self.metrics[key].append(value)

    def __getitem__(self, key):
        if key not in self.metrics:
            raise KeyError(f"{key} not in metrics")
        return self.metrics[key]

def save_plots(metrics, path):
    epochs = range(1, len(metrics['train_losses']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, metrics['train_losses'], label='train')
    axes[0].plot(epochs, metrics['valid_losses'], label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, metrics['valid_accuracies'])
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(epochs, metrics['valid_AUCROC'])
    axes[2].set_title('Validation AUC')
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main(args):
    train_loader, valid_loader, test_loader, img_channels, n_classes, device = load_data(args)
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
    model = PRESETS[args.model](img_channels, args.n_channels, n_classes, args.kernel_size, args.padding, args.num_layers, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Optional LR decay. Default 'none' reproduces every previously-published
    # run exactly (constant LR). 'plateau' halves the LR after `lr_patience`
    # epochs without validation-loss improvement, matching the protocol of
    # Veeling et al. (PCam). Needed on large datasets: with constant LR, 60
    # epochs over 262k images is ~123k Adam steps and validation oscillates
    # heavily rather than converging.
    scheduler = None
    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.lr_patience,
            min_lr=args.lr_min)
    elif args.lr_schedule == 'cosine':
        # Deterministic decay over the training budget. Preferred when the
        # validation signal is noisy, since 'plateau' triggers off that same
        # noisy metric and may fire erratically or never.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = Metrics()

    # initialize the metrics
    metrics.update(best_acc=0, best_auc=0, auc_on_best_acc=0, acc_on_best_auc=0, final_acc=0, final_auc=0)

    # resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (RuntimeError, ValueError) as e:
            print(f"  Warning: could not load optimizer state ({e}). Optimizer reset.")
        # Restore the LR schedule too. CosineAnnealingLR ties its shape to T_max and
        # its position to last_epoch; without this a resumed run rebuilds the schedule
        # from scratch, so the LR jumps back to its initial value and never reaches
        # eta_min -- a silently different trajectory from an uninterrupted run.
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"  Warning: could not load scheduler state ({e}). LR schedule restarts from epoch 0.")
        elif scheduler is not None:
            print("  Warning: checkpoint has no scheduler state (saved before this was tracked). "
                  "LR schedule restarts from epoch 0 and will not match an uninterrupted run.")
        metrics.metrics = checkpoint['metrics']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch} (checkpoint: {args.resume})")

    # training loop
    pbar = tqdm(range(start_epoch, args.epochs), desc=f'{args.model}', unit='epoch')
    for epoch in pbar:
        model.train()

        # start timing the epoch
        start_time = time.time()

        # loop through the batches
        batch_losses = []
        for i, (images, labels) in enumerate(train_loader):

            # images to configured device. NOTE: labels squeezed for every dataset?
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # end timing the epoch save the duration
        end_time = time.time()
        total_time = end_time - start_time
        train_loss = np.mean(batch_losses)

        # validation loop: evaluate at given interval or at the end
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()

            valid_loss, valid_acc, valid_prec, valid_rec, valid_auc = calculate_metrics(model, valid_loader, device, n_classes)
            pbar.set_postfix(train_loss=f'{train_loss:.4f}', val_acc=f'{valid_acc:.4f}', val_auc=f'{valid_auc:.4f}')
            metrics.append(train_losses=train_loss,
                           valid_losses=valid_loss,
                           valid_accuracies=valid_acc,
                           valid_precision=valid_prec,
                           valid_recall=valid_rec,
                           valid_AUCROC=valid_auc,
                           epoch_times=total_time)

            if scheduler is not None:
                # ReduceLROnPlateau consumes a metric; CosineAnnealingLR does not.
                if args.lr_schedule == 'plateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

            if valid_acc > metrics.metrics["best_acc"]:
                metrics.update(best_acc=valid_acc,
                               auc_on_best_acc=valid_auc)
                if args.save_on_best in ["acc", "acc_auc", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", args.run_name, f"best_acc_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.pt")
                    torch.save(model.state_dict(), path)

            if valid_auc > metrics.metrics["best_auc"]:
                metrics.update(best_auc=valid_auc,
                               acc_on_best_auc=valid_acc)
                if args.save_on_best in ["auc", "acc_auc", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", args.run_name, f"best_auc_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.pt")
                    torch.save(model.state_dict(), path)

            if epoch == args.epochs - 1:
                metrics.update(final_acc=valid_acc,
                               final_auc=valid_auc)
                if args.save_on_best in ["final", "all"]:
                    path = os.path.join("results", f"{args.data_flag}", args.run_name, f"final_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.pt")
                    torch.save(model.state_dict(), path)

            # save intermediate plot and resume checkpoint every plot_interval epochs
            if epoch % args.plot_interval == 0 and len(metrics.metrics.get('train_losses', [])) > 0:
                plot_path = os.path.join("results", f"{args.data_flag}", args.run_name, f"plots_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.png")
                save_plots(metrics.metrics, plot_path)

                resume_path = os.path.join("results", f"{args.data_flag}", args.run_name, f"resume_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'metrics': metrics.metrics,
                }, resume_path)

    # Save final metrics, plot, and resume checkpoint
    path = os.path.join("results", f"{args.data_flag}", args.run_name, f"metrics_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.json")
    with open(path, "w") as f:
        json.dump(metrics.metrics, f)

    plot_path = os.path.join("results", f"{args.data_flag}", args.run_name, f"plots_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.png")
    save_plots(metrics.metrics, plot_path)

    resume_path = os.path.join("results", f"{args.data_flag}", args.run_name, f"resume_{args.model}_{args.data_flag}_aug{args.augment}_run{args.run}.pt")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics.metrics,
    }, resume_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type=str, action="store", dest="data_flag", default="bloodmnist", help="The dataset to use: pathmnist, dermamnist, bloodmnist, etc...")
    parser.add_argument("--model", type=str, action="store", dest="model", default="VanillaCNN", help="The model to use")
    parser.add_argument("--batch_size", type=int, action="store", dest="batch_size", default=128, help="The batch size to use")
    parser.add_argument("--num_workers", type=int, action="store", dest="num_workers", default=4, help="DataLoader worker processes (0 = synchronous loading in main process)")
    parser.add_argument("--n_channels", type=int, action="store", dest="n_channels", default=32, help="The number of channels to use")
    parser.add_argument("--kernel_size", type=int, action="store", dest="kernel_size", default=3, help="The kernel size to use")
    parser.add_argument("--padding", type=int, action="store", dest="padding", default=1, help="The padding to use")
    parser.add_argument("--num_layers", type=int, action="store", dest="num_layers", default=5, help="The number of layers to use")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=100, help="The number of epochs to use")
    parser.add_argument("--lr", type=float, action="store", dest="lr", default=1e-4, help="The learning rate to use")
    parser.add_argument("--val_interval", type=int, action="store", dest="val_interval", default=1, help="The validation interval to use")
    parser.add_argument("--save_on_best", type=str, action="store", dest="save_on_best", default="acc", help="Whether to save on best: acc, auc, acc_auc, final, all")
    parser.add_argument("--run", type=int, action="store", dest="run", default=0, help="The run number to use")
    parser.add_argument("--augment", type=str, dest="augment", default="none", choices=["none", "discrete", "continuous"], help="Data augmentation: none, discrete (90deg rotations + flips), continuous (random angle + flips)")
    parser.add_argument("--weight_decay", type=float, dest="weight_decay", default=0.0, help="Weight decay for Adam optimizer")
    parser.add_argument("--lr_schedule", type=str, default="none", choices=["none", "plateau", "cosine"],
                        help="LR schedule. 'none' (default) = constant LR, reproduces all previously-published runs. 'cosine' = CosineAnnealingLR over --epochs (recommended: deterministic, robust to a noisy validation signal). 'plateau' = ReduceLROnPlateau on validation loss (factor 0.5); closer to Veeling et al.'s PCam protocol but triggers off the noisy metric itself.")
    parser.add_argument("--lr_patience", type=int, default=20,
                        help="Epochs without validation-loss improvement before halving LR (only used with --lr_schedule plateau). NOTE: Veeling et al. use 20, but their 'epoch' is 312 batches (~20k images), i.e. ~1.5 full passes over PCam -- so on full-dataset epochs the equivalent is ~2-3, not 20.")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Floor for the LR schedule, so repeated halvings cannot collapse the LR to zero")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (.pt file saved at end of a previous run)")
    parser.add_argument("--plot_interval", type=int, default=100, help="Save intermediate plot and resume checkpoint every N epochs")
    parser.add_argument("--no_normalize", action="store_true", default=False, help="Skip input normalization (use raw [0,1] tensor — matches original notebook)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for torch/numpy/random. seed=0 keeps the legacy run directory; nonzero seeds use run{N}_seed{S}.")
    parser.add_argument("--resize", type=int, default=None, help="If set, resize images to (resize, resize) before training. Required for exact D_4 equivariance when downsampling encounters odd intermediate sizes (e.g. use --resize 32 for SRE-Conv on DermaMNIST 28x28).")
    parser.add_argument("--escnn_pool_type", type=str, default="box", choices=["box", "antialiased"], help="Pooling operator for --escnn_pool. 'box' = 2x2 average, stride 2 (same operator as mk.Downsample, preserves D4). 'antialiased' = escnn PointwiseAvgPoolAntialiased, which MEASURED ~1e0 equivariance error at the first pool because it subsamples the even sublattice; kept only to reproduce that finding.")
    parser.add_argument("--escnn_pool", action="store_true", help="(TrivialIrrepMatrixECNN only) Insert escnn PointwiseAvgPoolAntialiased wherever the channel count doubles, matching MatrixMomentOriginal's 3 downsamples. Without this the ESCNN baseline runs at full resolution through every layer while the moment-kernel model downsamples, which confounds memory and per-epoch time comparisons. Default off so previously-published runs reproduce exactly.")
    parser.add_argument("--mk_downsample", action="store_true", help="(SREVanillaCNN only) Replace AvgPool2d(2,2) with parity-aware mk.Downsample. Gives exact D_4 equivariance on any input size, so the 32x32 resize is not needed.")
    parser.add_argument("--interp_mode", type=str, default="linear", choices=["linear", "cubic", "gaussian"],
                        help="(MatrixMomentOriginal/TrivialMomentOriginal/TrivialIrrepMomentOriginal only) Radial-profile parameterization: 'linear' (piecewise-linear interpolation between knots — original behavior), 'cubic' (natural cubic spline, smoother), 'gaussian' (fixed-center fixed-width Gaussian basis). Same parameter count for all three.")
    parser.add_argument("--sigma_factor", type=float, default=1.0, help="Only used with --interp_mode gaussian. Sigma of each Gaussian basis function = sigma_factor * knot_spacing. Default 1.0 (knot-spacing-wide Gaussians).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.run_name = f"run{args.run}" if args.seed == 0 else f"run{args.run}_seed{args.seed}"

    os.chdir(_script_dir)
    path = os.path.join(_script_dir, "results", f"{args.data_flag}", args.run_name)
    print(f"Path: {path}")
    os.makedirs(path, exist_ok=True)

    main(args)