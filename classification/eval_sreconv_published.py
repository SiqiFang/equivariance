#!/usr/bin/env python3
"""Evaluate a published-architecture SRE-ResNet18 checkpoint under our
8-orientation worst-case protocol (4 rotations × 2 reflections, joint).

This is the published SRE-Conv model (Du et al. 2025, ~3.9M params), trained
via their main.py with --model-type sre_resnet18 --sre-conv-size-list 9 9 5 5.
Their published evaluation reports rotation and flip metrics separately;
we re-evaluate the same checkpoint under our stricter protocol to make a
fair comparison with the moment kernel.

Usage:
    python3 eval_sreconv_published.py \\
        --checkpoint ~/sreconv_published/src/checkpoint/.../sre_resnet18_best.ckpt \\
        --data_flag dermamnist

Output mirrors benchmark_eval.py --mode standard:
  acc, wc_acc, wc_drop, auc, wc_auc.
"""
import argparse
import json
import os
import sys

# SRE-Conv lives as a source folder inside the cloned repo (not pip-installed).
# Add the repo's src/ to sys.path so `from SRE_Conv import ...` works regardless
# of which directory we run this script from. Override SRE_REPO_DIR via env var
# if your clone lives elsewhere.
SRE_REPO_DIR = os.environ.get("SRE_REPO_DIR", os.path.expanduser("~/sreconv_published"))
SRE_SRC_DIR = os.path.join(SRE_REPO_DIR, "src")
if os.path.isdir(SRE_SRC_DIR) and SRE_SRC_DIR not in sys.path:
    sys.path.insert(0, SRE_SRC_DIR)

from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from SRE_Conv import sre_resnet18, SRE_Conv2d


# Defaults match the published SRE-Conv command in train_sreconv_published.sh
# (--model-type sre_resnet18 --sre-conv-size-list 9 9 5 5 --res-keep-conv1).
# Mirrors the override block in their main.py:
#     model = sre_resnet18(...)
#     if args.res_keep_conv1:
#         model.conv1 = SRE_Conv2d(in, inplanes, kernel_size=5, stride=1, padding=2, bias=False)
#     model.maxpool = nn.Identity()
# i.e., maxpool is ALWAYS replaced with Identity (removes first downsampling so
# small-image features keep their spatial resolution); conv1 is replaced with a
# kernel=5 stride=1 SRE_Conv2d ONLY when --res-keep-conv1 is passed.
DEFAULT_CONFIG = {
    "inplanes": 64,
    "sre_conv_size": [9, 9, 5, 5],
    "large_conv": False,
    "sre_k": None,
    "res_keep_conv1": True,
}


def build_model(num_classes, in_channels, cfg):
    """Replicate SRE-Conv main.py model construction. The config dict comes
    either from DEFAULT_CONFIG or the sidecar JSON written by
    train_sreconv_published.sh, so any mismatch in training args surfaces as
    a load error rather than as silent random-weight inference.

    Reproduces the exact override block from their main.py: maxpool is ALWAYS
    replaced with nn.Identity(); conv1 is replaced with
    SRE_Conv2d(kernel=5, stride=1, padding=2, bias=False) only when
    --res-keep-conv1 was passed at training time.
    """
    model = sre_resnet18(
        num_classes=num_classes,
        inplanes=cfg["inplanes"],
        sre_conv_size=cfg["sre_conv_size"],
        large_conv=cfg["large_conv"],
        sre_k=cfg["sre_k"],
        in_channels=in_channels,
    )
    if cfg.get("res_keep_conv1", False):
        model.conv1 = SRE_Conv2d(
            in_channels, cfg["inplanes"],
            kernel_size=5, stride=1, padding=2, bias=False,
        )
    model.maxpool = nn.Identity()
    return model


def make_test_loader(data_flag, batch_size=128, num_workers=2):
    """MedMNIST test split with SRE-Conv's exact preprocessing
    (Resize 28→32 + ToTensor + Normalize((0.5,),(0.5,)))."""
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_set = DataClass(split="test", transform=transform, download=True)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_classes = len(info["label"])
    in_channels = info["n_channels"]
    return loader, n_classes, in_channels


def apply_orientation(x, rot, flip):
    """Apply one of 8 axis-aligned orientations to a (B, C, H, W) tensor.
    rot ∈ {0,1,2,3} = number of 90° CCW rotations. flip ∈ {0,1} = horizontal flip.
    """
    if flip:
        x = torch.flip(x, dims=[-1])
    if rot:
        x = torch.rot90(x, k=rot, dims=(-2, -1))
    return x


@torch.no_grad()
def evaluate_8_orient(model, loader, device, n_classes):
    """Run inference once per batch over all 8 orientations stacked into a
    single (8B, C, H, W) tensor; reshape back to (8, B, C). Computes acc,
    wc_acc, auc, wc_auc on the joint protocol.
    """
    model.eval()
    orient_indices = [(r, f) for r in range(4) for f in range(2)]

    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        y = (y.squeeze(-1) if y.ndim > 1 else y).to(device)
        all_labels.append(y)

        # (8, B, C, H, W) → (8B, C, H, W) → forward → (8, B, n_classes)
        xs = torch.stack([apply_orientation(x, r, f) for r, f in orient_indices], dim=0)
        B = x.size(0)
        logits = model(xs.view(8 * B, *x.shape[1:])).view(8, B, -1)
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=1).cpu()      # (8, N_test, C)
    labels = torch.cat(all_labels, dim=0).cpu()      # (N_test,)

    probs = torch.softmax(logits, dim=-1)
    correct_per_orient = (probs.argmax(dim=-1) == labels.unsqueeze(0))  # (8, N_test)

    acc = correct_per_orient.float().mean().item()
    wc_acc = correct_per_orient.all(dim=0).float().mean().item()

    labels_np = labels.numpy()
    aucs = []
    for o in range(8):
        try:
            aucs.append(roc_auc_score(
                labels_np, probs[o].numpy(), multi_class="ovr",
                average="macro", labels=list(range(n_classes))
            ))
        except ValueError:
            aucs.append(float("nan"))
    auc = float(np.mean(aucs))
    wc_auc = float(np.min(aucs))

    return {
        "acc": acc,
        "wc_acc": wc_acc,
        "wc_drop": acc - wc_acc,
        "auc": auc,
        "wc_auc": wc_auc,
        "n_test": len(labels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the .ckpt file saved by SRE-Conv's main.py")
    ap.add_argument("--data_flag", default="dermamnist")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--config",
                    help="Path to config JSON written by train_sreconv_published.sh "
                         "(model construction args). If omitted, looks for "
                         "<checkpoint>.config.json next to the checkpoint, then "
                         "falls back to defaults from train_sreconv_published.sh.")
    ap.add_argument("--output_json",
                    help="Path to write eval results as JSON. Defaults to "
                         "<checkpoint>.eval_8orient.json next to the checkpoint.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_path = args.config or args.checkpoint + ".config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Loaded config: {cfg_path}")
    else:
        cfg = DEFAULT_CONFIG
        print(f"WARNING: no config sidecar found at {cfg_path}; using defaults. "
              f"If training used non-default args, eval will fail at load_state_dict.")

    print(f"Loading test set: {args.data_flag}")
    loader, n_classes, in_channels = make_test_loader(
        args.data_flag, args.batch_size, args.num_workers)
    print(f"  classes: {n_classes}, in_channels: {in_channels}, n_test: {len(loader.dataset)}")

    print("Building model: published SRE-ResNet18")
    model = build_model(n_classes, in_channels, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)

    print("Running 8-orientation eval...")
    results = evaluate_8_orient(model, loader, device, n_classes)
    results["model"] = "sre_resnet18_published"
    results["data_flag"] = args.data_flag
    results["checkpoint"] = args.checkpoint
    results["n_params"] = n_params
    results["config"] = cfg

    print()
    print("=" * 60)
    print(f"Published SRE-ResNet18 ({args.data_flag}) — 8-orientation worst-case eval")
    print("=" * 60)
    print(f"  Accuracy:           {results['acc']:.4f}")
    print(f"  WC Accuracy:        {results['wc_acc']:.4f}")
    print(f"  WC drop:            {results['wc_drop']:+.4f}")
    print(f"  AUC:                {results['auc']:.4f}")
    print(f"  WC AUC:             {results['wc_auc']:.4f}")
    print(f"  Test set size:      {results['n_test']}")
    print(f"  Parameters:         {n_params:,}")

    out_path = args.output_json or args.checkpoint + ".eval_8orient.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
