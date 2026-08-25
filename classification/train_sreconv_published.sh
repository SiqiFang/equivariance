#!/bin/bash
# Train the published SRE-ResNet18 (3.9M params) on DermaMNIST using SRE-Conv's
# own training pipeline. After completion, evaluate under our 8-orientation
# worst-case protocol via eval_sreconv_published.py.
#
# Wall-clock: ~10-15 min on a single GPU (DermaMNIST is small; ResNet-18 forward
# at 28×28 is fast). 100 epochs matches the published paper (Du et al. 2025).
#
# Usage (from any working directory):
#   ./train_sreconv_published.sh                    # default: DermaMNIST, 100 epochs
#   EPOCHS=200 ./train_sreconv_published.sh          # override epochs
#   DATA_FLAG=bloodmnist ./train_sreconv_published.sh
#
# Outputs: checkpoint saved by their main.py (typically in their src/checkpoint/...).
# The eval script needs the path; set CKPT below or pass via --checkpoint.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_FLAG="${DATA_FLAG:-dermamnist}"
EPOCHS="${EPOCHS:-100}"
SRE_REPO_DIR="${SRE_REPO_DIR:-$HOME/sreconv_published}"

# ------------------------------------------------------------------
# Step 1: clone the SRE-Conv repo (if not already present)
# ------------------------------------------------------------------
if [ ! -d "$SRE_REPO_DIR" ]; then
    echo "=== Cloning SRE-Conv repo to $SRE_REPO_DIR ==="
    git clone https://github.com/XYPB/SRE-Conv.git "$SRE_REPO_DIR"
else
    echo "=== Using existing repo at $SRE_REPO_DIR ==="
    git -C "$SRE_REPO_DIR" pull --ff-only || echo "  (pull skipped, possibly offline)"
fi

# ------------------------------------------------------------------
# Step 2: train using their main.py with the published config
# ------------------------------------------------------------------
# From their README: --model-type sre_resnet18 --sre-conv-size-list 9 9 5 5
# Other defaults: medmnist_size=28 → resize to 32, batch_size=64, lr=1.0
# (Adadelta default), no augmentation flag → no rotation aug during training.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/cluster_logs/sreconv_published_${DATA_FLAG}_${EPOCHS}ep.log"
mkdir -p "$PROJECT_DIR/cluster_logs"

(
    cd "$SRE_REPO_DIR/src"
    echo "=== Training sre_resnet18 on $DATA_FLAG, $EPOCHS epochs ==="
    # Mirrors the published command from SRE-Conv's README.
    # --res-keep-conv1 (when passed) replaces the default sre_resnet18 conv1
    # with SRE_Conv2d(kernel=5, stride=1, padding=2, bias=False). main.py also
    # unconditionally sets maxpool=Identity (removes first downsampling).
    # eval_sreconv_published.py must mirror this exact override.
    python3 main.py \
        --med-mnist "$DATA_FLAG" \
        --model-type sre_resnet18 \
        --sre-conv-size-list 9 9 5 5 \
        -b 128 \
        --lr 2e-2 \
        --cos \
        --sgd \
        --eval-rot --eval-flip \
        --train-flip-p 0 \
        --log --cudnn \
        --moco-aug --translate-ratio 0.1 --translation \
        --save-model --save-best \
        --res-keep-conv1 \
        --epochs "$EPOCHS"
) 2>&1 | tee "$LOG_FILE"

# Locate the checkpoint produced by main.py — typically under
# $SRE_REPO_DIR/src/logs/<dataset>_<datetime>_<modeltype>_task_train_logs/ckpt/
# Search broadly under src/ to handle alternate save schemes.
CKPT="$(find "$SRE_REPO_DIR/src" -name "sre_resnet18_best.ckpt" -type f 2>/dev/null \
        -printf "%T@ %p\n" | sort -rn | head -1 | awk '{print $2}')"
[ -z "$CKPT" ] && CKPT="$(find "$SRE_REPO_DIR/src" -name "sre_resnet18_last.ckpt" -type f 2>/dev/null \
        -printf "%T@ %p\n" | sort -rn | head -1 | awk '{print $2}')"

# Write a config sidecar so eval_sreconv_published.py constructs the model
# with the same args we used here. Mismatch otherwise causes silent failures.
if [ -n "$CKPT" ]; then
    cat > "${CKPT}.config.json" <<JSON
{
  "inplanes": 64,
  "sre_conv_size": [9, 9, 5, 5],
  "large_conv": false,
  "sre_k": null,
  "res_keep_conv1": true,
  "epochs": $EPOCHS,
  "data_flag": "$DATA_FLAG"
}
JSON
    echo
    echo "=== Training done ==="
    echo "Checkpoint: $CKPT"
    echo "Config sidecar written: ${CKPT}.config.json"
    echo
    echo "To run our 8-orientation eval:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 eval_sreconv_published.py --checkpoint '$CKPT' --data_flag $DATA_FLAG"
else
    echo
    echo "=== Training done, but no checkpoint found ==="
    echo "Inspect $SRE_REPO_DIR/src/checkpoint/ manually."
fi
