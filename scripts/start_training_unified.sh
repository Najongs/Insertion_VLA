#!/bin/bash

# ===============================================================================
# Unified Training Script for QwenVLA with Sensor Integration
# ===============================================================================
# Single script that supports both Diffusion and Regression training
# ===============================================================================

set -e  # Exit on error

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# ===============================================================================
# Regression Training - Cache Building
# ===============================================================================
train_regression_cache() {
    echo "================================================================================"
    echo "📦 Building VL Cache for Regression Training"
    echo "================================================================================"
    echo ""
    echo "This step extracts and caches VL model features."
    echo "Run this once before training, or when adding new data."
    echo ""
    echo "Configuration:"
    echo "  - Image resize: 640x360"
    echo "  - Cache directory: /home/najo/NAS/VLA/dataset/cache/qwen_vl_features"
    echo "  - Sensor Window: 650 samples"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        training/A5st_VLA_TRAIN_Unified.py \
        --model-type regression \
        --mode cache \
        --dataset_dir /home/najo/NAS/VLA/dataset \
        --sensor_enabled \
        --fusion_strategy concat \
        --image_resize_height 360 \
        --image_resize_width 640 \
        --num_workers 8

    echo ""
    echo "✅ Cache building completed!"
}

# ===============================================================================
# Diffusion Training
# ===============================================================================
train_diffusion() {
    echo "================================================================================"
    echo "🚀 Starting Diffusion VLA Training"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Model Type: Diffusion"
    echo "  - Batch Size: 4 per GPU"
    echo "  - Gradient Accumulation: 8 steps"
    echo "  - Effective Batch: $(($NUM_GPUS * 4 * 8)) samples"
    echo "  - Learning Rate: 1e-4"
    echo "  - Epochs: 20"
    echo "  - Diffusion Timesteps: 100"
    echo "  - Sensor Window: 650 samples"
    echo "  - VLM Reuse: 1x"
    echo ""
    echo "Data Weighting:"
    echo "  - New datasets: 3x"
    echo "  - Priority old datasets: 2x"
    echo "  - Regular old datasets: 1x"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        training/A5st_VLA_TRAIN_Unified.py \
        --model-type diffusion \
        --dataset_dir /home/najo/NAS/VLA/dataset \
        --batch_size 4 \
        --grad_accum 8 \
        --lr 1e-4 \
        --epochs 20 \
        --diffusion_timesteps 100 \
        --sensor_enabled \
        --sensor_loss_weight 2.0 \
        --fusion_strategy concat \
        --val_split 0.1 \
        --num_workers 4

    echo ""
    echo "✅ Diffusion Training Completed!"
    echo "Checkpoints: ./checkpoints/diffusion_*.pt"
}



# ===============================================================================
# Regression Training
# ===============================================================================
train_regression_train() {
    echo "================================================================================"
    echo "🚀 Starting Regression VLA Training"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Model Type: Regression"
    echo "  - Batch Size: 16 per GPU"
    echo "  - Gradient Accumulation: 16 steps"
    echo "  - Effective Batch: $(($NUM_GPUS * 16 * 16)) samples"
    echo "  - Base LR: 5e-5"
    echo "  - Sensor LR: 5e-4"
    echo "  - Epochs: 100"
    echo "  - Sensor Window: 650 samples"
    echo "  - VLM Reuse: 3x"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        training/A5st_VLA_TRAIN_Unified.py \
        --model-type regression \
        --mode train \
        --dataset_dir /home/najo/NAS/VLA/dataset \
        --batch_size 16 \
        --grad_accum 16 \
        --lr 5e-5 \
        --sensor_lr 5e-4 \
        --min_lr 1e-6 \
        --epochs 100 \
        --sensor_enabled \
        --sensor_loss_weight 2.0 \
        --fusion_strategy concat \
        --image_resize_height 360 \
        --image_resize_width 640 \
        --val_split 0.05 \
        --num_workers 4

    echo ""
    echo "✅ Regression Training Completed!"
    echo "Checkpoints: ./checkpoints/regression_*.pt"
}

# ===============================================================================
# Main Menu
# ===============================================================================

echo ""
echo "================================================================================"
echo "🤖 Unified QwenVLA Training Script"
echo "================================================================================"
echo ""
echo "Please select training approach:"
echo ""
echo "1) Diffusion training (Recommended)"
echo "   - Better action prediction"
echo "   - No cache building needed"
echo "   - VLM reuse: 1x"
echo ""
echo "2) Regression training - Build cache"
echo "   - Build VL cache first (one-time)"
echo "   - Faster training after cache built"
echo ""
echo "3) Regression training - Train"
echo "   - Train with pre-built cache"
echo "   - VLM reuse: 3x (faster inference)"
echo ""
echo "4) Quick start: Diffusion (direct run)"
echo ""
echo "================================================================================"
echo ""

# Command line argument handling
if [ "$1" == "diffusion" ]; then
    train_diffusion
elif [ "$1" == "regression-cache" ]; then
    train_regression_cache
elif [ "$1" == "regression-train" ]; then
    train_regression_train
elif [ "$1" == "quick" ]; then
    train_diffusion
else
    # Interactive menu
    read -p "Enter choice [1-4]: " choice

    case $choice in
        1)
            train_diffusion
            ;;
        2)
            train_regression_cache
            ;;
        3)
            train_regression_train
            ;;
        4)
            train_diffusion
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

echo ""
echo "================================================================================"
echo "🎉 Training Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs in wandb"
echo "     - Diffusion: QwenVLA-Unified-Diffusion"
echo "     - Regression: QwenVLA-Unified-Regression"
echo "  2. Find checkpoints in ./checkpoints/"
echo "     - Diffusion: diffusion_epoch*.pt, diffusion_recent.pt"
echo "     - Regression: regression_best.pt, regression_latest.pt"
echo "  3. Run inference tests with trained model"
echo ""
echo "Usage examples:"
echo "  ./scripts/start_training_unified.sh diffusion"
echo "  ./scripts/start_training_unified.sh regression-cache"
echo "  ./scripts/start_training_unified.sh regression-train"
echo ""
