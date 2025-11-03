#!/bin/bash

# ===============================================================================
# Training Scripts for QwenVLA with Sensor Integration
# ===============================================================================
# This script provides commands for both training approaches:
# 1. Diffusion-based training (A5st_VLA_TRAIN_Diffusion_with_sensor.py)
# 2. Regression-based training (A5st_VLA_TRAIN_VL_Lora_with_sensor.py)
# ===============================================================================

set -e  # Exit on error

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# ===============================================================================
# Option 1: Diffusion-based Training
# ===============================================================================
# Recommended for better action prediction performance
# Uses diffusion model for action generation

train_diffusion() {
    echo "================================================================================"
    echo "🚀 Starting Diffusion VLA Training"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Model: QwenVLAWithSensorDiffusion"
    echo "  - VL Model: Frozen (cache-based)"
    echo "  - Training: Sensor Encoder + Diffusion Action Expert"
    echo "  - Batch Size: 4 per GPU"
    echo "  - Gradient Accumulation: 8 steps"
    echo "  - Effective Batch: $(($NUM_GPUS * 4 * 8)) samples"
    echo "  - Learning Rate: 1e-4"
    echo "  - Epochs: 20"
    echo "  - Diffusion Timesteps: 100"
    echo ""
    echo "Data Weighting:"
    echo "  - New datasets (Make_dataset/New_dataset): 3x weight"
    echo "  - Priority old datasets (Needle_insertion, White_silicone): 2x weight"
    echo "  - Regular old datasets: 1x weight"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
        --dataset_dir /home/najo/NAS/VLA/dataset \
        --batch_size 4 \
        --grad_accum 8 \
        --lr 1e-4 \
        --epochs 20 \
        --diffusion_timesteps 100 \
        --sensor_loss_weight 2.0 \
        --val_split 0.1

    echo ""
    echo "✅ Diffusion Training Completed!"
    echo "Checkpoints saved in ./checkpoints/"
}

# ===============================================================================
# Option 2: Regression-based Training
# ===============================================================================
# Traditional approach with direct action prediction
# Step 1: Build VL cache (once)
# Step 2: Train with cache

train_regression_cache() {
    echo "================================================================================"
    echo "📦 Building VL Cache"
    echo "================================================================================"
    echo ""
    echo "This step extracts and caches VL model features."
    echo "Run this once before training, or when adding new data."
    echo ""
    echo "Configuration:"
    echo "  - Image resize: 640x360"
    echo "  - Cache directory: /home/najo/NAS/VLA/dataset/cache/qwen_vl_features"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
        --mode cache \
        --sensor-enabled \
        --sensor-window-size 65 \
        --fusion-strategy concat \
        --image-resize-height 360 \
        --image-resize-width 640 \
        --num-workers 4

    echo ""
    echo "✅ Cache building completed!"
}

train_regression_train() {
    echo "================================================================================"
    echo "🚀 Starting Regression VLA Training"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Model: Not_freeze_QwenVLAWithSensor"
    echo "  - VL Model: Frozen (cache-based)"
    echo "  - Training: Sensor Encoder + Action Expert"
    echo "  - Batch Size: 16 per GPU"
    echo "  - Gradient Accumulation: 16 steps"
    echo "  - Effective Batch: $(($NUM_GPUS * 16 * 16)) samples"
    echo "  - Base LR: 5e-5"
    echo "  - Sensor LR: 5e-4"
    echo "  - Total Epochs: 100"
    echo ""
    echo "================================================================================"
    echo ""

    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
        --mode train \
        --batch-size 16 \
        --grad-accum-steps 16 \
        --lr 5e-5 \
        --sensor-lr 5e-4 \
        --min-lr 1e-6 \
        --warmup-ratio 0.03 \
        --hold-ratio 0.02 \
        --sched-on step \
        --sensor-enabled \
        --fusion-strategy concat \
        --sensor-loss-weight 2.0 \
        --image-resize-height 360 \
        --image-resize-width 640 \
        --num-workers 4

    echo ""
    echo "✅ Regression Training Completed!"
    echo "Checkpoints saved in ./checkpoints/"
}

# ===============================================================================
# Main Menu
# ===============================================================================

echo ""
echo "================================================================================"
echo "🤖 QwenVLA Training Script"
echo "================================================================================"
echo ""
echo "Please select training approach:"
echo ""
echo "1) Diffusion-based training (Recommended)"
echo "   - Better action prediction"
echo "   - No cache building needed"
echo "   - Longer inference time"
echo ""
echo "2) Regression-based training (Cache)"
echo "   - Build VL cache first"
echo "   - Faster inference"
echo ""
echo "3) Regression-based training (Train)"
echo "   - Train with pre-built cache"
echo ""
echo "4) Quick start: Diffusion training (direct run)"
echo ""
echo "================================================================================"
echo ""

# If argument provided, use it; otherwise show menu
if [ "$1" == "diffusion" ]; then
    train_diffusion
elif [ "$1" == "regression-cache" ]; then
    train_regression_cache
elif [ "$1" == "regression-train" ]; then
    train_regression_train
elif [ "$1" == "quick" ]; then
    # Quick start - just run diffusion
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
echo "🎉 All Done!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs in wandb"
echo "  2. Monitor checkpoints in ./checkpoints/"
echo "  3. Run inference tests with trained model"
echo ""
