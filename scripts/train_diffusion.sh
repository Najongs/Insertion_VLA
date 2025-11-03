torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode cache \


set -e  # Exit on error

echo "================================================================================"
echo "🚀 Starting Diffusion VLA Training (Stage 1)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Training Stage: Stage 1 (VL frozen)"
echo "  - Batch Size: 4"
echo "  - Gradient Accumulation: 8"
echo "  - Learning Rate: 1e-4"
echo "  - Epochs: 20"
echo "  - Diffusion Timesteps: 100"
echo "  - Sensor Window: 650 samples"
echo "  - GPUs: 4"
echo ""
echo "Data Weighting:"
echo "  - New datasets (Make_dataset/New_dataset): 3x weight"
echo "  - Priority old datasets: 2x weight"
echo ""
echo "================================================================================"
echo ""

# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 학습 실행
torchrun --nproc_per_node=4 \
  training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
  --dataset_dir /home/najo/NAS/VLA/dataset \
  --training-stage stage1 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 1e-4 \
  --epochs 20 \
  --diffusion_timesteps 100 \
  --sensor_loss_weight 2.0 \
  --val_split 0.05

echo ""
echo "================================================================================"
echo "✅ Training Completed!"
echo "================================================================================"
echo ""
echo "Checkpoints saved in:"
echo "  - ./checkpoints/diffusion_stage1_latest.pt"
echo "  - ./checkpoints/diffusion_stage1_best.pt"
echo ""
echo "Next steps:"
echo "  1. Check training logs in wandb"
echo "  2. Evaluate the model"
echo "  3. Run inference tests"
echo ""
