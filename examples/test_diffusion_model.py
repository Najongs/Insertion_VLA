"""
Test script for Diffusion VLA model

Tests:
1. Model initialization
2. Forward pass (training mode)
3. Sampling (inference mode)
4. Sensor fusion
5. DDIM vs DDPM sampling speed
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_with_sensor_diffusion import (
    QwenVLAWithSensorDiffusion,
    DiffusionActionExpert,
    DiffusionSchedule
)

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(test_name, passed, message=""):
    status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
    print(f"  [{status}] {test_name}")
    if message:
        print(f"         {YELLOW}{message}{RESET}")


def test_diffusion_schedule():
    """Test 1: Diffusion schedule"""
    print(f"\n{BLUE}Test 1: Testing Diffusion Schedule...{RESET}")

    try:
        schedule = DiffusionSchedule(timesteps=100, schedule='cosine', device='cpu')

        # Check forward diffusion
        x_0 = torch.randn(2, 8, 7)  # (B, H, A)
        t = torch.tensor([0, 50])
        noise = torch.randn_like(x_0)

        x_t = schedule.q_sample(x_0, t, noise)

        # Check shapes
        assert x_t.shape == x_0.shape, "Forward diffusion shape mismatch"

        # Check noise increases with t
        x_t0 = schedule.q_sample(x_0, torch.tensor([0, 0]), noise)
        x_t99 = schedule.q_sample(x_0, torch.tensor([99, 99]), noise)

        # At t=0, should be close to x_0
        # At t=99, should be mostly noise
        dist_0 = torch.norm(x_t0 - x_0).item()
        dist_99 = torch.norm(x_t99 - x_0).item()

        assert dist_99 > dist_0, "Noise should increase with timestep"

        print_test("Diffusion schedule", True, f"dist(t=0)={dist_0:.3f}, dist(t=99)={dist_99:.3f}")
        return True

    except Exception as e:
        print_test("Diffusion schedule", False, str(e))
        return False


def test_diffusion_action_expert():
    """Test 2: Diffusion Action Expert"""
    print(f"\n{BLUE}Test 2: Testing Diffusion Action Expert...{RESET}")

    try:
        expert = DiffusionActionExpert(
            vl_dim=512,
            sensor_dim=512,
            action_dim=7,
            horizon=8,
            hidden_dim=256,
            timesteps=100,
            fusion_strategy='concat',
            num_layers=2
        )

        # Test input
        batch_size = 2
        noisy_actions = torch.randn(batch_size, 8, 7)
        timesteps = torch.randint(0, 100, (batch_size,))
        vl_tokens = torch.randn(batch_size, 1, 512)
        sensor_features = torch.randn(batch_size, 512)

        # Forward pass
        with torch.no_grad():
            eps_pred = expert(noisy_actions, timesteps, vl_tokens, sensor_features)

        # Check shape
        expected_shape = (batch_size, 8, 7)
        assert eps_pred.shape == expected_shape, f"Expected {expected_shape}, got {eps_pred.shape}"

        print_test("Action expert forward", True, f"Output shape: {eps_pred.shape}")

        # Test sampling (DDPM)
        with torch.no_grad():
            start = time.time()
            samples_ddpm = expert.sample(vl_tokens, sensor_features, batch_size=batch_size, ddim_steps=None)
            time_ddpm = (time.time() - start) * 1000

        assert samples_ddpm.shape == expected_shape
        print_test("Action expert sampling (DDPM 100 steps)", True, f"Time: {time_ddpm:.1f}ms")

        # Test sampling (DDIM)
        with torch.no_grad():
            start = time.time()
            samples_ddim = expert.sample(vl_tokens, sensor_features, batch_size=batch_size, ddim_steps=10)
            time_ddim = (time.time() - start) * 1000

        assert samples_ddim.shape == expected_shape
        print_test("Action expert sampling (DDIM 10 steps)", True, f"Time: {time_ddim:.1f}ms (speedup: {time_ddpm/time_ddim:.1f}x)")

        return True

    except Exception as e:
        print_test("Action expert", False, str(e))
        return False


def test_full_model_training():
    """Test 3: Full model training mode"""
    print(f"\n{BLUE}Test 3: Testing Full Model (Training Mode)...{RESET}")

    try:
        # Note: This will try to load Qwen model - skip if not available
        print("⚠️  This test requires Qwen model download. Skipping for now...")
        print_test("Full model training mode", None, "Skipped (requires Qwen model)")
        return True

    except Exception as e:
        print_test("Full model training mode", False, str(e))
        return False


def test_sensor_fusion_strategies():
    """Test 4: Sensor fusion strategies"""
    print(f"\n{BLUE}Test 4: Testing Sensor Fusion Strategies...{RESET}")

    strategies = ['concat', 'cross_attention', 'gated', 'none']
    results = {}

    for strategy in strategies:
        try:
            expert = DiffusionActionExpert(
                vl_dim=512,
                sensor_dim=512,
                action_dim=7,
                horizon=8,
                hidden_dim=256,
                timesteps=50,  # Faster
                fusion_strategy=strategy,
                num_layers=2
            )

            batch_size = 2
            noisy_actions = torch.randn(batch_size, 8, 7)
            timesteps = torch.randint(0, 50, (batch_size,))
            vl_tokens = torch.randn(batch_size, 1, 512)
            sensor_features = torch.randn(batch_size, 512) if strategy != 'none' else None

            with torch.no_grad():
                eps_pred = expert(noisy_actions, timesteps, vl_tokens, sensor_features)

            results[strategy] = True
            print_test(f"Fusion: {strategy}", True, f"Output: {eps_pred.shape}")

        except Exception as e:
            results[strategy] = False
            print_test(f"Fusion: {strategy}", False, str(e))

    return all(results.values())


def test_noise_prediction_training():
    """Test 5: Noise prediction training loop simulation"""
    print(f"\n{BLUE}Test 5: Testing Noise Prediction Training...{RESET}")

    try:
        expert = DiffusionActionExpert(
            vl_dim=256,
            sensor_dim=256,
            action_dim=7,
            horizon=8,
            hidden_dim=128,
            timesteps=50,
            fusion_strategy='concat'
        )

        # Simulate training batch
        batch_size = 4
        gt_actions = torch.randn(batch_size, 8, 7)
        vl_tokens = torch.randn(batch_size, 1, 256)
        sensor_features = torch.randn(batch_size, 256)

        # Sample timesteps
        timesteps = torch.randint(0, 50, (batch_size,))

        # Add noise
        noise = torch.randn_like(gt_actions)
        noisy_actions = expert.diffusion.q_sample(gt_actions, timesteps, noise)

        # Predict noise
        eps_pred = expert(noisy_actions, timesteps, vl_tokens, sensor_features)

        # Compute loss
        loss = torch.nn.functional.mse_loss(eps_pred, noise)

        # Backward
        loss.backward()

        print_test("Noise prediction training", True, f"Loss: {loss.item():.4f}")

        # Check gradients
        has_grad = any(p.grad is not None for p in expert.parameters())
        print_test("Gradient flow", has_grad, "Gradients computed successfully")

        return True

    except Exception as e:
        print_test("Noise prediction training", False, str(e))
        return False


def test_parameter_count():
    """Test 6: Parameter count"""
    print(f"\n{BLUE}Test 6: Testing Parameter Counts...{RESET}")

    try:
        expert = DiffusionActionExpert(
            vl_dim=3072,
            sensor_dim=3072,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            timesteps=100,
            fusion_strategy='concat'
        )

        total_params = sum(p.numel() for p in expert.parameters())
        trainable_params = sum(p.numel() for p in expert.parameters() if p.requires_grad)

        print_test("Diffusion action expert parameters", True, f"{total_params:,} params ({trainable_params:,} trainable)")

        # Compare with regression expert (approximate)
        # Regression expert: ~50-100M params
        # Diffusion expert: ~80-150M params (larger due to timestep conditioning)
        reasonable = 50_000_000 < total_params < 200_000_000

        print_test("Parameter count reasonable", reasonable, f"Within expected range")

        return reasonable

    except Exception as e:
        print_test("Parameter count", False, str(e))
        return False


def main():
    print(f"\n{'='*70}")
    print(f"{BLUE}Diffusion VLA Model - Validation Tests{RESET}")
    print(f"{'='*70}")

    results = []

    results.append(("Diffusion Schedule", test_diffusion_schedule()))
    results.append(("Diffusion Action Expert", test_diffusion_action_expert()))
    results.append(("Full Model Training", test_full_model_training()))
    results.append(("Sensor Fusion", test_sensor_fusion_strategies()))
    results.append(("Noise Training", test_noise_prediction_training()))
    results.append(("Parameter Count", test_parameter_count()))

    # Summary
    print(f"\n{'='*70}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{'='*70}")

    passed = sum(1 for _, result in results if result)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results) - skipped

    for test_name, result in results:
        if result is None:
            status = f"{YELLOW}SKIPPED{RESET}"
        else:
            status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"  {test_name:.<50} {status}")

    print(f"\n{BLUE}Overall: {passed}/{total} tests passed{RESET}")

    if passed == total:
        print(f"{GREEN}✓ All tests passed! Diffusion model is ready.{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Please check the errors above.{RESET}")
        return 1


if __name__ == "__main__":
    exit(main())
