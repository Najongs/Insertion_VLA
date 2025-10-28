"""
Async VLA Inference Example

Demonstrates real-time asynchronous inference with:
- VLM running at ~3.33 Hz (300ms period)
- Action Expert running at 10 Hz (100ms period)
- Sensor data streaming at 650 Hz

Usage:
    python examples/async_inference_example.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import time
import sys
import threading
from pathlib import Path
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from models.model_with_sensor_async import AsyncQwenVLAWithSensor


class AsyncVLAInference:
    """
    Async VLA inference engine with TRUE asynchronous VLM execution

    Key features:
    - VLM runs in background thread (~311ms per execution)
    - Action Expert runs in main loop at 10 Hz (100ms period)
    - VL features are updated asynchronously without blocking Action Expert
    - Thread-safe VL feature cache with locking
    """

    def __init__(
        self,
        model,
        vlm_update_hz=0.59,  # 🔥 Updated for multi-view 5 images (~1484ms)
        action_expert_hz=10.0,
        sensor_hz=650,
        device="cuda",
    ):
        self.model = model.to(device)
        self.model.eval()

        # 🔥 IMPORTANT: Disable cache for real-time inference
        if hasattr(self.model, 'cache_mode'):
            self.model.cache_mode = "off"
            print("   ⚠️  VL cache disabled for real-time inference")

        self.device = device

        # Timing parameters
        self.vlm_update_period = 1.0 / vlm_update_hz  # ~300ms
        self.action_expert_period = 1.0 / action_expert_hz  # 100ms
        self.sensor_period = 1.0 / sensor_hz  # ~1.54ms

        # Calculate sensor window size
        self.sensor_window_size = int(sensor_hz * self.action_expert_period)  # 65 samples

        # 🔥 Thread-safe VL feature cache
        self.vl_feature_cache = None
        self.vl_cache_lock = threading.Lock()

        # VLM thread management
        self.vlm_thread = None
        self.vlm_running = False
        self.last_vlm_start_time = 0
        self.last_vlm_completion_time = 0

        # Statistics
        self.vlm_execution_times = deque(maxlen=100)
        self.action_execution_times = deque(maxlen=100)

        print(f"🚀 AsyncVLAInference initialized (TRUE async mode):")
        print(f"   VLM target period: {self.vlm_update_period*1000:.0f}ms ({vlm_update_hz:.2f} Hz)")
        print(f"   VLM actual time: ~1484ms (multi-view 5 images)")
        print(f"   Action Expert: {self.action_expert_period*1000:.0f}ms ({action_expert_hz:.2f} Hz)")
        print(f"   Sensor window: {self.sensor_window_size} samples")
        print(f"   VL feature reuse: ~{int(self.vlm_update_period / self.action_expert_period)}x")

    def _vlm_worker(self, instruction, image_paths):
        """
        VLM worker thread - runs in background

        This function executes the expensive VLM inference (~311ms)
        without blocking the main action prediction loop
        """
        try:
            start_time = time.time()

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl_features = self.model.extract_vl_features(
                    text_inputs=[instruction],
                    image_inputs=[image_paths],
                )

            # Thread-safe cache update
            with self.vl_cache_lock:
                self.vl_feature_cache = vl_features
                self.last_vlm_completion_time = time.time()

            elapsed = (time.time() - start_time) * 1000
            self.vlm_execution_times.append(elapsed)

            # print(f"   ✅ VLM completed in {elapsed:.1f}ms")

        except Exception as e:
            print(f"   ❌ VLM error: {e}")
        finally:
            self.vlm_running = False

    def should_start_vlm_update(self, current_time):
        """
        Check if we should start a new VLM update

        Only start if:
        1. No VLM is currently running
        2. Enough time has passed since last VLM start
        """
        if self.vlm_running:
            return False

        time_since_last_start = current_time - self.last_vlm_start_time
        return time_since_last_start >= self.vlm_update_period

    def start_vlm_update(self, instruction, image_paths):
        """
        Start VLM update in background thread (non-blocking)

        Args:
            instruction: Text instruction
            image_paths: List of image paths (multi-view)
        """
        if self.vlm_running:
            print("   ⚠️  VLM already running, skipping update")
            return

        self.vlm_running = True
        self.last_vlm_start_time = time.time()

        # Start VLM in background thread
        self.vlm_thread = threading.Thread(
            target=self._vlm_worker,
            args=(instruction, image_paths),
            daemon=True
        )
        self.vlm_thread.start()

    def predict_action(self, sensor_window):
        """
        Predict action using cached VL features and current sensor data
        (Fast operation, ~20-30ms)

        Args:
            sensor_window: (sensor_window_size, 1026) numpy array

        Returns:
            action: (7,) numpy array - predicted 6D pose delta + gripper
            or None if VL features not available yet
        """
        start_time = time.time()

        # Thread-safe access to VL cache
        with self.vl_cache_lock:
            if self.vl_feature_cache is None:
                return None  # VLM not ready yet
            vl_features = self.vl_feature_cache

        # Prepare sensor data
        sensor_tensor = torch.from_numpy(sensor_window).unsqueeze(0).to(
            device=self.device, dtype=torch.bfloat16
        )  # (1, window_size, 1026)

        # Dummy z_chunk (will be added to delta)
        z_chunk = torch.zeros(1, 8, 7, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions, _ = self.model.predict_actions_with_cached_vl(
                vl_features=vl_features,
                z_chunk=z_chunk,
                sensor_data=sensor_tensor,
            )

        # Return first action (1-step ahead)
        action = pred_actions[0, 0].cpu().numpy()  # (7,)

        elapsed = (time.time() - start_time) * 1000
        self.action_execution_times.append(elapsed)

        return action

    def run_async_control_loop(
        self,
        instruction,
        image_paths,
        sensor_stream_fn,
        duration_seconds=5.0,
    ):
        """
        Run TRUE async control loop

        VLM runs in background thread (~311ms) while Action Expert
        continues running at 10 Hz in main loop.

        Args:
            instruction: Text instruction
            image_paths: Image paths (updated periodically)
            sensor_stream_fn: Function that returns current sensor window
            duration_seconds: How long to run the loop

        Returns:
            actions: List of predicted actions
            timings: Dict with timing statistics
        """
        print(f"\n🎮 Starting TRUE async control loop for {duration_seconds}s...")
        print(f"   VLM runs in background thread (~311ms)")
        print(f"   Action Expert runs at 10 Hz in main loop (~100ms period)")

        actions = []
        action_timestamps = []
        vlm_start_times = []
        vlm_completion_times = []

        start_time = time.time()
        last_action_time = start_time - self.action_expert_period  # Allow immediate first action

        step_count = 0
        vlm_start_count = 0
        skipped_actions = 0  # Count actions skipped due to no VL features

        # 🔥 Start first VLM update immediately
        print(f"   🔄 Starting initial VLM update...")
        self.start_vlm_update(instruction, image_paths)
        vlm_start_times.append(time.time())

        while (time.time() - start_time) < duration_seconds:
            current_time = time.time()

            # 🔥 Start new VLM update if needed (non-blocking)
            if self.should_start_vlm_update(current_time):
                self.start_vlm_update(instruction, image_paths)
                vlm_start_times.append(current_time)
                vlm_start_count += 1
                print(f"   🔄 [Step {step_count}] Starting VLM update #{vlm_start_count} (background)")

            # 🔥 Action prediction at 10 Hz (main loop)
            if (current_time - last_action_time) >= self.action_expert_period:
                # Get current sensor window
                sensor_window = sensor_stream_fn()

                # Predict action (fast, ~20-30ms)
                action = self.predict_action(sensor_window)

                if action is not None:
                    actions.append(action)
                    action_timestamps.append(current_time)
                    last_action_time = current_time
                    step_count += 1

                    if step_count % 10 == 0:
                        avg_action_time = np.mean(list(self.action_execution_times)) if self.action_execution_times else 0
                        avg_vlm_time = np.mean(list(self.vlm_execution_times)) if self.vlm_execution_times else 0
                        vlm_status = "running" if self.vlm_running else "idle"
                        print(f"   ⚡ [Step {step_count}] Action: {avg_action_time:.1f}ms | VLM: {avg_vlm_time:.1f}ms ({vlm_status})")
                else:
                    # VL features not ready yet
                    skipped_actions += 1
                    if skipped_actions == 1:
                        print(f"   ⏳ [Step {step_count}] Waiting for first VLM completion...")

            # Small sleep to avoid busy waiting
            time.sleep(0.001)

        # Wait for last VLM to complete
        if self.vlm_thread and self.vlm_thread.is_alive():
            print(f"   ⏳ Waiting for last VLM to complete...")
            self.vlm_thread.join(timeout=1.0)

        total_time = time.time() - start_time

        # Count completed VLM updates
        vlm_completion_count = len(self.vlm_execution_times)

        # Statistics
        timings = {
            "total_time": total_time,
            "action_count": len(actions),
            "skipped_actions": skipped_actions,
            "vlm_started": len(vlm_start_times),
            "vlm_completed": vlm_completion_count,
            "avg_action_hz": len(actions) / total_time,
            "avg_vlm_hz": vlm_completion_count / total_time,
            "avg_vlm_time_ms": np.mean(list(self.vlm_execution_times)) if self.vlm_execution_times else 0,
            "avg_action_time_ms": np.mean(list(self.action_execution_times)) if self.action_execution_times else 0,
            "median_vlm_time_ms": np.median(list(self.vlm_execution_times)) if self.vlm_execution_times else 0,
            "median_action_time_ms": np.median(list(self.action_execution_times)) if self.action_execution_times else 0,
        }

        print(f"\n📊 Control loop statistics:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Actions predicted: {len(actions)} ({timings['avg_action_hz']:.2f} Hz)")
        print(f"   Actions skipped (no VL): {skipped_actions}")
        print(f"   VLM updates started: {timings['vlm_started']}")
        print(f"   VLM updates completed: {vlm_completion_count} ({timings['avg_vlm_hz']:.2f} Hz)")
        print(f"   Avg VLM time: {timings['avg_vlm_time_ms']:.1f}ms (median: {timings['median_vlm_time_ms']:.1f}ms)")
        print(f"   Avg Action time: {timings['avg_action_time_ms']:.1f}ms (median: {timings['median_action_time_ms']:.1f}ms)")

        return actions, timings


def create_dummy_sensor_stream(sensor_window_size=65):
    """Create dummy sensor stream for testing"""
    def sensor_stream_fn():
        # Generate dummy sensor data
        force = np.random.randn(sensor_window_size, 1).astype(np.float32) * 0.1
        alines = np.random.randn(sensor_window_size, 1025).astype(np.float32) * 0.01
        sensor_window = np.concatenate([force, alines], axis=1)  # (65, 1026)
        return sensor_window

    return sensor_stream_fn


def main():
    parser = argparse.ArgumentParser(description="Async VLA Inference Example")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration of control loop in seconds")
    parser.add_argument("--vlm-hz", type=float, default=0.59,
                        help="VLM update frequency (Hz, default: 0.59 for multi-view 5)")
    parser.add_argument("--action-hz", type=float, default=10.0,
                        help="Action Expert frequency (Hz)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Async VLA Inference Example")
    print(f"   Device: {device}")

    # Create model
    print("\n📦 Loading model...")
    model = AsyncQwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        finetune_vl="none",
        sensor_enabled=True,
        sensor_temporal_length=65,
        vlm_reuse_count=3,
    )

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"📥 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("   ✅ Checkpoint loaded")
    else:
        print("   ℹ️  No checkpoint provided, using random weights")

    # Create inference engine
    inference = AsyncVLAInference(
        model=model,
        vlm_update_hz=args.vlm_hz,
        action_expert_hz=args.action_hz,
        device=device,
    )

    # Dummy inputs
    instruction = "Perform needle insertion task with precision."
    image_paths = [
        "/path/to/view1.jpg",  # Dummy path
        "/path/to/view2.jpg",
    ]

    # Create dummy sensor stream
    sensor_stream_fn = create_dummy_sensor_stream(sensor_window_size=65)

    # Run control loop
    try:
        actions, timings = inference.run_async_control_loop(
            instruction=instruction,
            image_paths=image_paths,
            sensor_stream_fn=sensor_stream_fn,
            duration_seconds=args.duration,
        )

        print(f"\n✅ Async inference completed successfully!")
        print(f"   Generated {len(actions)} actions")

        # Show sample actions
        if actions:
            print(f"\n📋 Sample actions (first 3):")
            for i, action in enumerate(actions[:3]):
                print(f"   Action {i}: {action}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
