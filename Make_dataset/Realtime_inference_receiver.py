#!/usr/bin/env python3
"""
Real-time Inference Receiver with Data Logging
Receives multi-modal data (cameras, robot state, sensor) and performs real-time VLA inference

Key Features:
- Real-time data reception via ZMQ/UDP (same as Total_reciver.py)
- On-the-fly model inference with multi-view images + sensor data
- Circular buffer management for temporal sensor data (650 samples at 650Hz = 1 second)
- Low-latency inference pipeline
- Optional data saving for verification (images, sensor data, robot state, actions)

Data Sources:
- Cameras: 5 views (ZED left x4 + OAK x1) via ZMQ PULL (port 5555)
- Robot State: Joint angles + EE pose via ZMQ SUB (port 5556)
- Sensor Data: Force + OCT A-scan via UDP (port 9999)

Usage:
    # Inference only (adaptive rate enabled by default)
    python Realtime_inference_receiver.py --checkpoint checkpoints/best_model.pth

    # Inference + save all data for verification
    python Realtime_inference_receiver.py --checkpoint checkpoints/best_model.pth --save-data

    # Set target inference rate (will auto-adjust if model is slower)
    python Realtime_inference_receiver.py --checkpoint checkpoints/best_model.pth --inference-rate 1.5

    # Disable adaptive rate (force fixed rate even if too fast)
    python Realtime_inference_receiver.py --checkpoint checkpoints/best_model.pth --inference-rate 1.0 --no-adaptive

Performance Notes:
    - Model automatically measures inference time and adjusts rate to prevent overruns
    - If inference takes 400ms, max sustainable rate is ~2.2Hz (with 15% safety margin)
    - Overrun warnings appear when inference takes longer than target interval
    - Use --no-adaptive only for testing/benchmarking fixed rates
"""

import os, time, json, cv2, zmq, numpy as np, torch
import threading, argparse, signal, csv
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import struct

# Import VLA model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model_with_sensor import QwenVLAWithSensor


# ==============================
# Configuration
# ==============================
class Config:
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    ACTION_DIM = 7  # 6 joints + 1 gripper
    HORIZON = 8  # Action prediction horizon
    HIDDEN_DIM = 1024
    CACHE_DIR = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"

    # Sensor settings
    SENSOR_ENABLED = True
    SENSOR_TEMPORAL_LENGTH = 650  # 1 second at 650Hz
    SENSOR_INPUT_CHANNELS = 1026  # 1 force + 1025 A-scan
    FUSION_STRATEGY = 'concat'  # 'concat', 'cross_attention', 'gated'

    # Network settings (same as Total_reciver.py)
    ZMQ_CAM_PULL_PORT = 5555
    ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111"
    ZMQ_ROBOT_PUB_PORT = 5556
    ZMQ_ROBOT_TOPIC = b"robot_state"
    SENSOR_UDP_PORT = 9999
    SENSOR_UDP_IP = "0.0.0.0"
    SENSOR_BUFFER_SIZE = 4 * 1024 * 1024

    # Sensor packet format (same as Total_reciver.py)
    SENSOR_NXZRt = 1025
    SENSOR_PACKET_HEADER_FORMAT = '<ddf'  # ts, send_ts, force
    SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)
    SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
    SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)
    SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE
    SENSOR_CALIBRATION_COUNT = 50

    # Robot packet format
    ROBOT_PAYLOAD_FORMAT = '<ddf12f'  # ts, send_ts, force, 6 joints, 6 pose
    ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT)

    # Camera settings (ZED left only + OAK)
    ZED_SERIAL_TO_VIEW = {
        "41182735": "View1",  # ZED 1 left
        "49429257": "View2",  # ZED 2 left
        "44377151": "View3",  # ZED 3 left
        "49045152": "View4"   # ZED 4 left
    }
    OAK_KEYWORD = "OAK"

    # Inference settings
    INFERENCE_RATE_HZ = 2.0  # Target inference rate (will auto-adjust based on actual speed)
    MIN_INFERENCE_INTERVAL = 0.1  # Minimum 100ms between inferences (max 10Hz)
    INFERENCE_BATCH_SIZE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ADAPTIVE_RATE = True  # Auto-adjust inference rate based on actual inference time

    # Logging settings
    STATUS_PERIOD = 1.0
    STALL_SEC = 5.0


# ==============================
# Async Image Writer (from Total_reciver.py)
# ==============================
class AsyncImageWriter(threading.Thread):
    """Asynchronous image writer to avoid blocking main loop"""
    def __init__(self, max_queue=5000):
        super().__init__(daemon=True)
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()
        self.written_count = 0

    def submit(self, path, img):
        if not self.stop_flag.is_set():
            try:
                self.q.put_nowait((path, img))
            except:
                pass  # Drop frame if queue full

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
                try:
                    cv2.imwrite(path, img)
                    self.written_count += 1
                except Exception as e:
                    print(f"[Writer] Error saving {path}: {e}")
                finally:
                    self.q.task_done()
            except Empty:
                if self.stop_flag.is_set() and self.q.empty():
                    break
                continue

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images...")
        self.stop_flag.set()
        self.q.join()
        print(f"🛑 Writer thread stopped. Total written: {self.written_count}")


# ==============================
# Multi-View Image Buffer
# ==============================
class MultiViewImageBuffer:
    """
    Manages latest images from all camera views
    Ensures we have synchronized multi-view images for inference
    """
    def __init__(self, required_views=None, save_dir=None, writer=None):
        self.required_views = required_views or ['View1', 'View2', 'View3', 'View4', 'View5']
        self.latest_images = {}  # view_name -> (img, timestamp)
        self.lock = threading.Lock()
        self.update_count = defaultdict(int)
        self.save_dir = save_dir
        self.writer = writer
        self.save_enabled = save_dir is not None and writer is not None

    def update(self, view_name: str, img: np.ndarray, timestamp: float, cam_name: str = ""):
        with self.lock:
            self.latest_images[view_name] = (img, timestamp)
            self.update_count[view_name] += 1

            # Save image if enabled
            if self.save_enabled:
                filename = f"{cam_name}_{timestamp:.3f}.jpg" if cam_name else f"{view_name}_{timestamp:.3f}.jpg"
                save_path = os.path.join(self.save_dir, view_name, filename)
                self.writer.submit(save_path, img)

    def get_multi_view_set(self) -> dict:
        """Get latest image set from all views"""
        with self.lock:
            return {
                view: (img.copy(), ts)
                for view, (img, ts) in self.latest_images.items()
            }

    def is_ready(self) -> bool:
        """Check if we have at least one image from each required view"""
        with self.lock:
            return all(view in self.latest_images for view in self.required_views)

    def get_oldest_timestamp(self) -> float:
        """Get the oldest timestamp among current images"""
        with self.lock:
            if not self.latest_images:
                return 0.0
            return min(ts for _, ts in self.latest_images.values())


# ==============================
# Sensor Data Circular Buffer
# ==============================
class SensorCircularBuffer:
    """
    Maintains a circular buffer of sensor data for temporal window (650 samples)
    Thread-safe operations for real-time updates
    """
    def __init__(self, max_length=650, channels=1026, save_buffer=None):
        self.max_length = max_length
        self.channels = channels
        self.buffer = deque(maxlen=max_length)
        self.lock = threading.Lock()
        self.save_buffer = save_buffer  # Optional: list to save all data

    def add_samples(self, samples: list):
        """Add multiple samples (from UDP batch)"""
        with self.lock:
            for sample in samples:
                # Sample format: {'timestamp': float, 'send_timestamp': float, 'force': float, 'aline': np.array(1025,)}
                # Combine into (1026,) vector
                force = np.array([sample['force']], dtype=np.float32)
                aline = sample['aline'].astype(np.float32)
                combined = np.concatenate([force, aline])  # (1026,)
                self.buffer.append(combined)

                # Save to permanent buffer if enabled
                if self.save_buffer is not None:
                    self.save_buffer.append({
                        'timestamp': sample['timestamp'],
                        'send_timestamp': sample['send_timestamp'],
                        'force': sample['force'],
                        'aline': sample['aline']
                    })

    def get_tensor(self) -> torch.Tensor:
        """Get current buffer as torch tensor (T, C) with padding if needed"""
        with self.lock:
            if len(self.buffer) == 0:
                # Return zeros if no data yet
                return torch.zeros(self.max_length, self.channels, dtype=torch.float32)

            data = np.array(list(self.buffer), dtype=np.float32)  # (current_len, C)

            # Pad to max_length if needed
            if len(data) < self.max_length:
                pad_length = self.max_length - len(data)
                padding = np.zeros((pad_length, self.channels), dtype=np.float32)
                data = np.concatenate([padding, data], axis=0)

            return torch.from_numpy(data)  # (650, 1026)

    def is_ready(self) -> bool:
        """Check if buffer has enough samples (at least 50% full)"""
        with self.lock:
            return len(self.buffer) >= self.max_length // 2

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


# ==============================
# Real-time Inference Engine
# ==============================
class VLAInferenceEngine:
    """
    Manages VLA model inference with multi-modal inputs
    """
    def __init__(self, config: Config, checkpoint_path: str = None):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print(f"\n{'='*80}")
        print(f"Initializing VLA Inference Engine")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Sensor Enabled: {config.SENSOR_ENABLED}")
        print(f"Fusion Strategy: {config.FUSION_STRATEGY}")

        # Load model
        self.model = QwenVLAWithSensor(
            vl_model_name=config.MODEL_NAME,
            action_dim=config.ACTION_DIM,
            horizon=config.HORIZON,
            hidden_dim=config.HIDDEN_DIM,
            cache_dir=config.CACHE_DIR,
            sensor_enabled=config.SENSOR_ENABLED,
            sensor_input_channels=config.SENSOR_INPUT_CHANNELS,
            sensor_temporal_length=config.SENSOR_TEMPORAL_LENGTH,
            fusion_strategy=config.FUSION_STRATEGY
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Checkpoint loaded successfully")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Inference statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.recent_inference_times = deque(maxlen=10)  # Last 10 inference times
        self.lock = threading.Lock()

        print(f"{'='*80}\n")

    @torch.no_grad()
    def predict(self,
                images_dict: dict,
                sensor_data: torch.Tensor,
                text_prompt: str = "Perform needle insertion into the eye") -> dict:
        """
        Perform inference with multi-modal inputs

        Args:
            images_dict: {view_name: (img, timestamp)}
            sensor_data: (650, 1026) tensor
            text_prompt: Task description

        Returns:
            {
                'actions': (H, action_dim) predicted actions,
                'delta': (H, action_dim) action deltas,
                'inference_time': float,
                'timestamp': float,
                'image_timestamps': dict
            }
        """
        start_time = time.time()

        # Save images temporarily and prepare paths
        temp_dir = Path("/tmp/vla_inference")
        temp_dir.mkdir(exist_ok=True)

        image_paths = []
        image_timestamps = {}

        # Sort views for consistent ordering
        sorted_views = sorted(images_dict.keys())
        for view in sorted_views:
            img, ts = images_dict[view]
            temp_path = temp_dir / f"{view}_{ts:.3f}.jpg"
            cv2.imwrite(str(temp_path), img)
            image_paths.append(str(temp_path))
            image_timestamps[view] = ts

        # Prepare model inputs
        text_inputs = [text_prompt]
        image_inputs = [image_paths]  # List of lists for batch

        # Prepare z_chunk (initial action estimate - zeros for now)
        z_chunk = torch.zeros(1, self.config.HORIZON, self.config.ACTION_DIM,
                             dtype=torch.float32, device=self.device)

        # Prepare sensor data
        sensor_batch = sensor_data.unsqueeze(0).to(self.device)  # (1, 650, 1026)

        # Inference
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            pred_actions, delta = self.model(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                z_chunk=z_chunk,
                sensor_data=sensor_batch if self.config.SENSOR_ENABLED else None,
                cache_keys=[f"realtime_{time.time()}"],
                cache=False  # Disable cache for real-time inference
            )

        # Clean up temp files
        for path in image_paths:
            Path(path).unlink(missing_ok=True)

        inference_time = time.time() - start_time

        # Update statistics
        with self.lock:
            self.inference_count += 1
            self.total_inference_time += inference_time
            self.recent_inference_times.append(inference_time)

        return {
            'actions': pred_actions[0].cpu().numpy(),  # (H, action_dim)
            'delta': delta[0].cpu().numpy(),
            'inference_time': inference_time,
            'timestamp': time.time(),
            'image_timestamps': image_timestamps
        }

    def get_stats(self) -> dict:
        with self.lock:
            avg_time = self.total_inference_time / max(1, self.inference_count)

            # Calculate recent average (more responsive to current conditions)
            recent_avg_time = np.mean(self.recent_inference_times) if self.recent_inference_times else avg_time

            # Calculate max achievable rate (with 10% safety margin)
            max_rate = 1.0 / (recent_avg_time * 1.1) if recent_avg_time > 0 else 10.0

            return {
                'count': self.inference_count,
                'avg_inference_time': avg_time,
                'recent_avg_time': recent_avg_time,
                'total_time': self.total_inference_time,
                'max_achievable_rate': max_rate,
                'recommended_rate': min(max_rate, 5.0)  # Cap at 5Hz for safety
            }


# ==============================
# UDP Sensor Receiver (same as Total_reciver.py)
# ==============================
class SensorUDPReceiver(threading.Thread):
    """Receives sensor data via UDP and updates circular buffer"""
    def __init__(self, config: Config, sensor_buffer: SensorCircularBuffer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.sensor_buffer = sensor_buffer
        self.stop_event = stop_event
        self.clock_offset = None
        self.calibration_samples = []
        self.packet_count = 0

    def run(self):
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.SENSOR_BUFFER_SIZE)
            sock.bind((self.config.SENSOR_UDP_IP, self.config.SENSOR_UDP_PORT))
            sock.settimeout(1.0)
            print(f"✅ Sensor UDP Receiver started on port {self.config.SENSOR_UDP_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind UDP socket: {e}")
            return

        print(f"⏳ Calibrating sensor clock offset (first {self.config.SENSOR_CALIBRATION_COUNT} batches)...")

        while not self.stop_event.is_set():
            try:
                data, addr = sock.recvfrom(self.config.SENSOR_BUFFER_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[UDP Sensor] Receive error: {e}")
                continue

            recv_time = time.time()

            if len(data) < self.config.SENSOR_TOTAL_PACKET_SIZE:
                continue

            try:
                # Parse batch header
                num_packets = struct.unpack('<I', data[:4])[0]
                expected_size = 4 + (num_packets * self.config.SENSOR_TOTAL_PACKET_SIZE)

                if len(data) != expected_size or num_packets == 0:
                    continue

                # Parse packets
                records = []
                mv = memoryview(data)[4:]
                offset = 0
                last_send_ts = 0.0

                for _ in range(num_packets):
                    header = mv[offset:offset + self.config.SENSOR_PACKET_HEADER_SIZE]
                    ts, send_ts, force = struct.unpack(self.config.SENSOR_PACKET_HEADER_FORMAT, header)
                    offset += self.config.SENSOR_PACKET_HEADER_SIZE

                    aline_bytes = mv[offset:offset + self.config.SENSOR_ALINE_SIZE]
                    aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
                    offset += self.config.SENSOR_ALINE_SIZE

                    records.append({
                        'timestamp': ts,
                        'send_timestamp': send_ts,
                        'force': force,
                        'aline': aline
                    })
                    last_send_ts = send_ts

                # Clock calibration
                if self.clock_offset is None:
                    net_plus_offset = recv_time - last_send_ts
                    self.calibration_samples.append(net_plus_offset)

                    if len(self.calibration_samples) >= self.config.SENSOR_CALIBRATION_COUNT:
                        self.clock_offset = np.mean(self.calibration_samples)
                        print(f"\n✅ Sensor Clock Offset Calibrated: {self.clock_offset * 1000:.1f} ms\n")

                # Add to circular buffer
                self.sensor_buffer.add_samples(records)
                self.packet_count += num_packets

            except Exception as e:
                print(f"[ERROR] Sensor UDP unpack failed: {e}")
                continue

        sock.close()
        print("🛑 Sensor UDP Receiver stopped")


# ==============================
# Data Saving Functions
# ==============================
def save_robot_data_to_csv(data_list, filepath):
    """Save robot data to CSV"""
    if not data_list:
        print("💾 [Robot Save] No robot data to save.")
        return

    print(f"💾 Saving {len(data_list)} robot states to {filepath}")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                "recv_timestamp", "origin_timestamp", "send_timestamp", "force_placeholder",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"
            ])
            w.writerows(data_list)
        print(f"💾✅ Robot data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save robot data: {e}")


def save_sensor_data_to_npz(data_list, filepath):
    """Save sensor data to NPZ"""
    if not data_list:
        print("💾 [Sensor Save] No sensor data to save.")
        return

    print(f"💾 Saving {len(data_list)} sensor records to {filepath}")
    try:
        timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.float64)
        send_timestamps = np.array([d['send_timestamp'] for d in data_list], dtype=np.float64)
        forces = np.array([d['force'] for d in data_list], dtype=np.float32)
        alines = np.array([d['aline'] for d in data_list], dtype=np.float32)

        np.savez(filepath, timestamps=timestamps, send_timestamps=send_timestamps,
                forces=forces, alines=alines)
        print(f"💾✅ Sensor data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save sensor data: {e}")


def save_inference_results(results_list, filepath):
    """Save inference results to JSON"""
    if not results_list:
        print("💾 [Inference Save] No inference results to save.")
        return

    print(f"💾 Saving {len(results_list)} inference results to {filepath}")
    try:
        with open(filepath, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"💾✅ Inference results saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save inference results: {e}")


# ==============================
# Main Receiver with Inference
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Real-time VLA Inference Receiver with Data Logging')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--save-data', action='store_true', help='Save images, sensor data, robot state for verification')
    parser.add_argument('--inference-rate', type=float, default=2.0, help='Target inference rate in Hz (will auto-adjust if too fast)')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive rate adjustment (use fixed rate)')
    args = parser.parse_args()

    config = Config()
    config.INFERENCE_RATE_HZ = args.inference_rate
    config.ADAPTIVE_RATE = not args.no_adaptive

    # Setup output directory
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./inference_session_{session_time}")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories if saving data
    image_writer = None
    sensor_save_buffer = None
    robot_save_buffer = None

    if args.save_data:
        print(f"\n{'='*80}")
        print(f"📁 Data saving enabled: {output_dir}")
        print(f"{'='*80}\n")

        # Create view directories
        for view in ['View1', 'View2', 'View3', 'View4', 'View5']:
            (output_dir / view).mkdir(exist_ok=True)

        # Initialize image writer
        image_writer = AsyncImageWriter(max_queue=5000)
        image_writer.start()

        # Initialize save buffers
        sensor_save_buffer = []
        robot_save_buffer = []
    else:
        print(f"\n{'='*80}")
        print(f"📁 Inference only mode (no data saving)")
        print(f"📁 Inference results will be saved to: {output_dir}")
        print(f"{'='*80}\n")

    # Initialize components
    stop_event = threading.Event()
    image_buffer = MultiViewImageBuffer(
        save_dir=str(output_dir) if args.save_data else None,
        writer=image_writer
    )
    sensor_buffer = SensorCircularBuffer(
        max_length=config.SENSOR_TEMPORAL_LENGTH,
        channels=config.SENSOR_INPUT_CHANNELS,
        save_buffer=sensor_save_buffer
    )

    # Initialize inference engine
    inference_engine = VLAInferenceEngine(config, args.checkpoint)

    # ZMQ Setup
    ctx = zmq.Context.instance()

    # Camera socket
    cam_sock = ctx.socket(zmq.PULL)
    cam_sock.setsockopt(zmq.RCVHWM, 5000)
    cam_sock.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
    cam_sock.bind(f"tcp://0.0.0.0:{config.ZMQ_CAM_PULL_PORT}")
    print(f"✅ Camera PULL listening on port {config.ZMQ_CAM_PULL_PORT}")

    # Robot socket
    robot_sock = ctx.socket(zmq.SUB)
    robot_sock.setsockopt(zmq.RCVHWM, 100)
    robot_sock.connect(f"tcp://{config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")
    robot_sock.subscribe(config.ZMQ_ROBOT_TOPIC)
    print(f"✅ Robot SUB connected to {config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")

    # Poller
    poller = zmq.Poller()
    poller.register(cam_sock, zmq.POLLIN)
    poller.register(robot_sock, zmq.POLLIN)

    # Start sensor receiver
    sensor_thread = SensorUDPReceiver(config, sensor_buffer, stop_event)
    sensor_thread.start()

    # State tracking
    robot_state = None
    last_inference_time = time.time()
    last_status_print = time.time()
    inference_results = []
    cam_recv_count = defaultdict(int)

    # Adaptive inference rate tracking
    target_inference_interval = 1.0 / config.INFERENCE_RATE_HZ
    inference_overruns = 0  # Count how many times inference took longer than target interval

    # Signal handler
    def sigint_handler(sig, frame):
        print("\n🛑 Ctrl+C detected — Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    print(f"\n{'='*80}")
    print(f"Real-time Inference Started")
    print(f"{'='*80}")
    print(f"Target Inference Rate: {config.INFERENCE_RATE_HZ} Hz")
    print(f"Adaptive Rate: {'Enabled (will auto-adjust)' if config.ADAPTIVE_RATE else 'Disabled (fixed rate)'}")
    print(f"Device: {config.DEVICE}")
    print(f"Camera Views: 5 (ZED left x4 + OAK x1)")
    print(f"Data Saving: {'Enabled' if args.save_data else 'Disabled'}")
    print(f"\nWaiting for data from all sources...")
    print(f"Press Ctrl+C to stop\n")

    try:
        while not stop_event.is_set():
            # Poll for messages
            try:
                socks = dict(poller.poll(timeout=100))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[WARN] Poller error: {e}")
                time.sleep(0.1)
                continue

            now = time.time()

            # Process camera messages
            if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) < 2:
                            continue

                        meta_raw, jpg = parts[0], parts[1]
                        meta = json.loads(meta_raw.decode("utf-8"))

                        cam_name = meta.get("camera", "unknown")
                        timestamp = float(meta.get("timestamp", 0.0))

                        # Decode image
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        # Determine view name (left only for ZED)
                        view_name = None
                        cam_lower = cam_name.lower()

                        # Check if it's a left image from ZED
                        if "left" in cam_lower:
                            for serial, view in config.ZED_SERIAL_TO_VIEW.items():
                                if serial in cam_name:
                                    view_name = view
                                    break

                        # Check if it's OAK
                        if config.OAK_KEYWORD.lower() in cam_lower:
                            view_name = "View5"

                        if view_name:
                            image_buffer.update(view_name, img, timestamp, cam_name)
                            cam_recv_count[view_name] += 1

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[ERROR] Camera processing: {e}")
                        break

            # Process robot messages
            if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) != 2:
                            continue

                        topic, payload = parts[0], parts[1]
                        if len(payload) != config.ROBOT_PAYLOAD_SIZE:
                            continue

                        unpacked = struct.unpack(config.ROBOT_PAYLOAD_FORMAT, payload)
                        origin_ts, send_ts, force = unpacked[0:3]
                        joints, pose = unpacked[3:9], unpacked[9:15]

                        robot_state = {
                            'timestamp': origin_ts,
                            'joints': np.array(joints),
                            'pose': np.array(pose),
                            'recv_time': now
                        }

                        # Save robot data if enabled
                        if args.save_data:
                            robot_save_buffer.append([now] + list(unpacked))

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[ERROR] Robot processing: {e}")
                        break

            # Perform inference (adaptive or fixed rate)
            time_since_last_inference = now - last_inference_time

            # Check if it's time for next inference
            should_infer = time_since_last_inference >= target_inference_interval

            if should_infer:
                if image_buffer.is_ready() and sensor_buffer.is_ready():
                    # Get current multi-modal data
                    images_dict = image_buffer.get_multi_view_set()
                    sensor_tensor = sensor_buffer.get_tensor()

                    # Measure actual inference time
                    inference_start = time.time()
                    result = inference_engine.predict(images_dict, sensor_tensor)
                    actual_inference_time = result['inference_time']

                    # Check for overrun (inference took longer than target interval)
                    overrun_flag = ""
                    if actual_inference_time > target_inference_interval:
                        inference_overruns += 1
                        overrun_ms = (actual_inference_time - target_inference_interval) * 1000
                        overrun_flag = f" ⚠️ OVERRUN +{overrun_ms:.0f}ms"

                    # Log result
                    print(f"[INFERENCE #{len(inference_results)+1}] "
                          f"Actions[0]: [{result['actions'][0][0]:.3f}, {result['actions'][0][1]:.3f}, {result['actions'][0][2]:.3f}, ...] | "
                          f"Time: {actual_inference_time*1000:.1f}ms{overrun_flag} | "
                          f"Sensor: {sensor_buffer.size()}/650")

                    # Save inference result
                    inference_results.append({
                        'timestamp': result['timestamp'],
                        'actions': result['actions'].tolist(),
                        'delta': result['delta'].tolist(),
                        'inference_time': result['inference_time'],
                        'image_timestamps': result['image_timestamps'],
                        'robot_state': {
                            'joints': robot_state['joints'].tolist(),
                            'pose': robot_state['pose'].tolist(),
                            'timestamp': robot_state['timestamp']
                        } if robot_state else None
                    })

                    # Adaptive rate adjustment (after first few inferences)
                    if config.ADAPTIVE_RATE and len(inference_results) >= 5:
                        stats = inference_engine.get_stats()
                        recent_avg = stats['recent_avg_time']

                        # Adjust target interval based on recent performance
                        # Add 15% buffer to avoid constant overruns
                        adaptive_interval = max(recent_avg * 1.15, config.MIN_INFERENCE_INTERVAL)

                        if abs(adaptive_interval - target_inference_interval) > 0.05:  # Significant change
                            old_rate = 1.0 / target_inference_interval
                            new_rate = 1.0 / adaptive_interval
                            target_inference_interval = adaptive_interval
                            print(f"📊 Adaptive rate adjusted: {old_rate:.2f}Hz → {new_rate:.2f}Hz "
                                  f"(avg inference: {recent_avg*1000:.1f}ms)")

                    last_inference_time = now

                else:
                    # Data not ready
                    if time_since_last_inference >= 2.0:  # Print warning every 2s
                        print(f"[WAIT] Images: {image_buffer.is_ready()} "
                              f"({len(image_buffer.latest_images)}/{len(image_buffer.required_views)}) | "
                              f"Sensor: {sensor_buffer.is_ready()} ({sensor_buffer.size()}/650)")
                        last_inference_time = now

            # Status print
            if now - last_status_print >= config.STATUS_PERIOD:
                stats = inference_engine.get_stats()
                current_rate = 1.0 / target_inference_interval

                print(f"\n--- Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                print(f"Inferences: {stats['count']} | "
                      f"Recent avg: {stats.get('recent_avg_time', 0)*1000:.1f}ms | "
                      f"Current rate: {current_rate:.2f}Hz")

                if inference_overruns > 0:
                    overrun_pct = (inference_overruns / max(1, stats['count'])) * 100
                    print(f"⚠️  Overruns: {inference_overruns}/{stats['count']} ({overrun_pct:.1f}%) - "
                          f"Consider slower rate or check GPU load")

                if stats['count'] >= 3:
                    max_rate = stats.get('max_achievable_rate', 0)
                    print(f"📊 Max sustainable rate: ~{max_rate:.2f}Hz")

                print(f"Images recv: {', '.join([f'{v}:{cam_recv_count[v]}' for v in sorted(cam_recv_count.keys())])}")
                print(f"Sensor buffer: {sensor_buffer.size()}/{config.SENSOR_TEMPORAL_LENGTH}")

                if robot_state:
                    print(f"Robot: J1={robot_state['joints'][0]:.2f}°, Px={robot_state['pose'][0]:.2f}mm")

                if args.save_data and image_writer:
                    print(f"Writer queue: {image_writer.q.qsize()} | Written: {image_writer.written_count}")

                last_status_print = now

    finally:
        print(f"\n{'='*80}")
        print("Cleanup and Data Saving")
        print(f"{'='*80}\n")
        stop_event.set()

        # Wait for threads
        print("⏳ Waiting for sensor thread...")
        sensor_thread.join(timeout=2.0)

        # Save all data
        if args.save_data:
            # Save robot data
            if robot_save_buffer:
                robot_csv = output_dir / f"robot_state_{session_time}.csv"
                save_robot_data_to_csv(robot_save_buffer, str(robot_csv))

            # Save sensor data
            if sensor_save_buffer:
                sensor_npz = output_dir / f"sensor_data_{session_time}.npz"
                save_sensor_data_to_npz(sensor_save_buffer, str(sensor_npz))

            # Stop image writer
            if image_writer:
                image_writer.stop()
                image_writer.join()

        # Save inference results
        if inference_results:
            inference_json = output_dir / f"inference_results_{session_time}.json"
            save_inference_results(inference_results, str(inference_json))

        # Print final stats
        stats = inference_engine.get_stats()
        print(f"\n{'='*80}")
        print("Final Statistics")
        print(f"{'='*80}")
        print(f"Total inferences: {stats['count']}")
        print(f"Avg inference time: {stats['avg_inference_time']*1000:.1f}ms")
        print(f"Recent avg inference time: {stats.get('recent_avg_time', 0)*1000:.1f}ms")
        print(f"Total inference time: {stats['total_time']:.1f}s")

        if stats['count'] > 0:
            actual_avg_rate = stats['count'] / stats['total_time']
            print(f"Actual average rate: {actual_avg_rate:.2f}Hz")
            print(f"Max sustainable rate: {stats.get('max_achievable_rate', 0):.2f}Hz")

        if inference_overruns > 0:
            overrun_pct = (inference_overruns / max(1, stats['count'])) * 100
            print(f"⚠️  Total overruns: {inference_overruns} ({overrun_pct:.1f}%)")

        if args.save_data:
            print(f"\nData saved:")
            print(f"  Images: {image_writer.written_count if image_writer else 0}")
            print(f"  Robot states: {len(robot_save_buffer) if robot_save_buffer else 0}")
            print(f"  Sensor records: {len(sensor_save_buffer) if sensor_save_buffer else 0}")

        print(f"  Inference results: {len(inference_results)}")
        print(f"\n📁 Output directory: {output_dir}")
        print(f"{'='*80}\n")

        # Cleanup sockets
        cam_sock.close()
        robot_sock.close()
        ctx.term()

        print("✅ Shutdown complete")


if __name__ == "__main__":
    main()
