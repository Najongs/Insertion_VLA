#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Data Collector for VLA Model Training

Optimizations for Not_freeze_QwenVLAWithSensor:
- Episode-based data collection
- Synchronized multi-view camera capture (5 views @ 5Hz)
- High-frequency sensor data (650Hz → 1s windows of 650 samples)
- Robot state at 100Hz
- Automatic data alignment and synchronization
- HDF5 format for efficient training data loading

Data Format:
    episode_YYYYMMDD_HHMMSS/
    ├── metadata.json          # Episode info, timestamps, sync offsets
    ├── images/                # Multi-view images
    │   ├── View1/
    │   ├── View2/
    │   ├── View3/
    │   ├── View4/
    │   └── View5/
    ├── sensor_data.npz        # Sensor windows (N, 650, 1026)
    ├── robot_states.csv       # Robot states at 100Hz
    └── actions.npz            # Computed actions for training

Usage:
    python Optimized_Data_Collector.py
    python Optimized_Data_Collector.py --output-dir ./my_dataset
"""

import os, time, json, cv2, zmq, numpy as np
import threading, signal, socket, struct, csv
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import argparse

# ==============================
# Configuration
# ==============================
DEFAULT_OUTPUT_DIR = "./training_data"
DEFAULT_EPISODE_NAME_PREFIX = "episode"

# Camera settings
ZED_SERIAL_TO_VIEW = {"41182735": "View1", "49429257": "View2", "44377151": "View3", "49045152": "View4"}
OAK_KEYWORD = "OAK"
REQUIRED_VIEWS = set(ZED_SERIAL_TO_VIEW.values()) | {"View5"}

# Network settings
ZMQ_CAM_PULL_PORT = 5555
ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111"
ZMQ_ROBOT_PUB_PORT = 5556
ZMQ_ROBOT_TOPIC = b"robot_state"

SENSOR_UDP_PORT = 9999
SENSOR_UDP_IP = "0.0.0.0"
SENSOR_BUFFER_SIZE = 4 * 1024 * 1024

# Data formats
ROBOT_PAYLOAD_FORMAT = '<ddf12f'  # ts, send_ts, force, 6x joints, 6x pose
ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT)

SENSOR_NXZRt = 1025
SENSOR_PACKET_HEADER_FORMAT = '<ddf'  # ts, send_ts, force
SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)
SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)
SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE

# Model-specific settings
SENSOR_WINDOW_SIZE = 650  # 1 second @ 650Hz
SENSOR_INPUT_CHANNELS = 1026  # 1 force + 1025 A-scan
ACTION_HORIZON = 8
ACTION_DIM = 7

# Calibration and timing
SENSOR_CALIBRATION_COUNT = 50
STATUS_PERIOD = 1.0
STALL_SEC = 5.0

# Global flags
START_SAVE_FLAG = threading.Event()
ROBOT_RECEIVED_FIRST = False
SENSOR_RECEIVED_FIRST = False
CAM_RECEIVED_ALL_VIEWS = False
CAM_RECEIVED_VIEWS = set()

stop_event = threading.Event()


# ==============================
# Async Image Writer
# ==============================
class AsyncImageWriter(threading.Thread):
    def __init__(self, max_queue=5000):
        super().__init__(daemon=True)
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()
        self.saved_count = 0

    def submit(self, path, img):
        if not self.stop_flag.is_set():
            try:
                self.q.put_nowait((path, img))
            except:
                print(f"[Writer] Queue full, dropping frame: {path}")

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
                try:
                    cv2.imwrite(path, img)
                    self.saved_count += 1
                except Exception as e:
                    print(f"[Writer] Error saving {path}: {e}")
                finally:
                    self.q.task_done()
            except Empty:
                if self.stop_flag.is_set() and self.q.empty():
                    break
                continue

    def stop(self):
        print(f"⏳ Flushing remaining {self.q.qsize()} images...")
        self.stop_flag.set()
        self.q.join()
        print(f"✅ Writer stopped. Saved {self.saved_count} images")


# ==============================
# Episode Data Manager
# ==============================
class EpisodeDataManager:
    """Manages data collection for a single episode"""

    def __init__(self, base_dir, episode_name=None):
        self.base_dir = Path(base_dir)

        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"{DEFAULT_EPISODE_NAME_PREFIX}_{timestamp}"

        self.episode_dir = self.base_dir / episode_name
        self.images_dir = self.episode_dir / "images"

        # Create directories
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

        for view in REQUIRED_VIEWS:
            (self.images_dir / view).mkdir(exist_ok=True)

        # Data buffers
        self.camera_frames = []  # [{view1: (img, ts), view2: ..., timestamp: ...}]
        self.sensor_windows = []  # [(timestamp, window_data)]
        self.robot_states = []    # [(ts, recv_ts, origin_ts, joints, pose)]

        # Metadata
        self.metadata = {
            "episode_name": episode_name,
            "start_time": None,
            "end_time": None,
            "camera_views": list(REQUIRED_VIEWS),
            "sensor_hz": 650,
            "robot_hz": 100,
            "sensor_window_size": SENSOR_WINDOW_SIZE,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
        }

        # Statistics
        self.stats = {
            "camera_frames_saved": 0,
            "sensor_windows_saved": 0,
            "robot_states_saved": 0,
        }

        print(f"📁 Episode directory: {self.episode_dir}")

    def add_camera_frame(self, view_name, image, timestamp):
        """Add camera frame (will be grouped by timestamp)"""
        # For now, just save directly
        filename = f"{timestamp:.6f}.jpg"
        path = self.images_dir / view_name / filename
        return path

    def add_sensor_window(self, timestamp, force_data, aline_data):
        """Add sensor window (650 samples)"""
        # Combine force (650,) and aline (650, 1025) → (650, 1026)
        window = np.concatenate([
            force_data.reshape(-1, 1),  # (650, 1)
            aline_data                   # (650, 1025)
        ], axis=1)

        self.sensor_windows.append({
            'timestamp': timestamp,
            'data': window
        })

    def add_robot_state(self, recv_time, origin_ts, send_ts, force, joints, pose):
        """Add robot state"""
        self.robot_states.append([
            recv_time, origin_ts, send_ts, force,
            *joints, *pose
        ])

    def save(self):
        """Save all collected data"""
        print(f"\n{'='*80}")
        print(f"💾 Saving Episode Data")
        print(f"{'='*80}")

        # Update metadata
        self.metadata["end_time"] = time.time()
        self.metadata["camera_frames_count"] = len(self.camera_frames)
        self.metadata["sensor_windows_count"] = len(self.sensor_windows)
        self.metadata["robot_states_count"] = len(self.robot_states)

        # Save metadata
        metadata_path = self.episode_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")

        # Save sensor data
        if self.sensor_windows:
            sensor_path = self.episode_dir / "sensor_data.npz"
            timestamps = np.array([w['timestamp'] for w in self.sensor_windows])
            data = np.stack([w['data'] for w in self.sensor_windows])  # (N, 650, 1026)

            np.savez(sensor_path,
                    timestamps=timestamps,
                    data=data)
            print(f"✅ Saved sensor data: {sensor_path} | Shape: {data.shape}")

        # Save robot states
        if self.robot_states:
            robot_path = self.episode_dir / "robot_states.csv"
            with open(robot_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "recv_timestamp", "origin_timestamp", "send_timestamp", "force",
                    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                    "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"
                ])
                writer.writerows(self.robot_states)
            print(f"✅ Saved robot states: {robot_path} | Count: {len(self.robot_states)}")

        print(f"{'='*80}")
        print(f"✅ Episode saved successfully!")
        print(f"{'='*80}\n")


# ==============================
# Sensor Data Buffer Manager
# ==============================
class SensorWindowManager:
    """Manages sensor data and creates 650-sample windows"""

    def __init__(self, window_size=650):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size * 2)  # Keep 2x window for safety
        self.lock = threading.Lock()

    def add_samples(self, samples):
        """Add sensor samples (list of dicts with timestamp, force, aline)"""
        with self.lock:
            self.buffer.extend(samples)

    def get_latest_window(self):
        """Get latest 650 samples as a window"""
        with self.lock:
            if len(self.buffer) < self.window_size:
                return None

            # Get last 650 samples
            window_samples = list(self.buffer)[-self.window_size:]

            # Extract data
            forces = np.array([s['force'] for s in window_samples], dtype=np.float32)
            alines = np.array([s['aline'] for s in window_samples], dtype=np.float32)
            timestamp = window_samples[-1]['timestamp']  # Use last sample timestamp

            return timestamp, forces, alines


# ==============================
# UDP Sensor Receiver
# ==============================
def sensor_udp_receiver_thread(episode_manager, window_manager):
    global sensor_clock_offset_s, SENSOR_RECEIVED_FIRST

    sensor_clock_offset_s = None
    sensor_calibration_samples = []

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SENSOR_BUFFER_SIZE)
        sock.bind((SENSOR_UDP_IP, SENSOR_UDP_PORT))
        sock.settimeout(1.0)
        print(f"✅ Sensor UDP Receiver started on port {SENSOR_UDP_PORT}")
    except Exception as e:
        print(f"[ERROR] Failed to bind UDP socket: {e}")
        stop_event.set()
        return

    print(f"⏳ Calibrating sensor clock offset (first {SENSOR_CALIBRATION_COUNT} batches)...")

    last_save_time = time.time()
    SAVE_INTERVAL = 0.2  # Save sensor window every 200ms (5Hz to match camera)

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(SENSOR_BUFFER_SIZE)
        except socket.timeout:
            continue
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"[Sensor UDP] Error: {e}")
            continue

        recv_time = time.time()

        if len(data) < SENSOR_TOTAL_PACKET_SIZE:
            continue

        try:
            # Parse datagram
            num_packets = struct.unpack('<I', data[:4])[0]
            expected_size = 4 + (num_packets * SENSOR_TOTAL_PACKET_SIZE)

            if len(data) != expected_size or num_packets == 0:
                continue

            # Parse packets
            records = []
            mv = memoryview(data)[4:]
            offset = 0
            last_send_ts = 0.0

            for _ in range(num_packets):
                header = mv[offset:offset + SENSOR_PACKET_HEADER_SIZE]
                ts, send_ts, force = struct.unpack(SENSOR_PACKET_HEADER_FORMAT, header)
                offset += SENSOR_PACKET_HEADER_SIZE

                aline_bytes = mv[offset:offset + SENSOR_ALINE_SIZE]
                aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
                offset += SENSOR_ALINE_SIZE

                records.append({
                    "timestamp": ts,
                    "send_timestamp": send_ts,
                    "force": float(force),
                    "aline": aline
                })
                last_send_ts = send_ts

        except Exception as e:
            print(f"[Sensor UDP] Parse error: {e}")
            continue

        # Clock offset calibration
        if num_packets > 0:
            net_plus_offset = recv_time - last_send_ts

            if sensor_clock_offset_s is None:
                sensor_calibration_samples.append(net_plus_offset)

                if len(sensor_calibration_samples) >= SENSOR_CALIBRATION_COUNT:
                    sensor_clock_offset_s = np.mean(sensor_calibration_samples)
                    print(f"\n{'='*80}")
                    print(f"✅ Sensor Clock Offset Calibrated: {sensor_clock_offset_s*1000:.1f} ms")
                    print(f"{'='*80}\n")

                    if not SENSOR_RECEIVED_FIRST:
                        SENSOR_RECEIVED_FIRST = True
                        print("🔬 Sensor: Calibration complete. Checking readiness...")
                        check_all_ready()
                else:
                    print(f"⏳ Sensor Calibrating... ({len(sensor_calibration_samples)}/{SENSOR_CALIBRATION_COUNT})", end='\r')

        # Add to buffer
        window_manager.add_samples(records)

        # Save window periodically (if saving is enabled)
        if START_SAVE_FLAG.is_set() and (time.time() - last_save_time) >= SAVE_INTERVAL:
            window_data = window_manager.get_latest_window()
            if window_data:
                timestamp, forces, alines = window_data
                episode_manager.add_sensor_window(timestamp, forces, alines)
                last_save_time = time.time()

    sock.close()
    print("🛑 Sensor UDP Receiver stopped")


# ==============================
# Utility Functions
# ==============================
def get_view_name_from_cam(cam_name: str) -> str:
    """Extract view name from camera name"""
    cam_lower = cam_name.lower()
    for serial, view_name in ZED_SERIAL_TO_VIEW.items():
        if serial in cam_lower:
            return view_name
    if OAK_KEYWORD.lower() in cam_lower:
        return "View5"
    return "Unknown"


def check_all_ready():
    """Check if all data sources are ready"""
    global ROBOT_RECEIVED_FIRST, SENSOR_RECEIVED_FIRST, CAM_RECEIVED_ALL_VIEWS

    if ROBOT_RECEIVED_FIRST and SENSOR_RECEIVED_FIRST and CAM_RECEIVED_ALL_VIEWS:
        if not START_SAVE_FLAG.is_set():
            START_SAVE_FLAG.set()
            print(f"\n{'#'*80}")
            print(f"🚀 ALL SYSTEMS READY! STARTING DATA COLLECTION!")
            print(f"{'#'*80}\n")


def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C detected — Stopping data collection...")
    stop_event.set()


def parse_args():
    parser = argparse.ArgumentParser(description='Optimized Data Collector for VLA Training')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--episode-name', type=str, default=None,
                       help='Episode name (default: auto-generated with timestamp)')
    return parser.parse_args()


# ==============================
# Main
# ==============================
def main():
    global ROBOT_RECEIVED_FIRST, CAM_RECEIVED_ALL_VIEWS, CAM_RECEIVED_VIEWS

    args = parse_args()
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\n{'='*80}")
    print(f"📊 Optimized Data Collector for VLA Training")
    print(f"{'='*80}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Episode Name: {args.episode_name or 'auto-generated'}")
    print(f"{'='*80}\n")

    # Initialize
    episode_manager = EpisodeDataManager(args.output_dir, args.episode_name)
    episode_manager.metadata["start_time"] = time.time()

    window_manager = SensorWindowManager(window_size=SENSOR_WINDOW_SIZE)
    writer = AsyncImageWriter()
    writer.start()

    # Setup ZMQ
    ctx = zmq.Context.instance()

    cam_sock = ctx.socket(zmq.PULL)
    cam_sock.setsockopt(zmq.RCVHWM, 5000)
    cam_sock.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
    cam_sock.bind(f"tcp://0.0.0.0:{ZMQ_CAM_PULL_PORT}")
    print(f"✅ Camera PULL listening on port {ZMQ_CAM_PULL_PORT}")

    robot_sock = ctx.socket(zmq.SUB)
    robot_sock.setsockopt(zmq.RCVHWM, 100)
    robot_sock.connect(f"tcp://{ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}")
    robot_sock.subscribe(ZMQ_ROBOT_TOPIC)
    print(f"✅ Robot SUB connected to {ZMQ_ROBOT_PUB_ADDRESS}:{ZMQ_ROBOT_PUB_PORT}")

    poller = zmq.Poller()
    poller.register(cam_sock, zmq.POLLIN)
    poller.register(robot_sock, zmq.POLLIN)

    # Start sensor thread
    sensor_thread = threading.Thread(
        target=sensor_udp_receiver_thread,
        args=(episode_manager, window_manager),
        daemon=True
    )
    sensor_thread.start()

    # Statistics
    cam_cnt = defaultdict(int)
    robot_cnt = 0
    last_status_print = time.time()

    print(f"\n{'='*80}")
    print(f"🎬 Data Collection Started")
    print(f"{'='*80}")
    print(f"Waiting for all data sources (Camera, Robot, Sensor)...")
    print(f"Press Ctrl+C to stop\n")

    try:
        while not stop_event.is_set():
            socks = dict(poller.poll(timeout=100))
            now = time.time()

            # Camera data
            if cam_sock in socks:
                while True:
                    try:
                        parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) < 2:
                            continue

                        meta = json.loads(parts[0].decode("utf-8"))
                        jpg = parts[1]

                        cam_name = meta.get("camera", "unknown")
                        timestamp = float(meta.get("timestamp", 0.0))

                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        # Check if all views received
                        if not CAM_RECEIVED_ALL_VIEWS:
                            view_name = get_view_name_from_cam(cam_name)
                            if view_name and view_name != "Unknown":
                                CAM_RECEIVED_VIEWS.add(view_name)

                                if len(CAM_RECEIVED_VIEWS) == len(REQUIRED_VIEWS):
                                    CAM_RECEIVED_ALL_VIEWS = True
                                    print(f"✨ Cameras: All {len(REQUIRED_VIEWS)} views received. Checking readiness...")
                                    check_all_ready()

                        # Save if ready
                        if START_SAVE_FLAG.is_set():
                            view_name = get_view_name_from_cam(cam_name)
                            if view_name and view_name != "Unknown":
                                save_path = episode_manager.add_camera_frame(view_name, img, timestamp)
                                writer.submit(str(save_path), img)
                                cam_cnt[view_name] += 1

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[Camera] Error: {e}")
                        break

            # Robot data
            if robot_sock in socks:
                while True:
                    try:
                        parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) != 2:
                            continue

                        topic, payload = parts[0], parts[1]
                        if topic != ZMQ_ROBOT_TOPIC or len(payload) != ROBOT_PAYLOAD_SIZE:
                            continue

                        data = struct.unpack(ROBOT_PAYLOAD_FORMAT, payload)
                        origin_ts, send_ts, force = data[0:3]
                        joints, pose = data[3:9], data[9:15]

                        # First robot data received
                        if not ROBOT_RECEIVED_FIRST:
                            ROBOT_RECEIVED_FIRST = True
                            print("🤖 Robot: First data received. Checking readiness...")
                            check_all_ready()

                        # Save if ready
                        if START_SAVE_FLAG.is_set():
                            episode_manager.add_robot_state(now, origin_ts, send_ts, force, joints, pose)
                            robot_cnt += 1

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[Robot] Error: {e}")
                        break

            # Status update
            if now - last_status_print >= STATUS_PERIOD:
                if START_SAVE_FLAG.is_set():
                    cam_status = " | ".join([f"{k}:{v}" for k, v in sorted(cam_cnt.items())])
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Cameras: {cam_status} | Robot: {robot_cnt} | "
                          f"Sensor Windows: {len(episode_manager.sensor_windows)}")
                else:
                    ready_status = []
                    if ROBOT_RECEIVED_FIRST:
                        ready_status.append("Robot✅")
                    if SENSOR_RECEIVED_FIRST:
                        ready_status.append("Sensor✅")
                    if CAM_RECEIVED_ALL_VIEWS:
                        ready_status.append("Camera✅")
                    print(f"[WAIT] {' | '.join(ready_status)} | Views: {len(CAM_RECEIVED_VIEWS)}/{len(REQUIRED_VIEWS)}")

                last_status_print = now

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        stop_event.set()

    finally:
        print(f"\n{'='*80}")
        print(f"🧹 Cleanup and Saving")
        print(f"{'='*80}")

        # Stop sensor thread
        print("⏳ Stopping sensor thread...")
        stop_event.set()
        sensor_thread.join(timeout=5.0)

        # Stop writer
        writer.stop()
        writer.join()

        # Save episode data
        episode_manager.save()

        # Cleanup ZMQ
        print("⏳ Cleaning up ZMQ...")
        poller.unregister(cam_sock)
        poller.unregister(robot_sock)
        cam_sock.close()
        robot_sock.close()
        ctx.term()

        print(f"\n{'='*80}")
        print(f"✅ Data Collection Complete!")
        print(f"{'='*80}")
        print(f"Episode: {episode_manager.episode_dir}")
        print(f"Camera Frames: {writer.saved_count}")
        print(f"Sensor Windows: {len(episode_manager.sensor_windows)}")
        print(f"Robot States: {len(episode_manager.robot_states)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
