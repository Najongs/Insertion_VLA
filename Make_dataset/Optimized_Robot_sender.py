#!/usr/bin/env python3
"""
Optimized Robot Sender for Async VLA Inference

Optimizations:
- Send rate: 100Hz → 10Hz (matches Action Expert frequency)
- Synchronized with Action Expert inference loop
- Lower network bandwidth usage
- Real-time compatible with 100ms action period

Data Format (ZMQ PUB):
- Topic: b"robot_state"
- Payload: <ddf12f (68 bytes)
  - origin_timestamp: double (8 bytes)
  - send_timestamp: double (8 bytes)
  - force: float (4 bytes) - placeholder
  - joints[6]: float (24 bytes)
  - pose[6]: float (24 bytes)

Usage:
    # With robot control
    python Optimized_Robot_sender.py --robot on

    # Simulation mode (no robot)
    python Optimized_Robot_sender.py --robot off
"""

import os
import sys
import time
import csv
import random
import argparse
import logging
import pathlib
import threading
import struct
import zmq

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

# =========================
# Configuration
# =========================
# 저장 폴더
OUTPUT_DIR = "./dataset/Robot_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ZeroMQ Publisher Configuration
ZMQ_PUB_ADDRESS = "*"  # Bind to all interfaces
ZMQ_PUB_PORT = 5556
SENDER_RATE_HZ = 10  # 🔥 OPTIMIZED: 100Hz → 10Hz (matches Action Expert)
ZMQ_TOPIC = b"robot_state"

# 데이터 포맷 (unchanged)
PAYLOAD_FORMAT = '<ddf12f'
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)

print(f"📊 Optimized Robot Sender Configuration:")
print(f"   Send Rate: {SENDER_RATE_HZ} Hz ({1000/SENDER_RATE_HZ:.0f}ms interval)")
print(f"   Payload Size: {PAYLOAD_SIZE} bytes")
print(f"   ZMQ Port: {ZMQ_PUB_PORT}")


# ============================================================
# Global Clock
# ============================================================
class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.timestamp = round(time.time(), 3)
        self.running = True
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.timestamp

    def run(self):
        while self.running:
            with self.lock:
                self.timestamp = round(time.time(), 3)
            time.sleep(0.005)  # Update at ~200Hz

    def stop(self):
        self.running = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default=None, help="출력 폴더 접미사")
    p.add_argument("--robot", choices=["on", "off"], default="on",
                   help="로봇 제어 활성화 여부 (기본값: on)")
    return p.parse_args()


# ============================================================
# Real-time Robot Data Sampler
# ============================================================
class RtSampler(threading.Thread):
    """
    Real-time robot data sampler
    Samples at 10Hz (synchronized with Action Expert)
    """
    def __init__(self, robot, out_csv, clock, rate_hz=10):
        super().__init__(daemon=True)
        self.robot = robot
        self.out_csv = out_csv
        self.dt = 1.0 / float(rate_hz)
        self.clock = clock
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = None
        self.latest_p = None
        self.sample_count = 0

    def stop(self):
        self.stop_evt.set()

    def get_latest_data(self):
        with self.lock:
            return self.latest_q, self.latest_p

    def run(self):
        print(f"✅ Starting robot data sampling to {self.out_csv} at {1/self.dt:.1f} Hz...")
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "joint_angle_1", "joint_angle_2", "joint_angle_3",
                "joint_angle_4", "joint_angle_5", "joint_angle_6",
                "EE_x", "EE_y", "EE_z", "EE_a", "EE_b", "EE_r"
            ])

            next_t = time.time()
            last_status_time = time.time()

            while not self.stop_evt.is_set():
                q, p = None, None
                ts_now = self.clock.now()

                # Get joint angles
                for name in ("GetJoints", "GetJointPos", "GetJointAngles"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            q = list(fn())
                            break
                        except Exception:
                            pass

                # Get end-effector pose
                for name in ("GetPose", "GetPoseXYZABC", "GetCartesianPose"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            p = list(fn())
                            break
                        except Exception:
                            pass

                if q is not None and len(q) >= 6 and p is not None and len(p) >= 6:
                    w.writerow([f"{ts_now:.6f}"] + q[:6] + p[:6])
                    with self.lock:
                        self.latest_q = q[:6]
                        self.latest_p = p[:6]
                    self.sample_count += 1

                # Status print every 5 seconds
                if time.time() - last_status_time >= 5.0:
                    print(f"📊 Robot sampling: {self.sample_count} samples collected "
                          f"({self.sample_count/(time.time()-last_status_time+1e-9):.1f} Hz avg)")
                    last_status_time = time.time()

                # Precise timing
                next_t += self.dt
                sleep_dt = next_t - time.time()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    # Warn if falling behind (but less frequently)
                    if random.random() < 0.01:
                        print(f"[RtSampler WARN] Loop falling behind by {-sleep_dt*1000:.1f} ms")

        print(f"✅ Robot data sampling stopped. Total samples: {self.sample_count}")


# ============================================================
# ZMQ Publisher (10Hz)
# ============================================================
class ZmqPublisher(threading.Thread):
    """
    ZMQ Publisher for robot state
    Sends at 10Hz (synchronized with Action Expert)
    """
    def __init__(self, sampler, clock, address, port, stop_event, rate_hz=10):
        super().__init__(daemon=True)
        self.sampler = sampler
        self.clock = clock
        self.address = address
        self.port = port
        self.stop_event = stop_event
        self.dt = 1.0 / float(rate_hz)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        bind_addr = f"tcp://{self.address}:{self.port}"
        self.socket.set_hwm(10)
        self.socket.setsockopt(zmq.LINGER, 500)
        self.socket.bind(bind_addr)
        self.send_count = 0

        print(f"✅ ZMQ Publisher bound to {bind_addr} at {rate_hz} Hz.")
        print(f"   Topic: '{ZMQ_TOPIC.decode()}', Payload Size: {PAYLOAD_SIZE} bytes")

    def run(self):
        next_send_time = time.time()
        last_status_time = time.time()

        # Create poller for interruptible sleep
        poller = zmq.Poller()
        dummy_sock = self.context.socket(zmq.PULL)
        dummy_addr = "inproc://dummy_poll"
        try:
            dummy_sock.bind(dummy_addr)
            poller.register(dummy_sock, zmq.POLLIN)
        except zmq.ZMQError as e:
            print(f"[ZmqPublisher ERR] Failed to bind dummy socket: {e}")
            dummy_sock = None

        try:
            while not self.stop_event.is_set():
                q, p = self.sampler.get_latest_data()

                if q is not None and p is not None:
                    ts = self.clock.now()
                    force = 0.0  # Placeholder
                    send_ts = time.time()

                    try:
                        payload_bytes = struct.pack(PAYLOAD_FORMAT, ts, send_ts, force, *q, *p)

                        if len(payload_bytes) != PAYLOAD_SIZE:
                            print(f"[ZmqPublisher ERR] Payload size mismatch!")
                            continue

                        self.socket.send_multipart([ZMQ_TOPIC, payload_bytes], zmq.DONTWAIT)
                        self.send_count += 1

                    except zmq.Again:
                        pass
                    except zmq.ZMQError as e:
                        print(f"[ZmqPublisher ERR] Failed to send ZMQ message: {e}")
                        if e.errno == zmq.ETERM:
                            break
                    except Exception as e:
                        print(f"[ZmqPublisher ERR] Unexpected error during send: {e}")

                # Status print every 5 seconds
                if time.time() - last_status_time >= 5.0:
                    actual_rate = self.send_count / (time.time() - last_status_time + 1e-9)
                    print(f"📡 ZMQ sending: {self.send_count} messages "
                          f"({actual_rate:.1f} Hz avg, target: {1/self.dt:.1f} Hz)")
                    last_status_time = time.time()
                    self.send_count = 0

                # Precise sleep to maintain rate
                next_send_time += self.dt
                sleep_duration = next_send_time - time.time()

                if sleep_duration > 0 and dummy_sock:
                    try:
                        events = poller.poll(int(sleep_duration * 1000))
                        if self.stop_event.is_set():
                            break
                    except zmq.ZMQError as e:
                        if e.errno == zmq.ETERM:
                            break
                        print(f"[ZmqPublisher WARN] Poller error: {e}")
                        time.sleep(max(0, sleep_duration))
                    except Exception as e:
                        print(f"[ZmqPublisher WARN] Unexpected error during poll: {e}")
                        time.sleep(max(0, sleep_duration))
                elif sleep_duration > 0:
                    time.sleep(sleep_duration)

        finally:
            # Cleanup
            if dummy_sock:
                try:
                    poller.unregister(dummy_sock)
                    dummy_sock.close()
                except:
                    pass

            self.socket.close()
            self.context.term()
            print(f"🛑 ZMQ Publisher stopped")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    # Setup output directory
    if args.run_tag:
        session_name = f"robot_data_{args.run_tag}"
    else:
        from datetime import datetime
        session_name = f"robot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    csv_path = os.path.join(OUTPUT_DIR, f"{session_name}.csv")

    print(f"\n{'='*80}")
    print(f"🤖 Optimized Robot Sender")
    print(f"{'='*80}")
    print(f"Mode: {'Robot Control ON' if args.robot == 'on' else 'Simulation Mode'}")
    print(f"Send Rate: {SENDER_RATE_HZ} Hz (optimized for Action Expert)")
    print(f"CSV Output: {csv_path}")
    print(f"ZMQ Port: {ZMQ_PUB_PORT}")
    print(f"{'='*80}\n")

    # Initialize robot
    robot = None
    if args.robot == "on":
        print("🔌 Connecting to Mecademic robot...")
        try:
            robot = initializer.initialize_robot_connection()
            robot.ActivateRobot()
            robot.Home()
            robot.WaitHomed()
            print("✅ Robot initialized and homed")
        except Exception as e:
            print(f"❌ Failed to initialize robot: {e}")
            print("⚠️  Continuing in simulation mode...")
            robot = None

    # Create dummy robot if needed
    if robot is None:
        print("🤖 Creating dummy robot for simulation...")

        class DummyRobot:
            def __init__(self):
                self.t0 = time.time()

            def GetJoints(self):
                # Simulate slow sinusoidal motion
                t = time.time() - self.t0
                return [
                    10.0 + 5.0 * np.sin(0.1 * t),
                    20.0 + 5.0 * np.sin(0.15 * t),
                    30.0 + 5.0 * np.sin(0.12 * t),
                    0.0,
                    45.0,
                    0.0
                ]

            def GetPose(self):
                t = time.time() - self.t0
                return [
                    150.0 + 10.0 * np.sin(0.1 * t),
                    0.0,
                    200.0 + 10.0 * np.sin(0.12 * t),
                    0.0,
                    90.0,
                    0.0
                ]

        import numpy as np
        robot = DummyRobot()

    # Start global clock
    clock = GlobalClock()
    clock.start()

    # Start robot data sampler
    sampler = RtSampler(robot, csv_path, clock, rate_hz=SENDER_RATE_HZ)
    sampler.start()

    # Wait for sampler to collect some data
    time.sleep(0.5)

    # Start ZMQ publisher
    stop_event = threading.Event()
    publisher = ZmqPublisher(sampler, clock, ZMQ_PUB_ADDRESS, ZMQ_PUB_PORT,
                            stop_event, rate_hz=SENDER_RATE_HZ)
    publisher.start()

    print(f"\n{'='*80}")
    print(f"✅ Optimized Robot Sender Started")
    print(f"{'='*80}")
    print(f"Sampling & Sending at {SENDER_RATE_HZ} Hz")
    print(f"Press Ctrl+C to stop\n")

    # Wait for Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

    # Stop threads
    stop_event.set()
    sampler.stop()
    clock.stop()

    # Wait for threads to finish
    publisher.join(timeout=2.0)
    sampler.join(timeout=2.0)
    clock.join(timeout=1.0)

    print("\n✅ Robot Sender stopped successfully")
    print(f"📁 Data saved to: {csv_path}")


if __name__ == "__main__":
    main()
