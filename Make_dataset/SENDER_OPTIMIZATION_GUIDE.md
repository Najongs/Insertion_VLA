# Sender Optimization Guide for Data Collection

## ✅ Summary: Minimal Changes Required

Your existing sender implementations are already well-optimized for the model training! Here's what you need to know:

---

## 📷 Camera Sender (Optimized_Camera_sender.py)
### Status: ✅ **No Changes Needed**

Your current camera sender is already perfect:
- **Rate**: 5 Hz (200ms interval) - Perfect for VLM updates @ 2.6Hz
- **Views**: 5 views (ZED left x4 + OAK x1) - ✅ Matches model requirement
- **Resolution**: 1280x720 → Will be resized to 640x360 on receiver
- **Format**: JPEG with quality=75

**Recommendation**: Keep using as-is!

```bash
# Current usage (perfect!)
python Optimized_Camera_sender.py --server-ip 10.130.4.79 --fps 5
```

---

## 🤖 Robot Sender (Robot_sender.py)
### Status: ✅ **No Changes Needed**

Your robot sender is already optimal:
- **Rate**: 100 Hz - Excellent for action prediction
- **Data**: Joints (6) + Pose (6) + Force (1) + Timestamps
- **Protocol**: ZMQ PUB with topic filtering
- **Format**: Binary packed (68 bytes) - Very efficient

**Recommendation**: Keep using as-is!

```bash
# Current usage (perfect!)
python Robot_sender.py --robot on
```

---

## 🔬 Sensor Sender (C++ Implementation)
### Status: ⚠️ **Minor Verification Recommended**

Your sensor sender should work perfectly with the new collector, but please verify:

### ✅ What's Already Good:
- **Rate**: 650 Hz (1.5ms interval)
- **Data**: Force (1) + A-scan (1025) = 1026 channels
- **Protocol**: UDP batching for efficiency
- **Format**: Binary packed

### ⚠️ Things to Verify:

1. **Batch Size Consistency**
   ```cpp
   // In your C++ sender, check:
   // - Are you sending consistent batch sizes?
   // - Recommended: 10-50 packets per batch for smooth 650Hz delivery
   ```

2. **Timing Accuracy**
   ```cpp
   // Verify timestamp accuracy:
   double timestamp = get_current_time();  // Should be high-precision
   double send_timestamp = get_current_time();  // Right before sending
   ```

3. **Buffer Management**
   ```cpp
   // Ensure UDP socket buffer is large enough:
   int buffer_size = 4 * 1024 * 1024;  // 4MB recommended
   setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
   ```

### ✅ If Your Sensor Sender Already Does This:
**No changes needed!** The new collector (`Optimized_Data_Collector.py`) will automatically:
- Collect 650 samples into 1-second windows
- Align with camera timestamps
- Save in the correct format (650, 1026) for the model

### 🔧 Optional Optimization (If you see packet loss):
If you experience UDP packet loss during data collection:

```cpp
// Optional: Add pacing between batches
void send_sensor_batch(std::vector<SensorPacket>& batch) {
    // Pack and send batch
    send_udp_datagram(batch);

    // Optional: Small delay to prevent buffer overflow
    // Only needed if you see packet loss
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
```

---

## 🎯 New Data Collector Usage

### Starting All Components:

#### Terminal 1: Camera Sender
```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset
python Optimized_Camera_sender.py --server-ip 10.130.4.79 --fps 5
```

#### Terminal 2: Robot Sender
```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset
python Robot_sender.py --robot on
```

#### Terminal 3: Sensor Sender (C++)
```bash
# Your existing C++ sensor sender
./your_sensor_sender
```

#### Terminal 4: Data Collector
```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset
python Optimized_Data_Collector.py

# Or with custom output directory:
python Optimized_Data_Collector.py --output-dir /path/to/training_data

# Or with custom episode name:
python Optimized_Data_Collector.py --episode-name "insertion_trial_01"
```

---

## 📊 What the New Collector Does Differently

### Improvements Over Total_reciver.py:

1. **Episode-Based Organization**
   ```
   training_data/
   ├── episode_20251030_120000/
   │   ├── metadata.json          # Episode info
   │   ├── images/                # Multi-view images
   │   │   ├── View1/*.jpg
   │   │   ├── View2/*.jpg
   │   │   ├── View3/*.jpg
   │   │   ├── View4/*.jpg
   │   │   └── View5/*.jpg
   │   ├── sensor_data.npz        # (N, 650, 1026) windows
   │   └── robot_states.csv       # 100Hz robot states
   ```

2. **Model-Optimized Data Format**
   - Sensor: Automatically creates 1-second windows (650, 1026)
   - Camera: Synchronized 5-view capture
   - Robot: Ready for action computation

3. **Better Synchronization**
   - Waits for ALL data sources before starting
   - Clock offset calibration for sensor
   - Timestamp alignment

4. **Training-Ready Format**
   - No post-processing needed
   - Direct loading into model
   - Metadata for reproducibility

---

## 🔍 Monitoring During Collection

The collector will display:
```
[12:34:56] Cameras: View1:45 | View2:45 | View3:45 | View4:45 | View5:45 | Robot: 450 | Sensor Windows: 45
```

This shows:
- Camera frames per view
- Robot states collected
- Sensor windows created (each = 1 second of 650Hz data)

---

## 📝 Summary of Required Changes

### Camera Sender: ✅ None
### Robot Sender: ✅ None
### Sensor Sender: ⚠️ Optional verification only

**You're already 95% ready!** Just:
1. Use the new `Optimized_Data_Collector.py`
2. Verify your C++ sensor sender timing (optional)
3. Start collecting training data!

---

## 🆘 Troubleshooting

### If sensor windows are not collecting:
Check C++ sender timestamp precision and UDP batching

### If camera views are missing:
Verify all 5 cameras are running and connected

### If robot data is not arriving:
Check ZMQ_ROBOT_PUB_ADDRESS is correct (currently: 10.130.41.111)

### If you see "WAIT" status for too long:
One of your senders might not be running or not sending data

---

## 📧 Need C++ Sensor Sender Changes?

If you need to modify your C++ sensor sender, the key requirements are:

```cpp
// 1. Packet format (should already match):
struct SensorPacket {
    double timestamp;       // 8 bytes
    double send_timestamp;  // 8 bytes
    float force;            // 4 bytes
    float aline[1025];      // 4100 bytes
    // Total: 4120 bytes
};

// 2. Batch format:
// [4-byte packet count][packet1][packet2]...[packetN]

// 3. Target rate: 650 Hz ± 5%
// 4. UDP destination: Port 9999
```

Let me know if you need help with any C++ modifications!
