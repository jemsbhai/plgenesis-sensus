# 🧠 SENSUS — Contactless Multi-Modal Health Sensing Platform

> **HackUSF 2026** — Turning invisible WiFi signals into real-time health insights.
> No cameras. No wearables. No contact. Just physics.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![ESP32](https://img.shields.io/badge/ESP32--C6-CSI%20Mesh-green.svg)](https://www.espressif.com/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%205-Edge%20AI-red.svg)](https://raspberrypi.com)
[![Gemini](https://img.shields.io/badge/Gemini%20API-Health%20AI-4285F4.svg)](https://aistudio.google.com)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Voice%20Alerts-6366F1.svg)](https://elevenlabs.io)
[![MongoDB](https://img.shields.io/badge/MongoDB%20Atlas-Patient%20Data-10B981.svg)](https://mongodb.com)
[![Snowflake](https://img.shields.io/badge/Snowflake-Analytics-06B6D4.svg)](https://snowflake.com)
[![Auth0](https://img.shields.io/badge/Auth0-Secure%20Access-F59E0B.svg)](https://auth0.com)

---

## 💡 What is Sensus?

**Sensus** is a contactless health monitoring platform that extracts vital signs — heart rate, breathing rate, HRV, stress levels, and more — from ordinary **WiFi signals**, without touching the patient.

Inspired by [WiFi DensePose research](https://arxiv.org/abs/2301.00250) from Carnegie Mellon University and the [RuView](https://github.com/ruvnet/RuView) project, Sensus analyzes **Channel State Information (CSI)** disturbances caused by human chest movement to reconstruct breathing and cardiac patterns in real time.

### How It Works

```
WiFi Router → 2.4GHz radio waves fill the room
    ↓
Human body → chest expansion/contraction disturbs signal propagation
    ↓
ESP32-C6 Mesh (3 nodes) → captures 52 CSI subcarrier amplitudes + phases at 100 Hz
    ↓
MQTT → streams raw CSI to Raspberry Pi 5 over local network
    ↓
Signal Processing Pipeline (RuView-inspired):
  → Conjugate Multiplication (phase cleaning)
  → Hampel Filter (outlier removal)
  → Top-K Subcarrier Selection (motion-sensitive channels)
  → PCA Dimensionality Reduction
  → Bandpass Filtering + FFT (vital sign extraction)
  → Fresnel Zone Breathing Model (physics-based)
  → Multi-Node SNR-Weighted Fusion
    ↓
Health Engine → fuses CSI + environmental + audio + BLE data
    ↓
Gemini API → AI-powered clinical interpretation
    ↓
ElevenLabs → multilingual voice alerts for critical states
    ↓
Dashboard → real-time visualization at Pi:5000
```

---

## 🏥 Healthcare Use Case: Frontline Worker Monitoring

Sensus is designed for **frontline healthcare workers** — nurses, ER doctors, paramedics — who can't wear monitoring devices during shifts.

**The Problem:** Healthcare worker burnout kills. Nurses work 12-hour shifts with no continuous health monitoring. Stress, fatigue, and deteriorating vital signs go undetected until crisis.

**The Sensus Solution:** Place ESP32 sensor nodes around a break room, nursing station, or triage area. Sensus passively monitors anyone in the space — heart rate, breathing rate, stress via HRV, skin conductance, and environmental conditions — without any wearable or camera. When vital signs indicate dangerous stress or fatigue, Sensus alerts the shift supervisor via Gemini-powered voice alerts in their language.

### Why This Matters for Reach Capital

- **Privacy-first**: No cameras, no video, no identifiable data
- **Zero-burden**: Healthcare workers don't need to do anything — just be in the room
- **Scalable**: $54 hardware cost per monitored room (3x ESP32-C6 at ~$8 each + router)
- **Multi-language**: ElevenLabs provides alerts in any language (critical for diverse healthcare teams)

---

## 🔧 Hardware Architecture

| Component | Role | Cost |
|-----------|------|------|
| 3× Seeed XIAO ESP32-C6 | WiFi CSI extraction + BLE scanning + environmental sensors | ~$24 |
| 3× ESP32-S3 Sense | Audio capture via built-in MEMS microphone | ~$30 |
| 1× Raspberry Pi 5 (16GB) | Edge AI processing, fusion engine, dashboard | ~$80 |
| 1× TP-Link AX1500 Router | Dedicated 2.4GHz sensing network (Ch 1, 20MHz) | ~$40 |
| DHT11, CCS811, MAX30102, GSR | Environmental + ground-truth sensors | ~$15 |

**Total: ~$189** for a complete multi-modal health sensing station.

### Network Topology

```
┌─────────────────────────────────────────────────────┐
│                    SENSUS MESH                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ ESP32-C6 │  │ ESP32-C6 │  │ ESP32-C6 │  CSI Mesh │
│  │  node_1  │  │  node_2  │  │  node_3  │  (WiFi)   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│  ┌────┴──────┐ ┌────┴──────┐ ┌────┴──────┐           │
│  │ ESP32-S3  │ │ ESP32-S3  │ │ ESP32-S3  │  Audio    │
│  │ node_s3_1 │ │ node_s3_2 │ │ node_s3_3 │  Mesh     │
│  └────┬──────┘ └────┬──────┘ └────┬──────┘           │
│       │              │              │                 │
│       └──────────────┼──────────────┘                 │
│                      │ MQTT (192.168.0.59:1883)       │
│              ┌───────┴────────┐                       │
│              │  TP-Link AX1500 │  Dedicated CSI Net    │
│              │  sensus-csi     │  (2.4GHz, Ch1, 20MHz) │
│              └───────┬────────┘                       │
│                      │ Ethernet                       │
│              ┌───────┴────────┐                       │
│              │ Raspberry Pi 5  │  Edge AI + Dashboard  │
│              │    (16GB)       │  Port 5000             │
│              └───────┬────────┘                       │
│                      │ WiFi (phone hotspot)            │
│                      ↓                                │
│              Cloud APIs (Gemini, ElevenLabs,           │
│              MongoDB Atlas, Snowflake, Auth0)          │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Signal Processing — RuView-Inspired Pipeline

Our CSI processing pipeline is inspired by [RuView's](https://github.com/ruvnet/RuView) WiFi DensePose architecture:

1. **Conjugate Multiplication** — Removes random phase offsets between consecutive CSI packets by computing `p_clean[t] = p[t] × conj(p[t-1])`. This isolates phase *changes* caused by chest motion from static environmental effects.

2. **Hampel Filter** — Replaces outlier samples with local median values. More robust than moving average for handling WiFi interference spikes from other devices.

3. **Top-K Subcarrier Selection** — Scores all 52 subcarriers by cardiac+breathing band energy ratio and selects the top 10 most body-motion-sensitive channels. Inspired by RuView's learned graph partitioning.

4. **PCA Dimensionality Reduction** — Extracts the principal component (dominant motion signature) from selected subcarriers via eigendecomposition.

5. **Bandpass + FFT Extraction** — Separates breathing (0.1–0.5 Hz) and cardiac (0.8–2.0 Hz) frequency bands, identifies dominant peak frequency.

6. **Fresnel Zone Breathing Model** — Physics-based validation: at 2.4 GHz (λ=12.5cm), chest displacement of 1–5mm from breathing crosses Fresnel zone boundaries, creating predictable amplitude modulations.

7. **Multi-Node SNR-Weighted Fusion** — Combines vital signs from 3+ mesh nodes using signal-to-noise ratio as weights. Higher-quality nodes contribute more to the final estimate.

---

## 🤖 AI Integration

### Google Gemini (Prize: Best Use of Gemini API)
- Real-time clinical interpretation of fused vital signs
- Context-aware: considers environmental data (CO₂, temperature, humidity) when interpreting elevated breathing or heart rates
- Runs on `gemini-2.0-flash` for low-latency responses
- Toggle-controlled from dashboard to manage rate limits

### ElevenLabs (Prize: Best Use of ElevenLabs)
- Multilingual voice alerts using `eleven_multilingual_v2` model
- Speaks critical health alerts through Pi speaker
- Supports 29+ languages for diverse healthcare teams
- Only activates on warning/critical states to avoid alert fatigue

### MongoDB Atlas (Prize: Best Use of MongoDB Atlas)
- Stores all vital sign readings with timestamps
- Session management for patient monitoring periods
- Queryable history for trend analysis

### Snowflake (Prize: Best Use of Snowflake API)
- Analytics data warehouse for population health metrics
- Streams vital signs every 30 seconds for longitudinal analysis
- Enables cross-patient, cross-shift aggregate insights

### Auth0 (Prize: Best Use of Auth0)
- Secure clinician authentication for dashboard access
- Role-based access control (nurse vs. supervisor vs. admin)
- HIPAA-aligned session management

---

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 with Raspberry Pi OS
- 3× ESP32-C6 boards + WiFi router
- Python 3.11+

### Pi Setup
```bash
# Clone and bootstrap
git clone https://github.com/YOUR_USERNAME/sensus.git
cd sensus
chmod +x pi/scripts/bootstrap.sh
./pi/scripts/bootstrap.sh

# Configure API keys
cp pi/config/.env.template pi/config/.env
nano pi/config/.env  # Add your Gemini, ElevenLabs keys

# Start Sensus
cd pi/services
python main.py
```

### ESP32 Setup
1. Open `esp32/c6/sensus_node_c6.ino` in Arduino IDE
2. Board: **XIAO_ESP32C6** | USB CDC On Boot: **Enabled**
3. Change `NODE_ID` per board (`node_1`, `node_2`, `node_3`)
4. Flash each board

### Access Dashboard
Open `http://<pi-ip>:5000` in your browser.

---

## 📁 Project Structure

```
sensus/
├── README.md
├── docs/
│   └── WIRING.md              # Pin-by-pin wiring guide
├── esp32/
│   ├── c6/
│   │   └── sensus_node_c6.ino # CSI + BLE + sensors firmware
│   └── s3/
│       └── sensus_node_s3_audio.ino  # Audio capture firmware
└── pi/
    ├── config/
    │   ├── .env               # API keys (gitignored)
    │   └── .env.template      # Template
    ├── scripts/
    │   └── bootstrap.sh       # One-command Pi setup
    └── services/
        ├── main.py            # Fusion engine + MQTT handler
        ├── csi_processor.py   # RuView-inspired CSI pipeline
        ├── audio_processor.py # Audio event classification
        ├── env_processor.py   # Environmental context
        ├── health_engine.py   # Multi-modal health fusion
        ├── api_integrations.py # Gemini, ElevenLabs, MongoDB, Snowflake, Auth0
        └── dashboard.py       # Real-time web dashboard
```

---

## 🏆 Prize Targets

| Prize | Integration |
|-------|------------|
| Best Use of Gemini API | Real-time health interpretation engine |
| Best Use of ElevenLabs | Multilingual voice health alerts |
| Best Use of Gen AI | Gemini-powered clinical insight generation |
| Best Use of Reach Capital | Frontline healthcare worker monitoring |
| Best Use of MongoDB Atlas | Patient vital signs data store |
| Best Use of Snowflake API | Population health analytics warehouse |
| Best Use of Auth0 | Secure clinician access control |
| Best Use of DigitalOcean | Cloud deployment infrastructure |

---

## 📚 References

- [DensePose From WiFi — CMU (arXiv:2301.00250)](https://arxiv.org/abs/2301.00250)
- [RuView — WiFi DensePose Implementation](https://github.com/ruvnet/RuView)
- [ESP32 CSI — Espressif](https://github.com/espressif/esp-csi)
- [Fresnel Zone Model for Breathing Detection](https://ieeexplore.ieee.org/document/8067692)

---

## 👥 Team

Built at HackUSF 2026, University of South Florida, Tampa FL.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
