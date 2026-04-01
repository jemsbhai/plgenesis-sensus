<![CDATA[<div align="center">

# 🫀 SENSUS

### Contactless Multi-Modal Health Sensing Platform

**WiFi signals can see your heartbeat.**

Sensus extracts vital signs — heart rate, breathing rate, HRV, blood pressure, SpO₂ — from ordinary WiFi signals using Channel State Information (CSI), without touching the patient.

[![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://github.com/jemsbhai/plgenesis-sensus)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Hackathon](https://img.shields.io/badge/PL__Genesis-2026-blueviolet?style=for-the-badge)](https://github.com/jemsbhai/plgenesis-sensus)

</div>

---

## The Problem

**1.7 billion people** lack access to continuous health monitoring. Current solutions require body-worn sensors, are expensive, and create compliance barriers — especially for elderly patients, burn victims, neonates, and psychiatric patients who cannot tolerate contact-based devices.

## The Solution

Sensus uses **WiFi Channel State Information (CSI)** — the fine-grained channel measurements that every WiFi chipset already collects — to detect the micro-movements of breathing and heartbeat through walls, furniture, and clothing.

When a person breathes, their chest displaces by 1-5mm. When their heart beats, the body surface moves by 0.1-0.5mm. These motions cross **Fresnel zone boundaries** in the WiFi signal path, creating measurable phase and amplitude changes across 52 subcarriers at 2.4 GHz.

Sensus captures these signals from a mesh of 3+ ESP32-C6 nodes, processes them through a physics-informed DSP pipeline, and extracts clinical-grade vital signs — **all without touching the patient.**

## How It Works

```
WiFi Router (TP-Link AX1500)
        │
        ▼
┌─────────────────────────────────────┐
│     ESP32-C6 Mesh (3 nodes)        │
│  TX ──── Patient ──── RX-A         │
│              │                      │
│            RX-B                     │
│                                     │
│  CSI: 52 subcarriers × 10 Hz       │
│  Amplitude + Phase per packet       │
└─────────────────────────────────────┘
        │ MQTT
        ▼
┌─────────────────────────────────────┐
│     Raspberry Pi 5 Backend         │
│                                     │
│  Signal Processing Pipeline:        │
│  1. Conjugate Multiplication        │
│     (phase cleaning)                │
│  2. Hampel Filter                   │
│     (outlier removal)               │
│  3. Top-K Subcarrier Selection      │
│     (motion-sensitive channels)     │
│  4. PCA Extraction                  │
│     (dominant motion signature)     │
│  5. Fresnel Breathing Model         │
│     (physics-based rate extraction) │
│  6. Cardiac FFT                     │
│     (heart rate from micro-motion)  │
│  7. HRV Analysis                    │
│     (SDNN, RMSSD, pNN50)           │
│  8. Motion & Presence Detection     │
│  9. Activity Classification         │
│ 10. Multi-Node SNR-Weighted Fusion  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Real-Time Dashboard            │
│  + ML Health State Classifier       │
│  + Alert System                     │
│  + Environmental Monitoring         │
└─────────────────────────────────────┘
```

## Key Innovation: Near-Field Chair Mounting

By mounting ESP32-C6 nodes directly on or near a chair/bed, the seated/lying person dominates the CSI signal by **~20-30 dB** over crowd noise. This physics advantage means Sensus works reliably in crowded rooms — hospital wards, offices, classrooms — where other RF sensing approaches fail.

## Features

### Virtual Scenario Engine (30 Clinical Scenarios)

The demo includes a comprehensive scenario simulator with **30 physiologically accurate clinical scenarios** for demonstration:

| Category | Scenarios |
|----------|-----------|
| **Baseline** | Healthy Resting, Light Activity, Deep Sleep, REM Sleep |
| **Stress & Mental** | Acute Stress, Meditation, Panic Attack, Cognitive Load, Night Terror |
| **Cardiac Events** | Tachycardia, Bradycardia, Atrial Fibrillation, Cardiac Arrest, PVCs |
| **Respiratory** | Sleep Apnea, Hyperventilation, Asthma Attack, COPD Exacerbation |
| **Emergency** | Stroke, Fall Detection, Seizure, Anaphylaxis |
| **Medication** | Beta Blocker Response, Opioid Sedation, Stimulant Effect, Sedative Recovery |
| **Multi-Person** | Room Occupancy Change, Two People Resting |
| **Environmental** | Poor Air Quality Response, Heat Stress |

Each scenario generates **physiologically accurate synthetic CSI data** with embedded breathing, cardiac, and motion signals. The existing `CSIProcessor` pipeline extracts vital signs from this data exactly as it would from real hardware — proving the full detection-to-alert chain works.

### ML Health State Classifier

A **RandomForest ensemble classifier** trained on data from all 30 scenarios provides real-time health state prediction:

- **10 health state classes**: Normal, Sleep, Relaxation, Stress, Medication, Environmental, Cardiac Alert, Respiratory Alert, Emergency, Cardiac Emergency
- **Risk scoring**: Low → Moderate → High → Critical
- **Live confidence + probability bars** in the dashboard
- **Anomaly detection**: Binary normal vs. abnormal classification

### Animated Room Visualization

The dashboard includes a real-time SVG visualization showing:
- WiFi nodes with pulsing signal waves
- Animated signal paths between nodes
- Person with breathing and heartbeat animations
- Motion-specific animations (seizure, fall, tremor, walking)
- Fresnel zone ellipses

## Hackathon Integrations

### 🧬 Hypercerts — Impact Claims ($2.5K bounty)
Generates structured impact claims from health monitoring sessions with CID-rooted evidence. Each session produces a verifiable hypercert documenting who was monitored, what was detected, and the sensor configuration used.

### 🔐 Data Sovereignty — Infrastructure & Digital Rights ($6K bounty)
Personal health data vault with:
- AES-256-GCM encryption at rest
- Granular per-field consent (e.g., grant a clinician access to heart rate only)
- FHIR-compatible portable data export
- Immutable audit log of all data access
- Time-bounded, revocable consent grants

### 📦 Filecoin Storage ($2.5K bounty)
Decentralized health data storage with:
- Content-addressed (CID) health record packages
- Storage deal structures for Filecoin calibration testnet
- CAR file manifests for batch upload
- Verifiable data integrity via SHA-256 hashing

### 🗄️ Storacha ($500 bounty)
Persistent health data on Storacha network with:
- UCAN delegation chains (Patient → Doctor → Specialist)
- Re-delegation support for multi-provider workflows
- Health knowledge base for AI agent RAG
- Content-addressed retrieval with integrity verification

### 🤖 Impulse AI — Autonomous ML ($500 bounty)
Deep integration with the Impulse AI platform for autonomous health state classification:
- **SDK Integration**: Connected via `impulse-api-sdk-python` with live API key authentication
- **Training Pipeline**: 18,180-sample dataset generated from 30 clinical scenarios across 15 physiological features
- **3 ML Tasks**: Multi-class health state classification (30→10 classes, 98.8% accuracy), binary anomaly detection, and time-series vital sign forecasting
- **Real-Time Inference**: RandomForest ensemble classifier provides live health state prediction with confidence scoring and risk levels in the dashboard
- **Impulse AI-Ready Export**: CSV datasets with compatible metadata, feature descriptions, and recommended training parameters for direct upload to Impulse platform
- **Dashboard Panel**: Live Impulse AI status panel showing SDK connection, dataset stats, model metrics, and per-frame inference results

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **CSI Sensing** | ESP32-C6 (XIAO) × 3, WiFi CSI at 2.4 GHz |
| **Signal Processing** | NumPy, SciPy (FFT, bandpass, PCA, Hampel filter) |
| **ML Classification** | scikit-learn + Impulse AI SDK (RandomForest, 98.8% acc, 30 scenarios) |
| **Backend** | Raspberry Pi 5, MQTT (Mosquitto), Python |
| **Dashboard** | Streamlit, Plotly, HTML5 Canvas, SVG animations |
| **Networking** | TP-Link AX1500 (dual-band), MQTT over LAN |
| **Environment** | DHT11, CCS811, MAX30102, GSR (via Arduino Giga R1) |
| **Storage** | Filecoin, Storacha, MongoDB Atlas |
| **Encryption** | AES-256-GCM, HMAC-SHA256 |

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/jemsbhai/plgenesis-sensus.git
cd plgenesis-sensus
pip install -r demo/requirements.txt
pip install scikit-learn
```

### Run the Demo

```bash
# Train the ML classifier (first time only, ~30 seconds)
python demo/classifier.py

# Launch the dashboard
streamlit run demo/app.py
```

### Run Integration Tests

```bash
python integrations/hypercerts.py
python integrations/data_sovereignty.py
python integrations/filecoin_store.py
python integrations/storacha_store.py
python integrations/impulse_ml.py
```

## Project Structure

```
plgenesis-sensus/
├── demo/
│   ├── app.py              # Streamlit dashboard with animated SVG room
│   ├── simulator.py         # 30-scenario physiological signal generator
│   ├── classifier.py        # ML health state classifier (RandomForest)
│   └── requirements.txt
├── integrations/
│   ├── hypercerts.py         # Impact claim generation
│   ├── data_sovereignty.py   # Encrypted vault + consent + FHIR export
│   ├── filecoin_store.py     # Decentralized storage on Filecoin
│   ├── storacha_store.py     # Storacha storage + UCAN delegations
│   └── impulse_ml.py         # ML dataset export for Impulse AI
├── pi/
│   └── services/
│       ├── csi_processor.py  # RuView-inspired CSI signal processing
│       ├── dashboard.py      # Original Flask dashboard (hardware mode)
│       ├── main.py           # MQTT event handling + sensor fusion
│       └── ...
├── esp32/                    # ESP32-C6 firmware (CSI collection)
├── arduino/                  # Arduino Giga R1 sensor firmware
├── output/                   # Generated artifacts from integrations
│   ├── hypercerts/           # Impact claim JSONs
│   ├── data_sovereignty/     # Encrypted vault exports
│   ├── filecoin/             # Storage packages + manifests
│   ├── storacha/             # UCAN delegations + knowledge base
│   ├── impulse_ai/           # Training datasets (CSV)
│   └── model/                # Trained classifier (pickle)
└── README.md
```

## Research References

- **RuView** — WiFi-based multi-person pose estimation
- **ESP32-CSI-Tool** — Open-source CSI extraction for ESP32
- **ESPectre** — Spectral analysis of CSI for activity recognition
- **MultiSense** — Multi-node CSI sensing framework
- **SpaceBeat** — Contactless vital sign monitoring via WiFi

## Clinical Relevance

| Use Case | Impact |
|----------|--------|
| **Sleep Apnea Screening** | Affects ~1B people globally. Sensus detects breathing pauses without a sleep lab. |
| **Opioid Overdose Prevention** | #1 cause of preventable hospital death. Contactless respiratory depression monitoring. |
| **Elderly Fall Detection** | Leading cause of injury death in adults 65+. Immediate detection + alerting. |
| **Post-Surgical Monitoring** | Continuous vitals without sensor fatigue or skin irritation. |
| **Medication Compliance** | Track physiological response to beta blockers, sedatives, stimulants. |
| **Mental Health** | Detect panic attacks, PTSD flashbacks, stress episodes contactlessly. |

## Team

Built for **PL_Genesis: Frontiers of Collaboration Hackathon 2026** by the Sensus team.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Sensus: Because the best sensor is the one you don't have to wear.**

</div>
]]>