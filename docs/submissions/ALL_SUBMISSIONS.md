<![CDATA[# PL_Genesis Submission Summaries
=================================

## 1. NEUROTECH (Primary Track — $6K pool)

**Title:** Sensus — Contactless Neuro-Physiological Sensing via WiFi CSI

**Summary:**
Sensus is a contactless multi-modal health sensing platform that extracts vital signs from ordinary WiFi signals using Channel State Information (CSI). By deploying a mesh of 3 ESP32-C6 nodes, Sensus detects the micro-displacements of breathing (1-5mm) and heartbeat (0.1-0.5mm) through Fresnel zone physics — extracting heart rate, breathing rate, HRV (SDNN/RMSSD/pNN50), motion classification, and presence detection without any body-worn sensors.

The platform includes a virtual scenario engine with 30 physiologically accurate clinical scenarios (cardiac arrest, seizure, sleep apnea, medication tracking, stress response, etc.) and a real-time ML classifier trained on all scenarios that provides live health state prediction with risk scoring.

**Why Neurotech:** Sensus operates at the intersection of cognition, physiological sensing, and computation. Our HRV analysis provides autonomic nervous system insight (sympathetic/parasympathetic balance), stress detection monitors cognitive load, and the EEG-integration architecture (Emotiv EPOC X ready) enables direct neural-physiological correlation. The contactless approach addresses neural data rights by eliminating wearable compliance barriers.

---

## 2. AI & ROBOTICS ($6K pool)

**Title:** Sensus — Sensor Fusion AI for Contactless Health Monitoring

**Summary:**
Sensus demonstrates advanced sensor fusion and real-time AI inference for autonomous health monitoring. The system uses a mesh of ESP32-C6 WiFi nodes as a distributed sensing array, processes CSI data through a physics-informed pipeline (conjugate multiplication, Hampel filtering, PCA, FFT, Fresnel modeling), and fuses multi-node signals using SNR-weighted algorithms.

A RandomForest ML classifier trained on 30 clinical scenarios provides real-time health state classification across 10 risk categories, with live anomaly detection. The dashboard features animated room visualization showing WiFi signals interacting with the patient in real-time.

**Why AI & Robotics:** This is a sensor fusion network where distributed nodes share environmental data (CSI) to perform intelligent health inference. The real-time interpretability dashboard surfaces the AI's reasoning transparently. The system demonstrates human-in-the-loop oversight with configurable alert thresholds.

---

## 3. INFRASTRUCTURE & DIGITAL RIGHTS ($6K pool)

**Title:** Sensus — Health Data Sovereignty via Encrypted Personal Vaults

**Summary:**
Sensus includes a personal health data sovereignty layer that gives patients complete ownership and control over their continuously-generated physiological data. The system implements AES-256-GCM encryption at rest with granular per-field consent management — a patient can grant their cardiologist access to heart rate and HRV while keeping stress and motion data private.

Features: FHIR-compatible portable data export, time-bounded revocable consent grants, immutable audit logging, and W3C-inspired verifiable health attestations. All health data is encrypted field-by-field, enabling fine-grained access control impossible with traditional EHR systems.

**Why Infrastructure & Digital Rights:** Health data is the most sensitive personal data. Sensus generates continuous physiological streams that must be owned by the patient, not the platform. Our vault architecture with granular consent and portable export embodies the vision of user-owned data infrastructure.

---

## 4. HYPERCERTS ($2.5K pool)

**Title:** Sensus × Hypercerts — Verifiable Health Impact Claims from Sensor Data

**Summary:**
Sensus generates structured hypercert-compatible impact claims from health monitoring sessions. Each session produces CID-addressable evidence documenting: vital signs extracted, clinical scenarios detected, alerts generated, and sensor mesh configuration. The evidence is hashed (SHA-256) and structured for on-chain minting.

Example: A 60-second sleep apnea monitoring session generates a hypercert claiming "Contactless detection of 3 apnea events with 15-second breathing pauses, triggering clinical alerts" — with full sensor telemetry as verifiable evidence.

**Why Hypercerts:** Sensus is exactly what the Hypercerts challenge describes — "connect existing sources (APIs, sensors, reports) to automatically generate hypercert claims." Our sensors produce real-time health impact data that flows directly into structured, CID-rooted impact claims.

---

## 5. FILECOIN ($2.5K pool)

**Title:** Sensus × Filecoin — Decentralized Health Record Storage

**Summary:**
Sensus packages health monitoring sessions as content-addressed data objects for storage on Filecoin. Each session is compacted into a JSON package containing vital sign time-series, alerts, environmental data, and integrity hashes. Packages receive CIDs for trustless retrieval and verification.

The system generates CAR file manifests for batch upload, creates storage deal structures compatible with the Filecoin calibration testnet, and supports the Synapse SDK / web3.storage API for actual uploads.

**Why Filecoin:** Health records must survive any single platform failure. By storing CID-addressed health data on Filecoin, patient records become immutable, verifiable, and portable across providers — with cryptographic proof of data persistence.

---

## 6. STORACHA ($500 pool)

**Title:** Sensus × Storacha — Persistent Health Agent Memory with UCAN Delegation

**Summary:**
Sensus stores health monitoring data on Storacha with UCAN-based delegation chains for clinician access control. A patient can delegate read access to their doctor, who can re-delegate to a specialist — creating verifiable chains of trust without centralized access control.

The system also builds a decentralized health knowledge base from stored sessions, enabling AI agent RAG (Retrieval-Augmented Generation) for clinical reasoning across a patient's full history.

**Why Storacha:** Maps directly to Challenge #1 (Persistent Agent Memory) and #3 (Decentralized RAG Knowledge Base). Health data persists across sessions on Storacha, UCAN delegations enable secure multi-provider sharing, and the knowledge base enables intelligent health AI.

---

## 7. IMPULSE AI ($500 pool)

**Title:** Sensus × Impulse AI — Autonomous Health State Classification

**Summary:**
Sensus exports structured ML training datasets from its 30-scenario simulator for the Impulse AI platform. Three dataset types: 30-class health state classification (CSV), binary anomaly detection, and time-series vital sign forecasting. Each includes Impulse AI-compatible metadata with feature descriptions, target columns, and recommended training parameters.

A local RandomForest classifier demonstrates the model pipeline end-to-end — trained on synthetic data, it provides real-time inference in the dashboard with 10-class health state prediction and risk scoring.

**Why Impulse AI:** Sensus generates the labeled training data; Impulse AI provides the autonomous training and deployment pipeline. Upload our CSV → describe "classify health emergencies from vital signs" → get a production-ready model in minutes.

---

## 8. FRESH CODE ($50K pool — Top 10 get $5K each)

Use the Neurotech submission as the primary, referencing all sponsor challenge integrations.

---

## 9. COMMUNITY VOTE BOUNTY ($1K)

**Tweet template:**

🫀 Introducing SENSUS — we can see your heartbeat through WiFi.

Sensus uses WiFi CSI (Channel State Information) to extract vital signs contactlessly:
♥ Heart Rate
🫁 Breathing Rate  
📊 HRV, SpO2, Blood Pressure
⚠ 30 clinical emergency scenarios
🧠 Real-time ML classification

No sensors on the body. Just WiFi.

Built for @PL__Genesis @protocollabs #PLGenesis

🧬 Neurotech + AI track
📦 Filecoin + Storacha + Hypercerts integration
🔐 Health data sovereignty with encrypted vaults

Demo: [link]
GitHub: github.com/jemsbhai/plgenesis-sensus
]]>