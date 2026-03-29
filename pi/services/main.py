"""
Sensus — Multi-Modal Health Sensing Fusion Engine
===================================================
Main service running on Raspberry Pi 5.
Receives data from ESP32-C6 (CSI + BLE + sensors), ESP32-S3 (audio),
and Arduino Giga (env sensors + auth) via MQTT.

Inspired by RuView WiFi DensePose architecture.
"""

import json
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

from csi_processor import CSIProcessor, MultiNodeFusion
from audio_processor import AudioProcessor
from env_processor import EnvironmentalProcessor
from health_engine import HealthEngine
from api_integrations import (GeminiClient, ElevenLabsClient,
                               SnowflakeClient, MongoClient as SensusMongoClient)

# Load .env from config directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('sensus')

# ─── Global Control Flags (default OFF to avoid API rate limiting) ───
gemini_enabled = False
elevenlabs_enabled = False


class SensorBuffers:
    """Thread-safe buffers for all sensor modalities."""

    def __init__(self, max_len=1000):
        self.csi = {f'node_{i}': deque(maxlen=max_len) for i in range(1, 9)}
        self.csi.update({f'node_s3_{i}': deque(maxlen=max_len) for i in range(1, 4)})
        self.ble_rssi = deque(maxlen=200)
        self.env = deque(maxlen=500)
        self.audio_features = deque(maxlen=200)
        self.hr_ground_truth = deque(maxlen=100)
        self.gsr = deque(maxlen=200)
        self.auth_events = deque(maxlen=50)
        self.lock = threading.Lock()
        # Current authenticated user (from RFID/fingerprint)
        self.current_user = None
        self.auth_time = None

    def add_csi(self, node_id, data):
        with self.lock:
            if node_id not in self.csi:
                self.csi[node_id] = deque(maxlen=1000)
            self.csi[node_id].append(data)

    def add_ble(self, data):
        with self.lock:
            self.ble_rssi.append(data)

    def add_env(self, data):
        with self.lock:
            self.env.append(data)

    def add_audio(self, data):
        with self.lock:
            self.audio_features.append(data)

    def add_hr(self, data):
        with self.lock:
            self.hr_ground_truth.append(data)

    def add_gsr(self, data):
        with self.lock:
            self.gsr.append(data)

    def add_auth(self, data):
        with self.lock:
            self.auth_events.append(data)
            method = data.get('method', 'unknown')
            if method == 'rfid':
                self.current_user = data.get('uid', 'unknown')
                self.auth_time = time.time()
                logger.info(f'AUTH: RFID badge {self.current_user}')
            elif method == 'fingerprint' and data.get('id', -1) >= 0:
                self.current_user = f'fp_{data["id"]}'
                self.auth_time = time.time()
                logger.info(f'AUTH: Fingerprint ID {data["id"]} (conf: {data.get("confidence")})')


class MQTTHandler:
    """Handles MQTT subscriptions and message routing."""

    TOPICS = [
        'sensus/+/csi',
        'sensus/+/ble',
        'sensus/+/env',
        'sensus/+/audio',
        'sensus/+/hr',
        'sensus/+/gsr',
        'sensus/+/auth',
        'sensus/+/status',
        'sensus/control/#',
    ]

    def __init__(self, buffers: SensorBuffers, broker='localhost', port=1883):
        self.buffers = buffers
        self.client = mqtt.Client(client_id='sensus-fusion', protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker = broker
        self.port = port

    def start(self):
        self.client.connect(self.broker, self.port)
        self.client.loop_start()
        logger.info(f'MQTT connected to {self.broker}:{self.port}')

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        for topic in self.TOPICS:
            client.subscribe(topic)
            logger.info(f'Subscribed: {topic}')

    def _on_message(self, client, userdata, msg):
        global gemini_enabled, elevenlabs_enabled
        try:
            topic = msg.topic

            # Handle control messages
            if topic == 'sensus/control/gemini':
                val = msg.payload.decode().strip().lower()
                gemini_enabled = val in ('on', 'true', '1')
                logger.info(f'Gemini {"ENABLED" if gemini_enabled else "DISABLED"}')
                return
            if topic == 'sensus/control/elevenlabs':
                val = msg.payload.decode().strip().lower()
                elevenlabs_enabled = val in ('on', 'true', '1')
                logger.info(f'ElevenLabs {"ENABLED" if elevenlabs_enabled else "DISABLED"}')
                return

            payload = json.loads(msg.payload.decode())
            parts = topic.split('/')
            node_id = parts[1]
            data_type = parts[2]
            payload['_node'] = node_id
            payload['_ts'] = datetime.now(timezone.utc).isoformat()
            payload['_epoch'] = time.time()

            if data_type == 'csi':
                self.buffers.add_csi(node_id, payload)
            elif data_type == 'ble':
                self.buffers.add_ble(payload)
            elif data_type == 'env':
                self.buffers.add_env(payload)
            elif data_type == 'audio':
                self.buffers.add_audio(payload)
            elif data_type == 'hr':
                self.buffers.add_hr(payload)
            elif data_type == 'gsr':
                self.buffers.add_gsr(payload)
            elif data_type == 'auth':
                self.buffers.add_auth(payload)
            elif data_type == 'status':
                logger.info(f'Node {node_id}: {payload}')
        except Exception as e:
            logger.error(f'MQTT msg error: {e}')

    def publish(self, topic, payload):
        if isinstance(payload, dict):
            self.client.publish(topic, json.dumps(payload))
        else:
            self.client.publish(topic, str(payload))


class FusionLoop:
    """
    Core fusion engine: runs at 1 Hz, combines all sensor modalities
    into a unified health state, publishes to MQTT + stores in DB.
    """

    def __init__(self, buffers, mqtt_handler):
        self.buffers = buffers
        self.mqtt = mqtt_handler
        self.csi_proc = CSIProcessor()
        self.node_fusion = MultiNodeFusion()
        self.audio_proc = AudioProcessor()
        self.env_proc = EnvironmentalProcessor()
        self.health = HealthEngine()
        self.running = False
        self._prev_alert = 'normal'

        # Lazy-loaded API clients
        self._gemini = None
        self._elevenlabs = None
        self._snowflake = None
        self._mongo = None

    @property
    def gemini(self):
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

    @property
    def elevenlabs(self):
        if self._elevenlabs is None:
            self._elevenlabs = ElevenLabsClient()
        return self._elevenlabs

    @property
    def snowflake(self):
        if self._snowflake is None:
            self._snowflake = SnowflakeClient()
        return self._snowflake

    @property
    def mongo(self):
        if self._mongo is None:
            self._mongo = SensusMongoClient()
        return self._mongo

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info('Fusion loop started (1 Hz)')

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                t0 = time.time()
                state = self._fuse()

                if state:
                    state['gemini_active'] = gemini_enabled
                    state['elevenlabs_active'] = elevenlabs_enabled

                    # Add auth info
                    if self.buffers.current_user:
                        state['authenticated_user'] = self.buffers.current_user
                        state['auth_time'] = self.buffers.auth_time

                    # Publish fused state (strip spectrogram — too large for MQTT)
                    mqtt_state = {k: v for k, v in state.items() if k != 'spectrogram'}
                    self.mqtt.publish('sensus/fused/vitals', mqtt_state)

                    # ─── Push alert level to Giga NeoPixels ───
                    alert = state.get('alert_level', 'normal')
                    if alert != self._prev_alert:
                        self.mqtt.publish('sensus/giga/alert', alert)
                        self._prev_alert = alert
                        logger.info(f'Alert level → {alert} (sent to Giga LEDs)')

                    # MongoDB (every tick)
                    try:
                        self.mongo.store_vitals(state)
                    except Exception:
                        pass  # Silent fail — may not be configured

                    # Snowflake (every 30s)
                    if int(t0) % 30 == 0:
                        try:
                            self.snowflake.push_vitals(state)
                        except Exception:
                            pass

                    # Gemini interpretation (every 10s when enabled)
                    if gemini_enabled and int(t0) % 10 == 0:
                        self._ai_interpret(state)

                elapsed = time.time() - t0
                time.sleep(max(0, 1.0 - elapsed))

            except Exception as e:
                logger.error(f'Fusion loop error: {e}')
                time.sleep(1)

    def _fuse(self):
        """Fuse all sensor modalities into unified health state."""
        with self.buffers.lock:
            # ─── CSI Processing (per-node then multi-node fusion) ───
            per_node_vitals = {}
            for node_id, buf in self.buffers.csi.items():
                if len(buf) > 50:
                    vitals = self.csi_proc.extract_vitals(list(buf))
                    if vitals:
                        per_node_vitals[node_id] = vitals

            # Multi-node SNR-weighted fusion (RuView-inspired)
            csi_fused = self.node_fusion.fuse_nodes(per_node_vitals)

            # ─── Environmental ───
            env_state = {}
            if self.buffers.env:
                env_state = self.env_proc.get_current(list(self.buffers.env))

            # ─── Audio ───
            audio_events = []
            if self.buffers.audio_features:
                audio_events = self.audio_proc.classify_events(
                    list(self.buffers.audio_features))

            # ─── Ground truth HR (MAX30102) ───
            gt_hr = None
            if self.buffers.hr_ground_truth:
                latest_hr = self.buffers.hr_ground_truth[-1]
                gt_hr = latest_hr.get('hr')
                # Convert string to float if needed (Giga sends serialized())
                if isinstance(gt_hr, str):
                    try:
                        gt_hr = float(gt_hr)
                    except ValueError:
                        gt_hr = None
                # Ignore zero (no finger on sensor)
                if gt_hr is not None and gt_hr <= 0:
                    gt_hr = None

            # ─── GSR ───
            gsr_val = None
            if self.buffers.gsr:
                raw_gsr = self.buffers.gsr[-1].get('conductance')
                if isinstance(raw_gsr, str):
                    try:
                        gsr_val = float(raw_gsr)
                    except ValueError:
                        gsr_val = None
                else:
                    gsr_val = raw_gsr

            # ─── BLE Occupancy ───
            ble_count = 0
            ble_devices = []
            if self.buffers.ble_rssi:
                recent = [b for b in self.buffers.ble_rssi
                          if time.time() - b.get('_epoch', 0) < 30]
                seen = set(b.get('mac') for b in recent if b.get('mac'))
                ble_count = len(seen)
                ble_devices = list(seen)

        # ─── Health Engine Fusion ───
        state = self.health.fuse(
            csi_vitals=per_node_vitals,
            env=env_state,
            audio_events=audio_events,
            gt_hr=gt_hr,
            gsr=gsr_val,
            ble_count=ble_count,
            ble_devices=ble_devices,
        )

        # Merge CSI-specific fields from multi-node fusion
        if csi_fused:
            for key in ['waveform', 'spectrogram', 'is_present', 'presence_score',
                        'is_motion', 'motion_level', 'activity',
                        'hr_confidence', 'breath_confidence',
                        'snr_cardiac', 'snr_breath', 'hrv_pnn50']:
                if key in csi_fused and csi_fused[key] is not None:
                    state[key] = csi_fused[key]

            # Override vitals with fused values if better SNR
            if csi_fused.get('signal_quality', 0) > state.get('signal_quality', 0):
                for key in ['heart_rate', 'breathing_rate', 'hrv_sdnn', 'hrv_rmssd']:
                    if csi_fused.get(key):
                        state[key] = csi_fused[key]
                state['signal_quality'] = csi_fused['signal_quality']
                state['fusion_method'] = csi_fused.get('fusion_method', 'single')
                state['node_count'] = csi_fused.get('node_count', 1)

        return state

    def _ai_interpret(self, state):
        """Generate AI health interpretation via Gemini."""
        try:
            interpretation = self.gemini.interpret(state)
            if interpretation:
                state['ai_interpretation'] = interpretation
                self.mqtt.publish('sensus/ai/interpretation', {
                    'text': interpretation,
                    'ts': datetime.now(timezone.utc).isoformat()
                })

                # Voice alert for critical/warning states
                if elevenlabs_enabled and state.get('alert_level', 'normal') != 'normal':
                    self.elevenlabs.speak(interpretation)

        except Exception as e:
            logger.warning(f'AI interpretation error: {e}')


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    logger.info('╔══════════════════════════════════════════╗')
    logger.info('║  SENSUS — Contactless Health Sensing     ║')
    logger.info('║  HackUSF 2026                            ║')
    logger.info('║  WiFi CSI + BLE + Audio + Environmental  ║')
    logger.info('║  + RFID/Fingerprint Auth + NeoPixel LEDs ║')
    logger.info('╚══════════════════════════════════════════╝')
    logger.info('Initializing sensor fusion engine...')

    buffers = SensorBuffers()
    mqtt_handler = MQTTHandler(buffers)
    fusion = FusionLoop(buffers, mqtt_handler)

    mqtt_handler.start()
    fusion.start()

    from dashboard import create_app
    app = create_app(buffers, mqtt_handler)

    logger.info('─── System Ready ───')
    logger.info(f'Dashboard: http://0.0.0.0:5000')
    logger.info(f'Gemini: OFF (toggle from dashboard)')
    logger.info(f'ElevenLabs: OFF (toggle from dashboard)')
    logger.info(f'Waiting for sensor mesh nodes on MQTT...')

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
