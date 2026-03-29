"""
Sensus API Integrations
Sponsor-specific integrations for HackUSF 2026 prizes.
"""

import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger('sensus.api')


class GeminiClient:
    SYSTEM_PROMPT = """You are Sensus, an AI health monitoring assistant.
You receive real-time physiological data from a contactless WiFi-based sensing platform.
Your role is to interpret the data and provide clear, actionable health insights.
Be concise (2-3 sentences max). Flag anything concerning.
Always mention environmental context if relevant.
Speak in a calm, professional clinical tone.
If vital signs are normal, say so briefly.
Never diagnose - only flag patterns for clinical review."""

    def __init__(self):
        try:
            from google import genai
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and api_key != 'your_key_here':
                self.client = genai.Client(api_key=api_key)
                self.model_id = 'gemini-2.0-flash'
                self.enabled = True
                logger.info('Gemini API initialized (google.genai)')
            else:
                self.enabled = False
                logger.warning('Gemini API key not set')
        except Exception as e:
            self.enabled = False
            logger.warning(f'Gemini init failed: {e}')

    def interpret(self, state):
        if not self.enabled:
            return self._fallback_interpret(state)
        try:
            prompt = f"""{self.SYSTEM_PROMPT}

Current patient state (contactless WiFi CSI sensing):
- Heart Rate: {state.get('heart_rate', 'N/A')} bpm
- Breathing Rate: {state.get('breathing_rate', 'N/A')} breaths/min
- HRV RMSSD: {state.get('hrv_rmssd', 'N/A')} ms
- Stress Index: {state.get('stress_index', 'N/A')}
- Ground Truth HR (MAX30102): {state.get('ground_truth_hr', 'N/A')} bpm
- GSR: {state.get('gsr', 'N/A')} uS
- Alert Level: {state.get('alert_level', 'normal')}
- Room Occupancy: {state.get('occupancy', 'N/A')} persons
- Audio Events: {json.dumps(state.get('audio_events', []))}

Environmental context:
- Temperature: {state.get('environment', {}).get('temperature_c', 'N/A')} C
- Humidity: {state.get('environment', {}).get('humidity_pct', 'N/A')} %
- CO2: {state.get('environment', {}).get('co2_ppm', 'N/A')} ppm
- TVOC: {state.get('environment', {}).get('tvoc_ppb', 'N/A')} ppb
- Environmental Alerts: {state.get('environment', {}).get('alerts', [])}

Provide a brief clinical interpretation:"""
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f'Gemini API error: {e}')
            return self._fallback_interpret(state)

    def _fallback_interpret(self, state):
        parts = []
        hr = state.get('heart_rate')
        rr = state.get('breathing_rate')
        alert = state.get('alert_level', 'normal')
        if alert == 'critical':
            parts.append('CRITICAL: Vital signs outside safe range.')
        elif alert == 'warning':
            parts.append('Warning: Elevated vital signs detected.')
        else:
            parts.append('Vital signs within normal range.')
        if hr:
            parts.append(f'Heart rate {hr} bpm.')
        if rr:
            parts.append(f'Breathing rate {rr} breaths per minute.')
        env_alerts = state.get('environment', {}).get('alerts', [])
        if 'co2_elevated' in env_alerts:
            parts.append('Room CO2 is elevated - consider ventilation.')
        return ' '.join(parts)


class ElevenLabsClient:
    def __init__(self):
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if api_key and api_key != 'your_key_here':
                from elevenlabs import ElevenLabs
                self.client = ElevenLabs(api_key=api_key)
                self.voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'Rachel')
                self.enabled = True
                logger.info('ElevenLabs API initialized')
            else:
                self.enabled = False
                logger.warning('ElevenLabs API key not set')
        except Exception as e:
            self.enabled = False
            logger.warning(f'ElevenLabs init failed: {e}')

    def speak(self, text, language='en'):
        if not self.enabled:
            logger.info(f'[TTS MOCK] {text}')
            return
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id='eleven_multilingual_v2',
                output_format='mp3_44100_128',
            )
            filepath = '/tmp/sensus_alert.mp3'
            with open(filepath, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            import subprocess
            subprocess.Popen(['mpg123', '-q', filepath],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f'Spoke alert: {text[:50]}...')
        except Exception as e:
            logger.error(f'ElevenLabs TTS error: {e}')

    def speak_multilingual(self, text, target_lang='es'):
        self.speak(text, language=target_lang)


class SensusMongoClient:
    def __init__(self):
        try:
            from pymongo import MongoClient
            uri = os.getenv('MONGODB_URI')
            if uri and uri != 'mongodb+srv://user:pass@cluster.mongodb.net/sensus':
                self.client = MongoClient(uri)
                self.db = self.client['sensus']
                self.vitals = self.db['vitals']
                self.sessions = self.db['sessions']
                self.enabled = True
                logger.info('MongoDB Atlas connected')
            else:
                self.enabled = False
                logger.warning('MongoDB URI not set')
        except Exception as e:
            self.enabled = False
            logger.warning(f'MongoDB init failed: {e}')

    def store_vitals(self, state):
        if not self.enabled:
            return
        doc = {**state, 'created_at': datetime.now(timezone.utc)}
        doc.pop('_id', None)
        self.vitals.insert_one(doc)

    def create_session(self, patient_id='anonymous'):
        if not self.enabled:
            return 'mock-session-id'
        result = self.sessions.insert_one({
            'patient_id': patient_id,
            'started_at': datetime.now(timezone.utc),
            'status': 'active',
        })
        return str(result.inserted_id)

    def get_session_vitals(self, session_id, limit=100):
        if not self.enabled:
            return []
        return list(self.vitals.find(
            {'session_id': session_id}, {'_id': 0}
        ).sort('ts', -1).limit(limit))


class SnowflakeClient:
    def __init__(self):
        try:
            import snowflake.connector
            account = os.getenv('SNOWFLAKE_ACCOUNT')
            if account and account != 'your_account':
                self.conn = snowflake.connector.connect(
                    account=account,
                    user=os.getenv('SNOWFLAKE_USER'),
                    password=os.getenv('SNOWFLAKE_PASSWORD'),
                    database=os.getenv('SNOWFLAKE_DATABASE', 'SENSUS'),
                    schema=os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
                    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                )
                self._init_tables()
                self.enabled = True
                logger.info('Snowflake connected')
            else:
                self.enabled = False
                logger.warning('Snowflake account not set')
        except Exception as e:
            self.enabled = False
            logger.warning(f'Snowflake init failed: {e}')

    def _init_tables(self):
        try:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vitals_stream (
                    ts TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                    heart_rate FLOAT, breathing_rate FLOAT,
                    hrv_sdnn FLOAT, hrv_rmssd FLOAT,
                    stress_index VARCHAR, alert_level VARCHAR,
                    temperature_c FLOAT, humidity_pct FLOAT,
                    co2_ppm FLOAT, tvoc_ppb FLOAT,
                    occupancy INT, signal_quality FLOAT
                )
            """)
            cur.close()
        except Exception as e:
            logger.error(f'Snowflake table init error: {e}')

    def push_vitals(self, state):
        if not self.enabled:
            return
        try:
            cur = self.conn.cursor()
            env = state.get('environment', {})
            cur.execute("""
                INSERT INTO vitals_stream
                (heart_rate, breathing_rate, hrv_sdnn, hrv_rmssd,
                 stress_index, alert_level, temperature_c, humidity_pct,
                 co2_ppm, tvoc_ppb, occupancy, signal_quality)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                state.get('heart_rate'), state.get('breathing_rate'),
                state.get('hrv_sdnn'), state.get('hrv_rmssd'),
                state.get('stress_index'), state.get('alert_level'),
                env.get('temperature_c'), env.get('humidity_pct'),
                env.get('co2_ppm'), env.get('tvoc_ppb'),
                state.get('occupancy'), state.get('signal_quality'),
            ))
            cur.close()
        except Exception as e:
            logger.error(f'Snowflake push error: {e}')


MongoClient = SensusMongoClient
