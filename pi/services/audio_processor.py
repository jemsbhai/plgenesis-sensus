"""
Audio Processor
Classifies acoustic health events from USB mic / INMP441 features.
"""

import numpy as np
import logging

logger = logging.getLogger('sensus.audio')


class AudioProcessor:
    """Classifies audio events for health monitoring."""

    COUGH_ENERGY_THRESHOLD = 0.7
    SNORE_ENERGY_THRESHOLD = 0.4
    SILENCE_THRESHOLD = 0.05

    def classify_events(self, audio_buffer):
        events = []
        if not audio_buffer:
            return events
        recent = audio_buffer[-20:]
        for frame in recent:
            energy = frame.get('energy', 0)
            zcr = frame.get('zcr', 0)
            centroid = frame.get('spectral_centroid', 0)
            if energy > self.COUGH_ENERGY_THRESHOLD and zcr > 0.1:
                events.append({'type': 'cough', 'confidence': min(energy, 1.0), 'ts': frame.get('ts')})
            elif energy > self.SNORE_ENERGY_THRESHOLD and zcr < 0.05:
                events.append({'type': 'snore', 'confidence': min(energy, 1.0), 'ts': frame.get('ts')})
            elif energy > 0.3 and 0.03 < zcr < 0.1:
                events.append({'type': 'speech', 'confidence': min(energy, 1.0), 'ts': frame.get('ts')})
        deduplicated = []
        for evt in events:
            if not deduplicated or deduplicated[-1]['type'] != evt['type']:
                deduplicated.append(evt)
        return deduplicated

    def compute_voice_biomarkers(self, mfcc_sequence):
        if not mfcc_sequence or len(mfcc_sequence) < 10:
            return {}
        mfccs = np.array(mfcc_sequence)
        return {
            'jitter': float(np.std(np.diff(mfccs[:, 0]))),
            'shimmer': float(np.std(mfccs[:, 1])),
            'mfcc_mean': [float(x) for x in np.mean(mfccs, axis=0)],
            'mfcc_std': [float(x) for x in np.std(mfccs, axis=0)],
            'speech_rate_proxy': float(np.mean(np.abs(np.diff(mfccs[:, 0])))),
        }
