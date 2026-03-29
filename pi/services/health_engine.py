"""
Health Engine
Fuses multi-modal sensor data into a unified patient health state.
"""

import time
import logging

logger = logging.getLogger('sensus.health')


class HealthEngine:
    """Multi-modal health state fusion."""

    def fuse(self, csi_vitals=None, env=None, audio_events=None,
             gt_hr=None, gsr=None, ble_count=0, ble_devices=None):
        state = {
            'ts': time.time(),
            'heart_rate': None,
            'breathing_rate': None,
            'hrv_sdnn': None,
            'hrv_rmssd': None,
            'stress_index': None,
            'alert_level': 'normal',
            'audio_events': audio_events or [],
            'environment': env or {},
            'occupancy': ble_count,
            'devices': ble_devices or [],
            'ground_truth_hr': gt_hr,
            'gsr': gsr,
            'signal_quality': 0,
            'data_sources': [],
        }

        hr_estimates = []
        best_snr = -999

        if csi_vitals:
            for node_id, vitals in csi_vitals.items():
                if vitals.get('heart_rate'):
                    hr_estimates.append({
                        'value': vitals['heart_rate'],
                        'snr': vitals.get('signal_quality', 0),
                        'source': f'csi_{node_id}'
                    })
                    if vitals.get('signal_quality', 0) > best_snr:
                        best_snr = vitals['signal_quality']
                        state['heart_rate'] = vitals['heart_rate']
                        state['breathing_rate'] = vitals.get('breathing_rate')
                        state['hrv_sdnn'] = vitals.get('hrv_sdnn')
                        state['hrv_rmssd'] = vitals.get('hrv_rmssd')
                        state['signal_quality'] = vitals.get('signal_quality', 0)
                        state['data_sources'].append(f'csi_{node_id}')

        if gt_hr and gt_hr > 30 and gt_hr < 200:
            state['ground_truth_hr'] = gt_hr
            state['data_sources'].append('max30102')
            if state['heart_rate']:
                deviation = abs(state['heart_rate'] - gt_hr)
                if deviation > 15:
                    state['hr_confidence'] = 'low'
                elif deviation > 5:
                    state['hr_confidence'] = 'medium'
                else:
                    state['hr_confidence'] = 'high'

        if state['hrv_rmssd'] and state['hrv_rmssd'] > 0:
            if state['hrv_rmssd'] < 20:
                state['stress_index'] = 'high'
            elif state['hrv_rmssd'] < 40:
                state['stress_index'] = 'moderate'
            else:
                state['stress_index'] = 'low'

        if gsr and gsr > 5.0:
            if state['stress_index'] != 'high':
                state['stress_index'] = 'moderate'
            state['data_sources'].append('gsr')

        if env:
            state['data_sources'].append('environmental')
            env_alerts = env.get('alerts', [])
            if any('critical' in a for a in env_alerts):
                state['alert_level'] = 'warning'

        if audio_events:
            state['data_sources'].append('audio')
            cough_count = sum(1 for e in audio_events if e['type'] == 'cough')
            if cough_count >= 3:
                state['alert_level'] = 'warning'
                state['audio_alert'] = f'{cough_count} coughs detected'

        if state['heart_rate']:
            if state['heart_rate'] > 120 or state['heart_rate'] < 45:
                state['alert_level'] = 'critical'
            elif state['heart_rate'] > 100 or state['heart_rate'] < 50:
                if state['alert_level'] == 'normal':
                    state['alert_level'] = 'warning'

        if state['breathing_rate']:
            if state['breathing_rate'] > 30 or state['breathing_rate'] < 6:
                state['alert_level'] = 'critical'
            elif state['breathing_rate'] > 22:
                if state['alert_level'] == 'normal':
                    state['alert_level'] = 'warning'

        return state
