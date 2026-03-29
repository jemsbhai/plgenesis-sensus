"""
Environmental Processor
Provides causal context for physiological changes.
"""

import logging

logger = logging.getLogger('sensus.env')


class EnvironmentalProcessor:
    """Processes DHT11 + CCS811 environmental sensor data."""

    CO2_ELEVATED = 1000
    CO2_HIGH = 2000
    TVOC_ELEVATED = 400
    TEMP_HIGH = 28.0
    TEMP_LOW = 16.0
    HUMIDITY_HIGH = 70.0
    HUMIDITY_LOW = 30.0

    def get_current(self, env_buffer):
        if not env_buffer:
            return {}
        latest = env_buffer[-1]
        state = {
            'temperature_c': latest.get('temp'),
            'humidity_pct': latest.get('humidity'),
            'co2_ppm': latest.get('co2'),
            'tvoc_ppb': latest.get('tvoc'),
            'pressure_hpa': latest.get('pressure'),
            'alerts': [],
        }
        if state['co2_ppm'] and state['co2_ppm'] > self.CO2_HIGH:
            state['alerts'].append('co2_critical')
        elif state['co2_ppm'] and state['co2_ppm'] > self.CO2_ELEVATED:
            state['alerts'].append('co2_elevated')
        if state['tvoc_ppb'] and state['tvoc_ppb'] > self.TVOC_ELEVATED:
            state['alerts'].append('tvoc_elevated')
        if state['temperature_c']:
            if state['temperature_c'] > self.TEMP_HIGH:
                state['alerts'].append('temp_high')
            elif state['temperature_c'] < self.TEMP_LOW:
                state['alerts'].append('temp_low')
        if state['humidity_pct']:
            if state['humidity_pct'] > self.HUMIDITY_HIGH:
                state['alerts'].append('humidity_high')
            elif state['humidity_pct'] < self.HUMIDITY_LOW:
                state['alerts'].append('humidity_low')
        if len(env_buffer) > 60:
            recent_co2 = [e.get('co2', 0) for e in env_buffer[-60:] if e.get('co2')]
            if len(recent_co2) > 10:
                state['co2_trend'] = 'rising' if recent_co2[-1] > recent_co2[0] + 50 else \
                                     'falling' if recent_co2[-1] < recent_co2[0] - 50 else 'stable'
        return state

    def correlate_with_vitals(self, env_state, vitals):
        causes = []
        rr = vitals.get('breathing_rate')
        hr = vitals.get('heart_rate')
        if rr and rr > 20:
            if 'co2_elevated' in env_state.get('alerts', []):
                causes.append('elevated_breathing_likely_co2')
            elif 'temp_high' in env_state.get('alerts', []):
                causes.append('elevated_breathing_likely_heat')
            else:
                causes.append('elevated_breathing_investigate')
        if hr and hr > 100:
            if 'temp_high' in env_state.get('alerts', []):
                causes.append('elevated_hr_likely_heat')
            else:
                causes.append('elevated_hr_investigate')
        return causes if causes else ['vitals_normal_environment_ok']
