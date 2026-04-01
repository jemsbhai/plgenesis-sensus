"""
Generate large, specialized datasets for Impulse AI upload.
Multiple datasets, more samples per scenario, with random variation.
"""

import csv
import os
import sys
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
from simulator import build_all_scenarios, create_engine

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'impulse_ai')

SCENARIO_LABELS = {
    1: 'healthy_resting', 2: 'light_activity', 3: 'deep_sleep',
    4: 'rem_sleep', 5: 'acute_stress', 6: 'meditation',
    7: 'panic_attack', 8: 'cognitive_load', 9: 'night_terror',
    10: 'tachycardia', 11: 'bradycardia', 12: 'atrial_fibrillation',
    13: 'cardiac_arrest', 14: 'pvc_ectopic', 15: 'sleep_apnea',
    16: 'hyperventilation', 17: 'asthma_attack', 18: 'copd_exacerbation',
    19: 'stroke', 20: 'fall_detection', 21: 'seizure',
    22: 'anaphylaxis', 23: 'beta_blocker_response', 24: 'opioid_sedation',
    25: 'stimulant_effect', 26: 'sedative_recovery',
    27: 'occupancy_change', 28: 'multi_person',
    29: 'poor_air_quality', 30: 'heat_stress',
}

SEVERITY_MAP = {
    'healthy_resting': 'normal', 'light_activity': 'normal', 'deep_sleep': 'normal',
    'rem_sleep': 'normal', 'meditation': 'normal', 'cognitive_load': 'normal',
    'occupancy_change': 'normal', 'multi_person': 'normal',
    'acute_stress': 'warning', 'panic_attack': 'critical',
    'night_terror': 'critical', 'tachycardia': 'critical',
    'bradycardia': 'critical', 'atrial_fibrillation': 'warning',
    'cardiac_arrest': 'critical', 'pvc_ectopic': 'warning',
    'sleep_apnea': 'warning', 'hyperventilation': 'warning',
    'asthma_attack': 'warning', 'copd_exacerbation': 'critical',
    'stroke': 'critical', 'fall_detection': 'critical',
    'seizure': 'critical', 'anaphylaxis': 'critical',
    'beta_blocker_response': 'normal', 'opioid_sedation': 'warning',
    'stimulant_effect': 'normal', 'sedative_recovery': 'normal',
    'poor_air_quality': 'warning', 'heat_stress': 'warning',
}

FEATURE_COLS = [
    'heart_rate', 'breathing_rate', 'hrv_rmssd', 'hrv_sdnn',
    'spo2', 'gsr', 'motion_level', 'blood_pressure_sys',
    'blood_pressure_dia', 'skin_temp', 'stress_encoded',
    'is_motion', 'is_present', 'irregular_rhythm', 'signal_quality',
]


def extract_row(state, label, scenario_id):
    stress_map = {'low': 0, 'moderate': 1, 'high': 2}
    return {
        'heart_rate': round(state.get('heart_rate', 0), 1),
        'breathing_rate': round(state.get('breathing_rate', 0), 1),
        'hrv_rmssd': round(state.get('hrv_rmssd', 0), 1),
        'hrv_sdnn': round(state.get('hrv_sdnn', 0), 1),
        'spo2': round(state.get('spo2', 98), 1),
        'gsr': round(state.get('gsr', 2.5), 2),
        'motion_level': round(state.get('motion_level', 0), 3),
        'blood_pressure_sys': round(state.get('blood_pressure_sys', 120), 0),
        'blood_pressure_dia': round(state.get('blood_pressure_dia', 80), 0),
        'skin_temp': round(state.get('skin_temp', 36.5), 1),
        'stress_encoded': stress_map.get(state.get('stress_index', 'low'), 0),
        'is_motion': int(state.get('is_motion', False)),
        'is_present': int(state.get('is_present', True)),
        'irregular_rhythm': int(state.get('irregular_rhythm', False)),
        'signal_quality': round(state.get('signal_quality', 10), 1),
        'health_state': label,
        'scenario_id': scenario_id,
    }


def add_noise(row, noise_level=0.05):
    """Add small random noise to numeric features for variation."""
    noisy = dict(row)
    rng = np.random.default_rng()
    for col in ['heart_rate', 'breathing_rate', 'hrv_rmssd', 'hrv_sdnn', 'gsr',
                'motion_level', 'blood_pressure_sys', 'blood_pressure_dia', 
                'skin_temp', 'signal_quality']:
        val = float(noisy[col])
        if val != 0:
            noisy[col] = round(val * (1 + rng.normal(0, noise_level)), 2)
        else:
            noisy[col] = round(abs(rng.normal(0, 0.5)), 2)
    # Clamp spo2
    spo2 = float(noisy.get('spo2', 98))
    noisy['spo2'] = round(min(100, max(0, spo2 * (1 + rng.normal(0, noise_level * 0.3)))), 1)
    return noisy


def generate_dataset(name, scenario_ids, runs_per_scenario=5, speed=50.0, noise=0.05):
    """Generate a dataset CSV with multiple noisy runs per scenario."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenarios = build_all_scenarios()
    all_rows = []

    for sid in scenario_ids:
        if sid not in scenarios:
            continue
        scenario = scenarios[sid]
        label = SCENARIO_LABELS.get(sid, f'scenario_{sid}')
        steps = int(scenario.total_duration_sec / 0.1) + 1

        for run in range(runs_per_scenario):
            engine = create_engine(sid, speed=speed)
            for _ in range(steps):
                state = engine.step(0.1)
                row = extract_row(state, label, sid)
                if run > 0:
                    row = add_noise(row, noise)
                all_rows.append(row)

    # Write CSV
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    fieldnames = list(all_rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  ✅ {name}.csv — {len(all_rows)} samples from {len(scenario_ids)} scenarios × {runs_per_scenario} runs")
    return path, len(all_rows)


def generate_anomaly_dataset(runs_per_scenario=5, speed=50.0):
    """Binary anomaly detection: normal vs emergency."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenarios = build_all_scenarios()
    all_rows = []

    for sid, scenario in scenarios.items():
        label = SCENARIO_LABELS.get(sid, f'scenario_{sid}')
        severity = SEVERITY_MAP.get(label, 'normal')
        is_anomaly = 1 if severity == 'critical' else 0
        steps = int(scenario.total_duration_sec / 0.1) + 1

        for run in range(runs_per_scenario):
            engine = create_engine(sid, speed=speed)
            for _ in range(steps):
                state = engine.step(0.1)
                row = extract_row(state, label, sid)
                if run > 0:
                    row = add_noise(row, 0.05)
                # Replace health_state with binary label
                row['is_anomaly'] = is_anomaly
                del row['health_state']
                all_rows.append(row)

    path = os.path.join(OUTPUT_DIR, 'anomaly_detection_large.csv')
    fieldnames = list(all_rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    normal = sum(1 for r in all_rows if r['is_anomaly'] == 0)
    anomaly = sum(1 for r in all_rows if r['is_anomaly'] == 1)
    print(f"  ✅ anomaly_detection_large.csv — {len(all_rows)} samples ({normal} normal, {anomaly} anomaly)")
    return path, len(all_rows)


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING LARGE DATASETS FOR IMPULSE AI")
    print("=" * 70)
    total = 0

    # Dataset 1: Full 30-class classification (5 runs each = ~90K samples)
    print("\n📊 Dataset 1: Full 30-class health classification")
    _, n = generate_dataset(
        'sensus_full_30class',
        list(range(1, 31)),
        runs_per_scenario=5,
    )
    total += n

    # Dataset 2: Cardiac-focused (scenarios 1,10-14) — 10 runs
    print("\n📊 Dataset 2: Cardiac events focused")
    _, n = generate_dataset(
        'sensus_cardiac_focused',
        [1, 3, 10, 11, 12, 13, 14],
        runs_per_scenario=10,
    )
    total += n

    # Dataset 3: Respiratory-focused (1,3,15-18) — 10 runs
    print("\n📊 Dataset 3: Respiratory events focused")
    _, n = generate_dataset(
        'sensus_respiratory_focused',
        [1, 3, 15, 16, 17, 18],
        runs_per_scenario=10,
    )
    total += n

    # Dataset 4: Emergency vs Normal binary (5 runs)
    print("\n📊 Dataset 4: Emergency detection (binary)")
    _, n = generate_dataset(
        'sensus_emergency_detection',
        [1, 2, 3, 6, 7, 10, 13, 19, 20, 21, 22],
        runs_per_scenario=8,
    )
    total += n

    # Dataset 5: Medication tracking (1, 23-26) — 10 runs
    print("\n📊 Dataset 5: Medication response tracking")
    _, n = generate_dataset(
        'sensus_medication_tracking',
        [1, 23, 24, 25, 26],
        runs_per_scenario=10,
    )
    total += n

    # Dataset 6: Anomaly detection binary (large)
    print("\n📊 Dataset 6: Anomaly detection (binary, large)")
    _, n = generate_anomaly_dataset(runs_per_scenario=5)
    total += n

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total:,} samples across 6 datasets")
    print(f"Output: {OUTPUT_DIR}")

    # List files
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"  {f} — {size/1024/1024:.1f} MB")
