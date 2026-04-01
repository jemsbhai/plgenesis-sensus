"""
Impulse AI Integration — ML Training Dataset Export
====================================================
Exports structured training datasets from Sensus scenarios for
training vital sign classification models on Impulse AI platform.

Challenge: "Autonomous ML for Every App"

How Sensus fits:
  - Sensus simulator generates labeled physiological data
  - Each scenario is a labeled dataset (normal, tachycardia, apnea, etc.)
  - Export in CSV/JSON format compatible with Impulse AI platform
  - Train models to classify health states from CSI features

Features:
  - Multi-class health state classification dataset
  - Time-series vital sign prediction dataset
  - Anomaly detection training set (normal vs. emergency)
  - Feature engineering pipeline metadata

Usage:
    from integrations.impulse_ml import ImpulseDatasetExporter
    
    exporter = ImpulseDatasetExporter()
    exporter.generate_classification_dataset(output_dir)
    exporter.generate_anomaly_dataset(output_dir)
"""

import json
import csv
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timezone


class ImpulseDatasetExporter:
    """
    Generates ML training datasets from Sensus scenarios
    compatible with Impulse AI platform.
    """
    
    # Health state labels for classification
    HEALTH_STATES = {
        1: "healthy_resting",
        2: "light_activity",
        3: "deep_sleep",
        4: "rem_sleep",
        5: "acute_stress",
        6: "meditation",
        7: "panic_attack",
        8: "cognitive_load",
        9: "night_terror",
        10: "tachycardia",
        11: "bradycardia",
        12: "atrial_fibrillation",
        13: "cardiac_arrest",
        14: "pvc_ectopic",
        15: "sleep_apnea",
        16: "hyperventilation",
        17: "asthma_attack",
        18: "copd_exacerbation",
        19: "stroke",
        20: "fall_detection",
        21: "seizure",
        22: "anaphylaxis",
        23: "beta_blocker_response",
        24: "opioid_sedation",
        25: "stimulant_effect",
        26: "sedative_recovery",
        27: "occupancy_change",
        28: "multi_person",
        29: "poor_air_quality",
        30: "heat_stress",
    }
    
    # Severity mapping for anomaly detection
    SEVERITY_MAP = {
        "normal": 0,
        "info": 0,
        "warning": 1,
        "critical": 2,
    }
    
    # Emergency scenarios for binary anomaly detection
    EMERGENCY_SCENARIOS = {10, 11, 12, 13, 19, 20, 21, 22}
    
    def __init__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
        from simulator import build_all_scenarios, create_engine
        self._build_scenarios = build_all_scenarios
        self._create_engine = create_engine
    
    def generate_classification_dataset(
        self,
        output_dir: str,
        speed: float = 10.0,
    ) -> Dict:
        """
        Generate a multi-class health state classification dataset.
        Each row = one time sample with features and a label.
        
        Features:
          - heart_rate, breathing_rate, hrv_rmssd, hrv_sdnn
          - spo2, gsr, motion_level, presence_score
          - stress_index (encoded), is_motion, is_present
          
        Label: health_state (string from HEALTH_STATES)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = self._build_scenarios()
        all_rows = []
        
        for sid, scenario in scenarios.items():
            engine = self._create_engine(sid, speed=speed)
            steps = int(scenario.total_duration_sec / 0.1) + 1
            label = self.HEALTH_STATES.get(sid, f"scenario_{sid}")
            
            for step_i in range(steps):
                state = engine.step(0.1)
                
                # Feature extraction
                row = {
                    "heart_rate": state.get("heart_rate", 0),
                    "breathing_rate": state.get("breathing_rate", 0),
                    "hrv_rmssd": state.get("hrv_rmssd", 0),
                    "hrv_sdnn": state.get("hrv_sdnn", 0),
                    "spo2": state.get("spo2", 98),
                    "gsr": state.get("gsr", 2.5),
                    "motion_level": round(state.get("motion_level", 0), 3),
                    "presence_score": state.get("presence_score", 0),
                    "stress_encoded": {"low": 0, "moderate": 1, "high": 2}.get(
                        state.get("stress_index", "low"), 0
                    ),
                    "is_motion": int(state.get("is_motion", False)),
                    "is_present": int(state.get("is_present", True)),
                    "blood_pressure_sys": state.get("blood_pressure_sys", 120),
                    "blood_pressure_dia": state.get("blood_pressure_dia", 80),
                    "skin_temp": state.get("skin_temp", 36.5),
                    "irregular_rhythm": int(state.get("irregular_rhythm", False)),
                    "signal_quality": state.get("signal_quality", 0),
                    # Label
                    "health_state": label,
                    "scenario_id": sid,
                    "severity": self.SEVERITY_MAP.get(state.get("alert_level", "normal"), 0),
                }
                all_rows.append(row)
        
        # Write CSV
        csv_path = os.path.join(output_dir, "health_classification_dataset.csv")
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        
        # Write dataset metadata for Impulse AI
        metadata = {
            "dataset_name": "Sensus Health State Classification",
            "version": "1.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "source": "Sensus WiFi CSI Health Sensing Platform",
            "task": "multi-class classification",
            "target_column": "health_state",
            "feature_columns": [k for k in all_rows[0].keys() if k not in ("health_state", "scenario_id", "severity")],
            "num_classes": len(self.HEALTH_STATES),
            "classes": list(self.HEALTH_STATES.values()),
            "total_samples": len(all_rows),
            "samples_per_class": self._count_per_class(all_rows),
            "description": (
                "Multi-class health state classification dataset generated from "
                "Sensus WiFi CSI sensing simulator. 30 clinical scenarios covering "
                "cardiac events, respiratory conditions, stress states, medication "
                "responses, and emergencies. Features include vital signs extracted "
                "from WiFi Channel State Information using Fresnel zone physics."
            ),
            "impulse_config": {
                "problem_type": "classification",
                "target": "health_state",
                "features": "auto",
                "train_split": 0.8,
                "eval_metric": "f1_weighted",
            },
        }
        
        meta_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "csv_path": csv_path,
            "metadata_path": meta_path,
            "total_samples": len(all_rows),
            "num_classes": len(self.HEALTH_STATES),
        }
    
    def generate_anomaly_dataset(
        self,
        output_dir: str,
        speed: float = 10.0,
    ) -> Dict:
        """
        Generate a binary anomaly detection dataset.
        Normal (0) vs. Emergency (1).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = self._build_scenarios()
        all_rows = []
        
        for sid, scenario in scenarios.items():
            engine = self._create_engine(sid, speed=speed)
            steps = int(scenario.total_duration_sec / 0.1) + 1
            is_emergency = 1 if sid in self.EMERGENCY_SCENARIOS else 0
            
            for step_i in range(steps):
                state = engine.step(0.1)
                
                # Use alert level for finer-grained labeling
                alert = state.get("alert_level", "normal")
                label = 1 if alert == "critical" or (is_emergency and alert == "warning") else 0
                
                row = {
                    "heart_rate": state.get("heart_rate", 0),
                    "breathing_rate": state.get("breathing_rate", 0),
                    "hrv_rmssd": state.get("hrv_rmssd", 0),
                    "hrv_sdnn": state.get("hrv_sdnn", 0),
                    "spo2": state.get("spo2", 98),
                    "gsr": state.get("gsr", 2.5),
                    "motion_level": round(state.get("motion_level", 0), 3),
                    "blood_pressure_sys": state.get("blood_pressure_sys", 120),
                    "skin_temp": state.get("skin_temp", 36.5),
                    "irregular_rhythm": int(state.get("irregular_rhythm", False)),
                    "is_anomaly": label,
                }
                all_rows.append(row)
        
        csv_path = os.path.join(output_dir, "anomaly_detection_dataset.csv")
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        
        normal = sum(1 for r in all_rows if r["is_anomaly"] == 0)
        anomaly = sum(1 for r in all_rows if r["is_anomaly"] == 1)
        
        metadata = {
            "dataset_name": "Sensus Health Anomaly Detection",
            "version": "1.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "task": "binary classification / anomaly detection",
            "target_column": "is_anomaly",
            "total_samples": len(all_rows),
            "normal_samples": normal,
            "anomaly_samples": anomaly,
            "anomaly_ratio": round(anomaly / len(all_rows), 3) if all_rows else 0,
            "impulse_config": {
                "problem_type": "classification",
                "target": "is_anomaly",
                "features": "auto",
                "eval_metric": "f1",
            },
        }
        
        meta_path = os.path.join(output_dir, "anomaly_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "csv_path": csv_path,
            "metadata_path": meta_path,
            "total_samples": len(all_rows),
            "normal": normal,
            "anomaly": anomaly,
        }
    
    def generate_timeseries_dataset(
        self,
        output_dir: str,
        window_size: int = 30,
        speed: float = 10.0,
    ) -> Dict:
        """
        Generate a time-series forecasting dataset.
        Predict next vital signs from a window of previous readings.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = self._build_scenarios()
        all_rows = []
        
        for sid in [1, 5, 7, 10, 13, 15, 21]:  # Representative subset
            engine = self._create_engine(sid, speed=speed)
            steps = int(scenarios[sid].total_duration_sec / 0.1) + 1
            buffer = []
            
            for step_i in range(steps):
                state = engine.step(0.1)
                buffer.append({
                    "hr": state.get("heart_rate", 0),
                    "br": state.get("breathing_rate", 0),
                    "hrv": state.get("hrv_rmssd", 0),
                    "spo2": state.get("spo2", 98),
                    "motion": round(state.get("motion_level", 0), 3),
                })
                
                if len(buffer) > window_size:
                    window = buffer[-window_size-1:-1]
                    target = buffer[-1]
                    
                    row = {}
                    for i, w in enumerate(window):
                        for k, v in w.items():
                            row[f"{k}_t{i}"] = v
                    row["target_hr"] = target["hr"]
                    row["target_br"] = target["br"]
                    row["scenario"] = self.HEALTH_STATES.get(sid, f"s{sid}")
                    all_rows.append(row)
        
        csv_path = os.path.join(output_dir, "timeseries_dataset.csv")
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        
        return {
            "csv_path": csv_path,
            "total_samples": len(all_rows),
            "window_size": window_size,
            "features_per_step": 5,
        }
    
    def _count_per_class(self, rows: List[Dict]) -> Dict[str, int]:
        counts = {}
        for r in rows:
            label = r.get("health_state", "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("IMPULSE AI INTEGRATION — SELF TEST")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'impulse_ai')
    exporter = ImpulseDatasetExporter()
    
    print("\n1. Classification dataset...")
    result = exporter.generate_classification_dataset(output_dir, speed=50.0)
    print(f"   {result['total_samples']} samples, {result['num_classes']} classes")
    print(f"   → {result['csv_path']}")
    
    print("\n2. Anomaly detection dataset...")
    result = exporter.generate_anomaly_dataset(output_dir, speed=50.0)
    print(f"   {result['total_samples']} samples ({result['normal']} normal, {result['anomaly']} anomaly)")
    print(f"   → {result['csv_path']}")
    
    print("\n3. Time-series dataset...")
    result = exporter.generate_timeseries_dataset(output_dir, speed=50.0)
    print(f"   {result['total_samples']} samples, window={result['window_size']}")
    print(f"   → {result['csv_path']}")
    
    print(f"\nAll datasets exported to: {output_dir}")
    print("\n✅ Impulse AI integration test passed")
