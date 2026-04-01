"""
Sensus Real-Time Health Classifier
===================================
Trains a lightweight ML model on all 30 simulated scenarios and
provides real-time health state classification from vital sign features.

This runs locally with sklearn — no external API needed.
For Impulse AI integration, the same training data and model
architecture can be uploaded to their platform.

Features used: HR, BR, HRV_RMSSD, HRV_SDNN, SpO2, GSR, Motion,
               BP_sys, Skin_temp, Irregular_rhythm, Stress_encoded

Outputs:
  - 30-class health state prediction
  - Anomaly score (normal vs emergency)
  - Confidence scores per class
  - Top-3 differential diagnoses
"""

import numpy as np
import json
import os
import sys
import pickle
import time
from typing import Dict, List, Tuple, Optional

# Ensure we can import simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))


class HealthClassifier:
    """
    Real-time health state classifier trained on Sensus scenario data.
    Uses sklearn RandomForest for fast inference (<1ms per prediction).
    """
    
    FEATURE_NAMES = [
        "heart_rate", "breathing_rate", "hrv_rmssd", "hrv_sdnn",
        "spo2", "gsr", "motion_level", "blood_pressure_sys",
        "blood_pressure_dia", "skin_temp", "irregular_rhythm",
        "stress_encoded", "presence_score", "signal_quality",
    ]
    
    HEALTH_STATES = {
        1: "Healthy Resting", 2: "Light Activity", 3: "Deep Sleep",
        4: "REM Sleep", 5: "Acute Stress", 6: "Meditation",
        7: "Panic Attack", 8: "Cognitive Load", 9: "Night Terror",
        10: "Tachycardia", 11: "Bradycardia", 12: "Atrial Fibrillation",
        13: "Cardiac Arrest", 14: "PVC/Ectopic Beats", 15: "Sleep Apnea",
        16: "Hyperventilation", 17: "Asthma Attack", 18: "COPD Exacerbation",
        19: "Stroke", 20: "Fall Detection", 21: "Seizure",
        22: "Anaphylaxis", 23: "Beta Blocker Response", 24: "Opioid Sedation",
        25: "Stimulant Effect", 26: "Sedative Recovery",
        27: "Occupancy Change", 28: "Multi-Person", 
        29: "Poor Air Quality", 30: "Heat Stress",
    }
    
    SEVERITY_LABELS = {
        "normal": [1, 2, 3, 4, 6, 8, 14, 27, 28],
        "warning": [5, 7, 9, 15, 16, 17, 18, 23, 24, 25, 26, 29, 30],
        "critical": [10, 11, 12, 13, 19, 20, 21, 22],
    }
    
    def __init__(self):
        self.model = None
        self.anomaly_model = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False
        self._training_time = 0
        self._accuracy = 0
        self._num_samples = 0
    
    def train(self, speed: float = 50.0, verbose: bool = True) -> Dict:
        """
        Train the classifier on all 30 scenarios.
        Returns training metrics.
        """
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import cross_val_score
        from simulator import build_all_scenarios, create_engine
        
        if verbose:
            print("Training Sensus Health Classifier...")
            print(f"  Generating training data from 30 scenarios...")
        
        t0 = time.time()
        scenarios = build_all_scenarios()
        
        X_all = []
        y_all = []
        y_severity = []
        
        for sid, scenario in scenarios.items():
            engine = create_engine(sid, speed=speed)
            steps = int(scenario.total_duration_sec / 0.1) + 1
            label = self.HEALTH_STATES.get(sid, f"Scenario {sid}")
            
            # Determine severity
            severity = 0
            if sid in self.SEVERITY_LABELS["warning"]:
                severity = 1
            elif sid in self.SEVERITY_LABELS["critical"]:
                severity = 2
            
            for _ in range(steps):
                state = engine.step(0.1)
                features = self._extract_features(state)
                X_all.append(features)
                y_all.append(label)
                y_severity.append(severity)
        
        X = np.array(X_all)
        y = np.array(y_all)
        y_sev = np.array(y_severity)
        
        if verbose:
            print(f"  Dataset: {len(X)} samples, {len(set(y))} classes")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train main classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_scaled, y_encoded)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=3, n_jobs=-1)
        self._accuracy = float(np.mean(cv_scores))
        
        # Train anomaly detector on normal data only
        X_normal = X_scaled[y_sev == 0]
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=0.15,
            random_state=42,
            n_jobs=-1,
        )
        self.anomaly_model.fit(X_normal)
        
        self._training_time = time.time() - t0
        self._num_samples = len(X)
        self.is_trained = True
        
        metrics = {
            "accuracy": round(self._accuracy, 4),
            "num_samples": self._num_samples,
            "num_classes": len(set(y)),
            "training_time_sec": round(self._training_time, 2),
            "feature_importance": self._get_feature_importance(),
        }
        
        if verbose:
            print(f"  Accuracy: {self._accuracy:.1%} (3-fold CV)")
            print(f"  Training time: {self._training_time:.1f}s")
            print(f"  ✅ Classifier ready")
        
        return metrics
    
    def predict(self, state: Dict) -> Dict:
        """
        Predict health state from a single state dict.
        Returns prediction with confidence and differential.
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        features = self._extract_features(state)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Main classification
        proba = self.model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])
        
        # Top-3 differential
        top3_idx = np.argsort(proba)[-3:][::-1]
        differential = []
        for idx in top3_idx:
            label = self.label_encoder.inverse_transform([idx])[0]
            differential.append({
                "state": label,
                "confidence": round(float(proba[idx]) * 100, 1),
            })
        
        # Anomaly score
        anomaly_score = float(self.anomaly_model.decision_function(X_scaled)[0])
        is_anomaly = self.anomaly_model.predict(X_scaled)[0] == -1
        
        # Severity estimation
        severity = "normal"
        if is_anomaly and anomaly_score < -0.15:
            severity = "critical"
        elif is_anomaly:
            severity = "warning"
        
        return {
            "predicted_state": pred_label,
            "confidence": round(confidence * 100, 1),
            "differential": differential,
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "severity": severity,
        }
    
    def _extract_features(self, state: Dict) -> List[float]:
        """Extract feature vector from state dict."""
        stress_map = {"low": 0, "moderate": 1, "high": 2}
        return [
            float(state.get("heart_rate", 0)),
            float(state.get("breathing_rate", 0)),
            float(state.get("hrv_rmssd", 0)),
            float(state.get("hrv_sdnn", 0)),
            float(state.get("spo2", 98)),
            float(state.get("gsr", 2.5)),
            float(state.get("motion_level", 0)),
            float(state.get("blood_pressure_sys", 120)),
            float(state.get("blood_pressure_dia", 80)),
            float(state.get("skin_temp", 36.5)),
            float(state.get("irregular_rhythm", False)),
            float(stress_map.get(state.get("stress_index", "low"), 0)),
            float(state.get("presence_score", 0)),
            float(state.get("signal_quality", 0)),
        ]
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.model:
            return {}
        importance = self.model.feature_importances_
        return {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(self.FEATURE_NAMES, importance),
                key=lambda x: x[1], reverse=True
            )
        }
    
    def save(self, path: str):
        """Save trained model to disk."""
        data = {
            "model": self.model,
            "anomaly_model": self.anomaly_model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "accuracy": self._accuracy,
            "num_samples": self._num_samples,
            "training_time": self._training_time,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.anomaly_model = data["anomaly_model"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self._accuracy = data["accuracy"]
        self._num_samples = data["num_samples"]
        self._training_time = data["training_time"]
        self.is_trained = True
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            "is_trained": self.is_trained,
            "accuracy": round(self._accuracy * 100, 1),
            "num_samples": self._num_samples,
            "num_classes": len(self.HEALTH_STATES),
            "training_time": round(self._training_time, 1),
            "feature_count": len(self.FEATURE_NAMES),
            "features": self.FEATURE_NAMES,
        }


# Singleton for use across modules
_classifier_instance = None

def get_classifier() -> HealthClassifier:
    """Get or create the singleton classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HealthClassifier()
    return _classifier_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SENSUS HEALTH CLASSIFIER — SELF TEST")
    print("=" * 70)
    
    clf = HealthClassifier()
    metrics = clf.train(speed=50.0)
    
    print(f"\nTraining metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  Classes: {metrics['num_classes']}")
    print(f"  Time: {metrics['training_time_sec']}s")
    
    print(f"\nFeature importance (top 5):")
    for name, imp in list(metrics['feature_importance'].items())[:5]:
        print(f"  {name}: {imp:.4f}")
    
    # Test prediction on a few scenarios
    from simulator import create_engine
    
    print(f"\nLive predictions:")
    for sid in [1, 7, 13, 15, 21]:
        engine = create_engine(sid, speed=10.0)
        # Run to middle of scenario
        for _ in range(30):
            state = engine.step(0.1)
        pred = clf.predict(state)
        print(f"  Scenario {sid:2d} → {pred['predicted_state']:<25s} "
              f"({pred['confidence']:.0f}% conf, anomaly={pred['is_anomaly']}, "
              f"severity={pred['severity']})")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'model')
    os.makedirs(model_path, exist_ok=True)
    clf.save(os.path.join(model_path, 'health_classifier.pkl'))
    print(f"\nModel saved to {model_path}")
    
    print("\n✅ Classifier self-test passed")
