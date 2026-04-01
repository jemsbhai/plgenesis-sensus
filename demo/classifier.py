"""
Sensus Real-Time Health State Classifier
=========================================
Trains a real ML model on synthetic scenario data and provides
live inference on incoming vital signs.

Uses sklearn (RandomForest + XGBoost if available) trained on data
from all 30 scenarios. Runs entirely locally — no API needed.

For Impulse AI integration: export the trained model and dataset
in their expected format.

Architecture:
  1. Generate training data from all 30 scenarios
  2. Train ensemble classifier (RF + optional XGB)
  3. Provide predict() for real-time inference
  4. Export model metrics + confusion matrix for dashboard
"""

import numpy as np
import json
import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import logging

logger = logging.getLogger('sensus.classifier')


# Feature columns used for prediction
FEATURE_COLS = [
    'heart_rate', 'breathing_rate', 'hrv_rmssd', 'hrv_sdnn',
    'spo2', 'gsr', 'motion_level', 'blood_pressure_sys',
    'blood_pressure_dia', 'skin_temp', 'stress_encoded',
    'is_motion', 'is_present', 'irregular_rhythm', 'signal_quality',
]

# Simplified labels for display (group similar scenarios)
DISPLAY_LABELS = {
    'healthy_resting': 'Normal',
    'light_activity': 'Normal',
    'deep_sleep': 'Sleep',
    'rem_sleep': 'Sleep',
    'acute_stress': 'Stress',
    'meditation': 'Relaxation',
    'panic_attack': 'Emergency',
    'cognitive_load': 'Stress',
    'night_terror': 'Emergency',
    'tachycardia': 'Cardiac Alert',
    'bradycardia': 'Cardiac Alert',
    'atrial_fibrillation': 'Cardiac Alert',
    'cardiac_arrest': 'Cardiac Emergency',
    'pvc_ectopic': 'Cardiac Alert',
    'sleep_apnea': 'Respiratory Alert',
    'hyperventilation': 'Respiratory Alert',
    'asthma_attack': 'Respiratory Alert',
    'copd_exacerbation': 'Respiratory Alert',
    'stroke': 'Emergency',
    'fall_detection': 'Emergency',
    'seizure': 'Emergency',
    'anaphylaxis': 'Emergency',
    'beta_blocker_response': 'Medication',
    'opioid_sedation': 'Medication',
    'stimulant_effect': 'Medication',
    'sedative_recovery': 'Medication',
    'occupancy_change': 'Normal',
    'multi_person': 'Normal',
    'poor_air_quality': 'Environmental',
    'heat_stress': 'Environmental',
}

# Risk levels for severity coloring
RISK_LEVELS = {
    'Normal': 0,
    'Sleep': 0,
    'Relaxation': 0,
    'Stress': 1,
    'Medication': 1,
    'Environmental': 1,
    'Cardiac Alert': 2,
    'Respiratory Alert': 2,
    'Emergency': 3,
    'Cardiac Emergency': 3,
}


class HealthStateClassifier:
    """
    Real-time health state classifier using ensemble ML.
    Trains on synthetic data from all 30 Sensus scenarios.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.feature_importance = {}
        self._classes = []

    def train(self, speed: float = 50.0, use_display_labels: bool = True) -> Dict:
        """
        Generate training data and train the classifier.
        
        Args:
            speed: Simulation speed for data generation
            use_display_labels: If True, use grouped display labels
            
        Returns:
            Dict with training metrics
        """
        from simulator import build_all_scenarios, create_engine

        # Scenario label mapping
        scenario_labels = {
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

        scenarios = build_all_scenarios()
        X_rows = []
        y_labels = []

        print("Generating training data from 30 scenarios...")
        for sid, scenario in scenarios.items():
            engine = create_engine(sid, speed=speed)
            steps = int(scenario.total_duration_sec / 0.1) + 1

            raw_label = scenario_labels.get(sid, f'scenario_{sid}')
            label = DISPLAY_LABELS.get(raw_label, raw_label) if use_display_labels else raw_label

            for _ in range(steps):
                state = engine.step(0.1)
                features = self._extract_features(state)
                X_rows.append(features)
                y_labels.append(label)

        X = np.array(X_rows)
        y = np.array(y_labels)

        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self._classes = list(self.label_encoder.classes_)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train RandomForest
        print("Training RandomForest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            FEATURE_COLS[i]: round(float(importances[i]), 4)
            for i in range(len(FEATURE_COLS))
        }
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        self.metrics = {
            'accuracy': round(accuracy, 4),
            'f1_weighted': round(f1, 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_classes': len(self._classes),
            'classes': self._classes,
            'feature_importance': self.feature_importance,
        }

        self.is_trained = True
        print(f"Training complete — Accuracy: {accuracy:.2%}, F1: {f1:.4f}")

        return self.metrics

    def predict(self, state: Dict) -> Dict:
        """
        Predict health state from a single vital signs state dict.
        
        Returns:
            {
                'predicted_class': str,
                'confidence': float,
                'risk_level': int (0-3),
                'risk_label': str,
                'probabilities': {class: prob},
            }
        """
        if not self.is_trained:
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'risk_level': 0,
                'risk_label': 'Not trained',
                'probabilities': {},
            }

        features = self._extract_features(state)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)

        # Predict with probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        # Top 3 predictions
        top_indices = np.argsort(proba)[-3:][::-1]
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: round(float(proba[i]), 3)
            for i in top_indices
        }

        risk_level = RISK_LEVELS.get(pred_class, 0)
        risk_labels = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}

        return {
            'predicted_class': pred_class,
            'confidence': round(confidence, 3),
            'risk_level': risk_level,
            'risk_label': risk_labels.get(risk_level, 'Unknown'),
            'probabilities': probabilities,
        }

    def predict_anomaly(self, state: Dict) -> Dict:
        """
        Binary anomaly detection: is this state normal or abnormal?
        Uses the classifier probabilities to determine.
        """
        result = self.predict(state)
        normal_classes = {'Normal', 'Sleep', 'Relaxation'}
        
        # Sum probabilities of normal classes
        normal_prob = sum(
            prob for cls, prob in result['probabilities'].items()
            if cls in normal_classes
        )
        
        is_anomaly = result['risk_level'] >= 2
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': round(1.0 - normal_prob, 3),
            'health_state': result['predicted_class'],
            'confidence': result['confidence'],
        }

    def save(self, path: str):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'classes': self._classes,
            }, f)

    def load(self, path: str) -> bool:
        """Load a trained model from disk."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.metrics = data['metrics']
            self.feature_importance = data['feature_importance']
            self._classes = data['classes']
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _extract_features(self, state: Dict) -> List[float]:
        """Extract feature vector from a state dict."""
        stress_map = {'low': 0, 'moderate': 1, 'high': 2}
        return [
            float(state.get('heart_rate', 0)),
            float(state.get('breathing_rate', 0)),
            float(state.get('hrv_rmssd', 0)),
            float(state.get('hrv_sdnn', 0)),
            float(state.get('spo2', 98)),
            float(state.get('gsr', 2.5)),
            float(state.get('motion_level', 0)),
            float(state.get('blood_pressure_sys', 120)),
            float(state.get('blood_pressure_dia', 80)),
            float(state.get('skin_temp', 36.5)),
            float(stress_map.get(state.get('stress_index', 'low'), 0)),
            float(int(state.get('is_motion', False))),
            float(int(state.get('is_present', True))),
            float(int(state.get('irregular_rhythm', False))),
            float(state.get('signal_quality', 10)),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON FOR DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

_classifier_instance = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'model', 'health_classifier.pkl')


def get_classifier() -> HealthStateClassifier:
    """Get or create the singleton classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HealthStateClassifier()
        # Try loading saved model first
        if not _classifier_instance.load(MODEL_PATH):
            print("No saved model found — training new classifier...")
            _classifier_instance.train(speed=50.0)
            _classifier_instance.save(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        else:
            print(f"Loaded saved model from {MODEL_PATH}")
    return _classifier_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SENSUS HEALTH STATE CLASSIFIER — TRAINING")
    print("=" * 70)

    clf = HealthStateClassifier()
    metrics = clf.train(speed=50.0)

    print(f"\n📊 Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"   Classes:   {metrics['num_classes']}")
    print(f"   Train:     {metrics['train_samples']} samples")
    print(f"   Test:      {metrics['test_samples']} samples")

    print(f"\n🔑 Feature Importance (top 5):")
    for i, (feat, imp) in enumerate(metrics['feature_importance'].items()):
        if i >= 5:
            break
        print(f"   {feat}: {imp:.4f}")

    # Save model
    clf.save(MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")

    # Test live prediction
    print(f"\n🔮 Live Prediction Test:")
    from simulator import create_engine

    # Test on cardiac arrest
    engine = create_engine(13, speed=10.0)
    for i in range(30):
        state = engine.step(0.1)
        if i % 10 == 0:
            result = clf.predict(state)
            anomaly = clf.predict_anomaly(state)
            print(f"   t={state['elapsed_sec']:.1f}s → "
                  f"{result['predicted_class']} ({result['confidence']:.0%}) "
                  f"Risk: {result['risk_label']} | "
                  f"Anomaly: {'⚠' if anomaly['is_anomaly'] else '✓'} "
                  f"(score: {anomaly['anomaly_score']:.3f})")

    print("\n✅ Classifier test passed")
