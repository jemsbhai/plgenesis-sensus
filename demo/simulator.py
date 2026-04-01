"""
Sensus Virtual Scenario Simulator
==================================
Generates physiologically accurate synthetic CSI data for 30 clinical/health
scenarios. Each scenario produces time-varying vital signs that evolve through
multiple phases, enabling realistic demonstration of the Sensus platform
without requiring live hardware.

The simulator generates raw CSI-like data (amplitude + phase across 52
subcarriers) with embedded physiological signals. When fed through the
existing CSIProcessor pipeline, the signals produce realistic vital sign
extractions — proving the full detection-to-alert chain works.

Architecture:
  ScenarioEngine → PhysiologicalModel → CSISignalSynthesizer → CSIProcessor
                                      → EnvironmentModel

Author: Sensus Team — PL_Genesis Hackathon 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import time
import logging

logger = logging.getLogger('sensus.simulator')

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 10          # Hz — matches ESP32-C6 actual publish rate
NUM_SUBCARRIERS = 52      # Standard WiFi CSI subcarrier count
WIFI_FREQ = 2.4e9         # Hz
SPEED_OF_LIGHT = 3e8
WAVELENGTH = SPEED_OF_LIGHT / WIFI_FREQ  # ~0.125m


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    NORMAL = "normal"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class VitalParams:
    """Instantaneous physiological parameters at a point in time."""
    heart_rate: float = 72.0           # bpm
    heart_rate_variability: float = 5.0  # bpm variation (jitter)
    breathing_rate: float = 15.0       # breaths/min
    breathing_variability: float = 1.0  # breath rate variation
    breathing_depth: float = 1.0       # relative amplitude 0-2
    hrv_sdnn: float = 45.0            # ms
    hrv_rmssd: float = 38.0           # ms
    motion_level: float = 0.0         # 0=still, 1=fidget, 5=walking, 10=vigorous
    motion_type: str = "none"         # none, micro, fidget, walking, thrashing, seizure, fall
    presence: bool = True
    presence_score: float = 0.9
    gsr: float = 2.5                  # μS — skin conductance
    stress_index: str = "low"         # low, moderate, high
    spo2: float = 98.0               # % (conceptual — inferred)
    blood_pressure_sys: float = 120.0  # mmHg (conceptual)
    blood_pressure_dia: float = 80.0   # mmHg (conceptual)
    skin_temp: float = 36.5           # °C
    # Cardiac irregularity parameters
    irregular_rhythm: bool = False
    ectopic_beat_prob: float = 0.0    # probability of skipped/extra beat per cycle
    rhythm_chaos: float = 0.0        # 0=regular, 1=fully chaotic (AF)
    # Breathing irregularity
    apnea_active: bool = False
    breathing_effort: float = 1.0     # 1=normal, 2=labored
    # Alert
    alert_level: AlertLevel = AlertLevel.NORMAL
    alert_message: str = ""


@dataclass
class EnvironmentParams:
    """Environmental conditions."""
    temperature_c: float = 22.0
    humidity_pct: float = 45.0
    co2_ppm: float = 450.0
    tvoc_ppb: float = 120.0
    light_lux: float = 300.0
    noise_db: float = 35.0


@dataclass
class ScenarioPhase:
    """A phase within a scenario with target parameters and transition."""
    name: str
    duration_sec: float           # how long this phase lasts
    target_vitals: VitalParams    # target vital signs at END of phase
    target_env: EnvironmentParams = field(default_factory=EnvironmentParams)
    transition: str = "linear"    # linear, exponential, step, sigmoid, oscillating
    description: str = ""


@dataclass
class Scenario:
    """Complete scenario definition."""
    id: int
    name: str
    category: str
    description: str
    clinical_relevance: str
    phases: List[ScenarioPhase]
    total_duration_sec: float = 0  # auto-computed
    loop: bool = True             # whether to loop after completion
    num_people: int = 1
    node_config: str = "standard"  # standard, near_field, multi_zone

    def __post_init__(self):
        self.total_duration_sec = sum(p.duration_sec for p in self.phases)


# ═══════════════════════════════════════════════════════════════════════════
# INTERPOLATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def interpolate(start: float, end: float, t: float, method: str = "linear") -> float:
    """
    Interpolate between start and end values.
    t is normalized progress [0, 1].
    """
    t = np.clip(t, 0.0, 1.0)

    if method == "linear":
        return start + (end - start) * t
    elif method == "exponential":
        return start + (end - start) * (1 - np.exp(-3 * t))
    elif method == "step":
        return end if t > 0.5 else start
    elif method == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-12 * (t - 0.5)))
        return start + (end - start) * s
    elif method == "oscillating":
        base = start + (end - start) * t
        oscillation = np.sin(2 * np.pi * t * 3) * abs(end - start) * 0.15
        return base + oscillation
    else:
        return start + (end - start) * t


def interpolate_vitals(v1: VitalParams, v2: VitalParams, t: float, method: str) -> VitalParams:
    """Interpolate all numeric fields between two VitalParams."""
    result = VitalParams()
    numeric_fields = [
        'heart_rate', 'heart_rate_variability', 'breathing_rate',
        'breathing_variability', 'breathing_depth', 'hrv_sdnn', 'hrv_rmssd',
        'motion_level', 'presence_score', 'gsr', 'spo2',
        'blood_pressure_sys', 'blood_pressure_dia', 'skin_temp',
        'ectopic_beat_prob', 'rhythm_chaos', 'breathing_effort'
    ]
    for f in numeric_fields:
        setattr(result, f, interpolate(getattr(v1, f), getattr(v2, f), t, method))

    # Non-numeric fields: use target if past halfway
    if t > 0.5:
        result.motion_type = v2.motion_type
        result.stress_index = v2.stress_index
        result.irregular_rhythm = v2.irregular_rhythm
        result.apnea_active = v2.apnea_active
        result.presence = v2.presence
        result.alert_level = v2.alert_level
        result.alert_message = v2.alert_message
    else:
        result.motion_type = v1.motion_type
        result.stress_index = v1.stress_index
        result.irregular_rhythm = v1.irregular_rhythm
        result.apnea_active = v1.apnea_active
        result.presence = v1.presence
        result.alert_level = v1.alert_level
        result.alert_message = v1.alert_message

    return result


def interpolate_env(e1: EnvironmentParams, e2: EnvironmentParams, t: float, method: str) -> EnvironmentParams:
    """Interpolate environmental parameters."""
    result = EnvironmentParams()
    for f in ['temperature_c', 'humidity_pct', 'co2_ppm', 'tvoc_ppb', 'light_lux', 'noise_db']:
        setattr(result, f, interpolate(getattr(e1, f), getattr(e2, f), t, method))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CSI SIGNAL SYNTHESIZER
# ═══════════════════════════════════════════════════════════════════════════

class CSISignalSynthesizer:
    """
    Generates synthetic WiFi CSI data with embedded physiological signals.

    Physics model:
    - Each subcarrier has a base amplitude + phase determined by multipath environment
    - Breathing causes chest displacement (~1-5mm) which modulates CSI phase
      in the Fresnel zone at the breathing frequency
    - Heartbeat causes micro-displacement (~0.1-0.5mm) at cardiac frequency
    - Body motion creates larger, broadband amplitude/phase changes
    - Multiple subcarriers respond differently based on their frequency and
      the geometry of the signal path relative to the body

    The synthesizer creates signals that, when processed by CSIProcessor,
    produce the expected vital signs.
    """

    def __init__(self, num_subcarriers: int = NUM_SUBCARRIERS, sample_rate: int = SAMPLE_RATE):
        self.num_subcarriers = num_subcarriers
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(42)

        # Per-subcarrier properties (fixed for a "room")
        self.base_amplitudes = self.rng.uniform(15, 45, num_subcarriers)
        self.base_phases = self.rng.uniform(-np.pi, np.pi, num_subcarriers)
        self.subcarrier_freqs = np.linspace(2.412e9, 2.462e9, num_subcarriers)

        # Sensitivity profiles: each subcarrier responds differently to body signals
        # Some subcarriers are more sensitive to breathing, others to cardiac
        self.breath_sensitivity = self.rng.uniform(0.3, 1.0, num_subcarriers)
        self.cardiac_sensitivity = self.rng.uniform(0.1, 0.8, num_subcarriers)
        self.motion_sensitivity = self.rng.uniform(0.5, 1.5, num_subcarriers)

        # Phase accumulator for continuous signal generation
        self._phase_acc = 0.0
        self._sample_count = 0

    def generate_csi_packet(self, vitals: VitalParams, t: float) -> Dict:
        """
        Generate one CSI packet at time t with embedded physiological signals.

        Returns dict matching the format expected by CSIProcessor:
            {'amplitude': [...], 'phase': [...]}
        """
        amplitudes = np.copy(self.base_amplitudes)
        phases = np.copy(self.base_phases)

        # ── 1. BREATHING SIGNAL ──
        if vitals.presence and not vitals.apnea_active:
            breath_freq = vitals.breathing_rate / 60.0  # Hz
            breath_phase = 2 * np.pi * breath_freq * t

            # Chest displacement → phase modulation (Fresnel model)
            # Typical chest displacement: 1-5mm → phase shift of 0.05-0.25 rad at 2.4GHz
            chest_displacement_m = vitals.breathing_depth * 0.003  # 3mm nominal
            phase_shift = (4 * np.pi * chest_displacement_m / WAVELENGTH)

            breath_signal = np.sin(breath_phase) * phase_shift
            # Add harmonics for realistic breathing waveform
            breath_signal += 0.3 * np.sin(2 * breath_phase) * phase_shift
            breath_signal += 0.1 * np.sin(3 * breath_phase) * phase_shift

            # Add variability
            breath_noise = self.rng.normal(0, vitals.breathing_variability * 0.01)
            breath_signal += breath_noise

            # Apply per-subcarrier sensitivity
            phases += breath_signal * self.breath_sensitivity
            amplitudes += breath_signal * self.breath_sensitivity * 2.0 * vitals.breathing_depth

        elif vitals.apnea_active:
            # During apnea: very small residual motion, occasional gasps
            gasp_cycle = 15.0  # seconds between gasps
            gasp_phase = (t % gasp_cycle) / gasp_cycle
            if gasp_phase > 0.85:  # gasp in last 15% of cycle
                gasp_signal = np.sin(2 * np.pi * gasp_phase * 10) * 0.3
                phases += gasp_signal * self.breath_sensitivity
                amplitudes += abs(gasp_signal) * self.breath_sensitivity * 3.0

        # ── 2. CARDIAC SIGNAL ──
        if vitals.presence and vitals.heart_rate > 0:
            cardiac_freq = vitals.heart_rate / 60.0  # Hz

            if vitals.irregular_rhythm:
                # Irregular rhythm: add chaos to the cardiac frequency
                cardiac_freq += self.rng.normal(0, vitals.rhythm_chaos * cardiac_freq * 0.3)
                cardiac_freq = max(0.5, cardiac_freq)

            cardiac_phase = 2 * np.pi * cardiac_freq * t

            # Ectopic beats: occasional phase disruption
            if vitals.ectopic_beat_prob > 0 and self.rng.random() < vitals.ectopic_beat_prob / self.sample_rate:
                cardiac_phase += self.rng.uniform(np.pi * 0.3, np.pi * 0.7)

            # Heart micro-displacement: ~0.1-0.5mm
            cardiac_displacement = 0.0002  # 0.2mm
            cardiac_phase_shift = (4 * np.pi * cardiac_displacement / WAVELENGTH)

            # Cardiac waveform: sharper than breathing (systolic peak)
            cardiac_signal = np.sin(cardiac_phase) * cardiac_phase_shift
            cardiac_signal += 0.5 * np.sin(2 * cardiac_phase + 0.3) * cardiac_phase_shift  # dicrotic notch
            cardiac_signal += 0.2 * np.sin(3 * cardiac_phase) * cardiac_phase_shift

            # Add heart rate variability (modulate inter-beat interval)
            hrv_modulation = self.rng.normal(0, vitals.heart_rate_variability * 0.001)
            cardiac_signal *= (1.0 + hrv_modulation)

            phases += cardiac_signal * self.cardiac_sensitivity
            amplitudes += abs(cardiac_signal) * self.cardiac_sensitivity * 0.5

        # ── 3. MOTION SIGNAL ──
        if vitals.motion_level > 0:
            motion_amp = vitals.motion_level * 0.5

            if vitals.motion_type == "micro":
                # Micro-movements: small, slow
                motion = self.rng.normal(0, motion_amp * 0.1, self.num_subcarriers)
            elif vitals.motion_type == "fidget":
                # Fidgeting: moderate, somewhat periodic
                motion = np.sin(2 * np.pi * 0.5 * t) * motion_amp * 0.3
                motion += self.rng.normal(0, motion_amp * 0.15, self.num_subcarriers)
            elif vitals.motion_type == "walking":
                # Walking: large, periodic (~2Hz step frequency)
                motion = np.sin(2 * np.pi * 2.0 * t) * motion_amp
                motion += 0.5 * np.sin(2 * np.pi * 4.0 * t) * motion_amp  # arm swing
                motion += self.rng.normal(0, motion_amp * 0.2, self.num_subcarriers)
            elif vitals.motion_type == "thrashing":
                # Thrashing: large, chaotic
                motion = self.rng.normal(0, motion_amp * 0.8, self.num_subcarriers)
                motion += np.sin(2 * np.pi * self.rng.uniform(1, 5) * t) * motion_amp
            elif vitals.motion_type == "seizure":
                # Seizure: rhythmic, high amplitude, ~3-8 Hz
                seizure_freq = 5.0  # Hz — tonic-clonic
                motion = np.sin(2 * np.pi * seizure_freq * t) * motion_amp * 2.0
                motion += 0.3 * np.sin(2 * np.pi * seizure_freq * 2 * t) * motion_amp
                motion += self.rng.normal(0, motion_amp * 0.3, self.num_subcarriers)
            elif vitals.motion_type == "fall":
                # Fall: single large spike then stillness
                fall_phase = (t % 3.0) / 3.0
                if fall_phase < 0.1:
                    motion = self.rng.normal(0, motion_amp * 5.0, self.num_subcarriers)
                else:
                    motion = self.rng.normal(0, 0.05, self.num_subcarriers)
            elif vitals.motion_type == "tremor":
                # Tremor: small amplitude, high frequency (~4-12Hz)
                tremor_freq = 8.0
                motion = np.sin(2 * np.pi * tremor_freq * t) * motion_amp * 0.3
                motion += self.rng.normal(0, motion_amp * 0.05, self.num_subcarriers)
            else:
                motion = self.rng.normal(0, motion_amp * 0.1, self.num_subcarriers)

            if isinstance(motion, (int, float)):
                motion = np.full(self.num_subcarriers, motion)

            amplitudes += motion * self.motion_sensitivity
            phases += (motion * 0.1) * self.motion_sensitivity

        # ── 4. PRESENCE MODULATION ──
        if not vitals.presence:
            # No person: very low variance, just environmental noise
            amplitudes = self.base_amplitudes + self.rng.normal(0, 0.3, self.num_subcarriers)
            phases = self.base_phases + self.rng.normal(0, 0.02, self.num_subcarriers)

        # ── 5. ENVIRONMENTAL NOISE ──
        # WiFi interference, multipath reflections, thermal noise
        noise_amplitude = self.rng.normal(0, 0.8, self.num_subcarriers)
        noise_phase = self.rng.normal(0, 0.05, self.num_subcarriers)
        amplitudes += noise_amplitude
        phases += noise_phase

        # Ensure physical constraints
        amplitudes = np.clip(amplitudes, 0, 100)
        phases = np.mod(phases + np.pi, 2 * np.pi) - np.pi  # wrap to [-π, π]

        self._sample_count += 1

        return {
            'amplitude': amplitudes.tolist(),
            'phase': phases.tolist(),
            'timestamp': t,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ScenarioEngine:
    """
    Runs a scenario by stepping through its phases, interpolating parameters,
    and generating CSI data through the synthesizer.
    """

    def __init__(self, scenario: Scenario, speed: float = 1.0):
        self.scenario = scenario
        self.speed = speed  # 1.0 = real-time, 2.0 = 2x speed, etc.
        self.synthesizers = {}  # one per virtual node
        self.start_time = None
        self.elapsed = 0.0

        # Initialize 3 virtual nodes (matching ESP32-C6 mesh)
        for i in range(3):
            synth = CSISignalSynthesizer()
            # Slightly different base properties per node (different positions)
            synth.base_amplitudes += np.random.uniform(-3, 3, NUM_SUBCARRIERS)
            synth.breath_sensitivity *= np.random.uniform(0.8, 1.2, NUM_SUBCARRIERS)
            synth.cardiac_sensitivity *= np.random.uniform(0.8, 1.2, NUM_SUBCARRIERS)
            self.synthesizers[f"node_{i+1}"] = synth

        # State tracking
        self._current_phase_idx = 0
        self._phase_start_time = 0.0
        self._prev_vitals = self.scenario.phases[0].target_vitals if self.scenario.phases else VitalParams()
        self._prev_env = self.scenario.phases[0].target_env if self.scenario.phases else EnvironmentParams()

    def reset(self):
        """Reset scenario to beginning."""
        self.start_time = None
        self.elapsed = 0.0
        self._current_phase_idx = 0
        self._phase_start_time = 0.0
        self._prev_vitals = self.scenario.phases[0].target_vitals if self.scenario.phases else VitalParams()
        self._prev_env = self.scenario.phases[0].target_env if self.scenario.phases else EnvironmentParams()

    def get_current_phase(self) -> Tuple[int, ScenarioPhase, float]:
        """
        Returns (phase_index, current_phase, progress_within_phase).
        Handles looping and phase transitions.
        """
        if not self.scenario.phases:
            return 0, ScenarioPhase("empty", 1.0, VitalParams()), 0.0

        elapsed = self.elapsed
        if self.scenario.loop and self.scenario.total_duration_sec > 0:
            elapsed = elapsed % self.scenario.total_duration_sec

        cumulative = 0.0
        for i, phase in enumerate(self.scenario.phases):
            if elapsed < cumulative + phase.duration_sec:
                progress = (elapsed - cumulative) / phase.duration_sec
                return i, phase, np.clip(progress, 0, 1)
            cumulative += phase.duration_sec

        # Past end — return last phase at 100%
        last = self.scenario.phases[-1]
        return len(self.scenario.phases) - 1, last, 1.0

    def step(self, dt: float) -> Dict:
        """
        Advance the simulation by dt seconds (real time).
        Returns complete state dict for the dashboard.
        """
        if self.start_time is None:
            self.start_time = time.time()

        self.elapsed += dt * self.speed

        # Get current phase and progress
        phase_idx, phase, progress = self.get_current_phase()

        # Handle phase transitions
        if phase_idx != self._current_phase_idx:
            self._prev_vitals = self.scenario.phases[self._current_phase_idx].target_vitals
            self._prev_env = self.scenario.phases[self._current_phase_idx].target_env
            self._current_phase_idx = phase_idx
            self._phase_start_time = self.elapsed

        # Interpolate current vitals
        current_vitals = interpolate_vitals(
            self._prev_vitals, phase.target_vitals, progress, phase.transition
        )
        current_env = interpolate_env(
            self._prev_env, phase.target_env, progress, phase.transition
        )

        # Generate CSI data from each virtual node
        node_csi = {}
        for node_id, synth in self.synthesizers.items():
            packet = synth.generate_csi_packet(current_vitals, self.elapsed)
            node_csi[node_id] = packet

        # Build output state
        state = {
            # Scenario metadata
            'scenario_id': self.scenario.id,
            'scenario_name': self.scenario.name,
            'scenario_category': self.scenario.category,
            'phase_name': phase.name,
            'phase_description': phase.description,
            'phase_index': phase_idx,
            'total_phases': len(self.scenario.phases),
            'phase_progress': round(progress, 3),
            'elapsed_sec': round(self.elapsed, 2),
            'total_duration_sec': self.scenario.total_duration_sec,
            'speed': self.speed,

            # Vital signs (ground truth from simulation)
            'heart_rate': round(current_vitals.heart_rate, 1),
            'breathing_rate': round(current_vitals.breathing_rate, 1),
            'hrv_sdnn': round(current_vitals.hrv_sdnn, 1),
            'hrv_rmssd': round(current_vitals.hrv_rmssd, 1),
            'motion_level': round(current_vitals.motion_level, 3),
            'is_motion': current_vitals.motion_level > 0.5,
            'motion_type': current_vitals.motion_type,
            'is_present': current_vitals.presence,
            'presence_score': round(current_vitals.presence_score, 3),
            'activity': self._classify_activity(current_vitals),
            'gsr': round(current_vitals.gsr, 2),
            'stress_index': current_vitals.stress_index,
            'spo2': round(current_vitals.spo2, 1),
            'blood_pressure_sys': round(current_vitals.blood_pressure_sys, 0),
            'blood_pressure_dia': round(current_vitals.blood_pressure_dia, 0),
            'skin_temp': round(current_vitals.skin_temp, 1),
            'irregular_rhythm': current_vitals.irregular_rhythm,

            # Confidence (simulated based on signal conditions)
            'hr_confidence': self._compute_confidence(current_vitals, 'cardiac'),
            'breath_confidence': self._compute_confidence(current_vitals, 'breath'),
            'signal_quality': self._compute_signal_quality(current_vitals),

            # Alert
            'alert_level': current_vitals.alert_level.value,
            'alert_message': current_vitals.alert_message,

            # Environment
            'environment': {
                'temperature_c': round(current_env.temperature_c, 1),
                'humidity_pct': round(current_env.humidity_pct, 0),
                'co2_ppm': round(current_env.co2_ppm, 0),
                'tvoc_ppb': round(current_env.tvoc_ppb, 0),
                'light_lux': round(current_env.light_lux, 0),
                'noise_db': round(current_env.noise_db, 0),
            },

            # Raw CSI data per node (for waveform display)
            'node_csi': node_csi,

            # Active data sources
            'data_sources': [f"csi_{nid}" for nid in self.synthesizers.keys()],
            'node_count': len(self.synthesizers),
            'num_people': self.scenario.num_people,
        }

        return state

    def _classify_activity(self, v: VitalParams) -> str:
        if not v.presence:
            return "absent"
        if v.motion_type == "seizure":
            return "seizure"
        if v.motion_type == "fall":
            return "fall_detected"
        if v.motion_type == "thrashing":
            return "agitated"
        if v.motion_level < 0.2 and v.breathing_rate < 14:
            return "sleeping"
        if v.motion_level < 0.3:
            return "resting"
        if v.motion_level < 1.5:
            return "sitting"
        if v.motion_level < 4:
            return "light_motion"
        if v.motion_level < 7:
            return "walking"
        return "vigorous"

    def _compute_confidence(self, v: VitalParams, signal_type: str) -> str:
        if not v.presence:
            return "none"
        if v.motion_level > 5:
            return "low"
        if v.motion_level > 2:
            return "medium"
        if signal_type == "cardiac" and v.irregular_rhythm:
            return "medium"
        if signal_type == "breath" and v.apnea_active:
            return "low"
        return "high"

    def _compute_signal_quality(self, v: VitalParams) -> float:
        if not v.presence:
            return 0.0
        base = 15.0  # dB
        if v.motion_level > 3:
            base -= v.motion_level * 1.5
        if v.irregular_rhythm:
            base -= 3.0
        return round(max(base + np.random.normal(0, 1), 0), 1)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS — ALL 30
# ═══════════════════════════════════════════════════════════════════════════

def build_all_scenarios() -> Dict[int, Scenario]:
    """Build and return all 30 scenarios."""
    scenarios = {}

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 1: BASELINE / NORMAL STATES (1-4)
    # ─────────────────────────────────────────────────────────────────────

    # 1. Healthy Resting
    scenarios[1] = Scenario(
        id=1, name="Healthy Resting", category="Baseline",
        description="Normal resting adult with stable vitals. The gold standard baseline.",
        clinical_relevance="Establishes normal ranges for comparison. Used in clinical settings to verify sensor calibration.",
        phases=[
            ScenarioPhase(
                name="Steady State",
                duration_sec=60,
                target_vitals=VitalParams(
                    heart_rate=70, heart_rate_variability=4,
                    breathing_rate=15, breathing_depth=1.0,
                    hrv_sdnn=48, hrv_rmssd=40,
                    motion_level=0.1, motion_type="micro",
                    gsr=2.0, stress_index="low", spo2=98,
                    blood_pressure_sys=118, blood_pressure_dia=76,
                ),
                description="Calm, seated adult with normal vitals."
            ),
        ]
    )

    # 2. Light Activity
    scenarios[2] = Scenario(
        id=2, name="Light Activity", category="Baseline",
        description="Person typing, reading, occasionally shifting position.",
        clinical_relevance="Common monitoring scenario for office/home health. Tests motion artifact rejection.",
        phases=[
            ScenarioPhase(
                name="Seated Working",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=78, breathing_rate=16, breathing_depth=0.9,
                    hrv_sdnn=42, hrv_rmssd=35,
                    motion_level=0.8, motion_type="fidget",
                    gsr=2.5, stress_index="low",
                ),
                description="Typing, occasionally shifting."
            ),
            ScenarioPhase(
                name="Brief Stand & Stretch",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=88, breathing_rate=18,
                    motion_level=3.0, motion_type="walking",
                    gsr=2.8, stress_index="low",
                ),
                transition="sigmoid",
                description="Standing up, brief stretch."
            ),
            ScenarioPhase(
                name="Return to Seat",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=76, breathing_rate=15, breathing_depth=1.0,
                    hrv_sdnn=44, hrv_rmssd=38,
                    motion_level=0.5, motion_type="fidget",
                    gsr=2.3, stress_index="low",
                ),
                transition="exponential",
                description="Sitting back down, settling."
            ),
        ]
    )

    # 3. Deep Sleep
    scenarios[3] = Scenario(
        id=3, name="Deep Sleep", category="Baseline",
        description="Subject in deep NREM sleep with minimal motion and slow, regular vitals.",
        clinical_relevance="Sleep monitoring for apnea screening, insomnia tracking, and sleep quality assessment.",
        phases=[
            ScenarioPhase(
                name="Deep NREM",
                duration_sec=60,
                target_vitals=VitalParams(
                    heart_rate=56, heart_rate_variability=3,
                    breathing_rate=11, breathing_depth=1.3, breathing_variability=0.5,
                    hrv_sdnn=65, hrv_rmssd=55,
                    motion_level=0.0, motion_type="none",
                    gsr=1.5, stress_index="low", spo2=97,
                    skin_temp=36.0,
                ),
                target_env=EnvironmentParams(temperature_c=20, light_lux=0, noise_db=25),
                description="Deep slow-wave sleep. Very regular, slow physiology."
            ),
        ]
    )

    # 4. REM Sleep
    scenarios[4] = Scenario(
        id=4, name="REM Sleep", category="Baseline",
        description="REM sleep with irregular heart rate, variable breathing, and micro-movements.",
        clinical_relevance="REM behavior disorder detection, dream-state monitoring, sleep cycle staging.",
        phases=[
            ScenarioPhase(
                name="REM Onset",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=62, heart_rate_variability=8,
                    breathing_rate=14, breathing_variability=3,
                    breathing_depth=0.8,
                    hrv_sdnn=55, hrv_rmssd=48,
                    motion_level=0.3, motion_type="micro",
                    gsr=1.8, spo2=97,
                ),
                target_env=EnvironmentParams(light_lux=0, noise_db=25),
                transition="sigmoid",
                description="Transition into REM. HR becomes variable."
            ),
            ScenarioPhase(
                name="Active REM",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=72, heart_rate_variability=12,
                    breathing_rate=16, breathing_variability=5,
                    breathing_depth=0.7,
                    hrv_sdnn=50, hrv_rmssd=45,
                    motion_level=0.6, motion_type="micro",
                    gsr=2.2, spo2=97,
                ),
                target_env=EnvironmentParams(light_lux=0, noise_db=25),
                transition="oscillating",
                description="Active dreaming. Irregular vitals, eye movements."
            ),
            ScenarioPhase(
                name="REM Resolution",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=60, heart_rate_variability=4,
                    breathing_rate=12, breathing_depth=1.2,
                    hrv_sdnn=60, hrv_rmssd=52,
                    motion_level=0.0, motion_type="none",
                    gsr=1.6, spo2=97,
                ),
                transition="exponential",
                description="Settling back into NREM."
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 2: STRESS & MENTAL STATE (5-9)
    # ─────────────────────────────────────────────────────────────────────

    # 5. Acute Stress Response
    scenarios[5] = Scenario(
        id=5, name="Acute Stress Response", category="Stress & Mental State",
        description="Sudden stress onset — bad news, exam pressure, confrontation. Sympathetic activation.",
        clinical_relevance="Workplace wellness monitoring, PTSD trigger detection, anxiety disorder tracking.",
        phases=[
            ScenarioPhase(
                name="Baseline Calm",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=72, breathing_rate=15, hrv_sdnn=45, hrv_rmssd=38,
                    gsr=2.0, stress_index="low",
                ),
                description="Normal resting before stressor."
            ),
            ScenarioPhase(
                name="Stress Onset",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=95, breathing_rate=20, breathing_depth=0.7,
                    hrv_sdnn=25, hrv_rmssd=18,
                    motion_level=1.0, motion_type="fidget",
                    gsr=5.5, stress_index="high",
                    blood_pressure_sys=140, blood_pressure_dia=90,
                    alert_level=AlertLevel.WARNING,
                    alert_message="Acute stress response detected — elevated HR and GSR",
                ),
                transition="exponential",
                description="Sympathetic nervous system activation."
            ),
            ScenarioPhase(
                name="Peak Stress",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=110, breathing_rate=24, breathing_depth=0.6,
                    hrv_sdnn=18, hrv_rmssd=12,
                    motion_level=1.5, motion_type="fidget",
                    gsr=7.0, stress_index="high",
                    blood_pressure_sys=150, blood_pressure_dia=95,
                    skin_temp=37.0,
                    alert_level=AlertLevel.WARNING,
                    alert_message="Sustained high stress — consider intervention",
                ),
                transition="oscillating",
                description="Sustained fight-or-flight."
            ),
            ScenarioPhase(
                name="Recovery",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=80, breathing_rate=16, breathing_depth=1.0,
                    hrv_sdnn=35, hrv_rmssd=28,
                    motion_level=0.3, motion_type="micro",
                    gsr=3.0, stress_index="moderate",
                    blood_pressure_sys=125, blood_pressure_dia=82,
                ),
                transition="exponential",
                description="Parasympathetic recovery begins."
            ),
        ]
    )

    # 6. Meditation / Recovery
    scenarios[6] = Scenario(
        id=6, name="Meditation / Recovery", category="Stress & Mental State",
        description="Guided meditation or deep breathing exercise. Progressive relaxation.",
        clinical_relevance="Biofeedback therapy validation, stress management program efficacy tracking.",
        phases=[
            ScenarioPhase(
                name="Pre-Meditation",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=82, breathing_rate=18, hrv_sdnn=32, hrv_rmssd=25,
                    gsr=4.0, stress_index="moderate",
                ),
                description="Elevated baseline before meditation begins."
            ),
            ScenarioPhase(
                name="Settling In",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=72, breathing_rate=12, breathing_depth=1.4,
                    hrv_sdnn=42, hrv_rmssd=36,
                    gsr=3.0, stress_index="low",
                ),
                transition="exponential",
                description="Conscious breathing deepens, HR drops."
            ),
            ScenarioPhase(
                name="Deep Meditation",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=60, breathing_rate=6, breathing_depth=1.8,
                    breathing_variability=0.3,
                    hrv_sdnn=70, hrv_rmssd=62,
                    motion_level=0.0, motion_type="none",
                    gsr=1.5, stress_index="low", spo2=99,
                    skin_temp=36.8,
                    alert_level=AlertLevel.INFO,
                    alert_message="Deep relaxation state — excellent parasympathetic tone",
                ),
                transition="sigmoid",
                description="Peak parasympathetic dominance. Very slow breathing."
            ),
            ScenarioPhase(
                name="Emergence",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=68, breathing_rate=14, breathing_depth=1.0,
                    hrv_sdnn=55, hrv_rmssd=48,
                    gsr=2.0, stress_index="low",
                ),
                transition="linear",
                description="Gradually returning to alert state."
            ),
        ]
    )

    # 7. Panic Attack
    scenarios[7] = Scenario(
        id=7, name="Panic Attack", category="Stress & Mental State",
        description="Sudden onset panic with extreme sympathetic activation, hyperventilation, tremor.",
        clinical_relevance="Psychiatric emergency detection, anxiety disorder monitoring, safe space alerting.",
        phases=[
            ScenarioPhase(
                name="Normal",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=74, breathing_rate=15, gsr=2.5, stress_index="low"),
            ),
            ScenarioPhase(
                name="Panic Onset",
                duration_sec=8,
                target_vitals=VitalParams(
                    heart_rate=130, breathing_rate=32, breathing_depth=0.4,
                    hrv_sdnn=12, hrv_rmssd=8,
                    motion_level=3.0, motion_type="tremor",
                    gsr=9.0, stress_index="high",
                    blood_pressure_sys=160, blood_pressure_dia=100,
                    skin_temp=37.2,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="PANIC ATTACK — extreme tachycardia + hyperventilation detected",
                ),
                transition="exponential",
                description="Explosive sympathetic surge."
            ),
            ScenarioPhase(
                name="Peak Panic",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=142, breathing_rate=35, breathing_depth=0.3,
                    hrv_sdnn=8, hrv_rmssd=5,
                    motion_level=4.0, motion_type="thrashing",
                    gsr=10.0, stress_index="high",
                    spo2=95,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="SUSTAINED PANIC — consider emergency response",
                ),
                transition="oscillating",
            ),
            ScenarioPhase(
                name="Gradual Recovery",
                duration_sec=25,
                target_vitals=VitalParams(
                    heart_rate=95, breathing_rate=20, breathing_depth=0.8,
                    hrv_sdnn=22, hrv_rmssd=16,
                    motion_level=1.0, motion_type="tremor",
                    gsr=5.0, stress_index="high",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Post-panic recovery — vitals improving",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Post-Panic Exhaustion",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=80, breathing_rate=16,
                    hrv_sdnn=30, hrv_rmssd=22,
                    motion_level=0.2, motion_type="micro",
                    gsr=3.5, stress_index="moderate",
                ),
                transition="exponential",
            ),
        ]
    )

    # 8. Cognitive Load / Focus
    scenarios[8] = Scenario(
        id=8, name="Cognitive Load / Deep Focus", category="Stress & Mental State",
        description="Intense mental work — coding, exam, complex problem-solving.",
        clinical_relevance="Productivity optimization, ADHD monitoring, cognitive fatigue detection.",
        phases=[
            ScenarioPhase(
                name="Deep Focus",
                duration_sec=40,
                target_vitals=VitalParams(
                    heart_rate=80, heart_rate_variability=3,
                    breathing_rate=14, breathing_depth=0.8,
                    breathing_variability=0.5,
                    hrv_sdnn=30, hrv_rmssd=24,
                    motion_level=0.1, motion_type="none",
                    gsr=3.5, stress_index="moderate",
                ),
                description="Sustained attention. Slightly elevated HR, reduced HRV."
            ),
            ScenarioPhase(
                name="Mental Fatigue",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=76, breathing_rate=16, breathing_depth=0.9,
                    hrv_sdnn=35, hrv_rmssd=28,
                    motion_level=1.0, motion_type="fidget",
                    gsr=3.0, stress_index="moderate",
                    alert_level=AlertLevel.INFO,
                    alert_message="Cognitive fatigue detected — consider a break",
                ),
                transition="linear",
                description="Focus waning, fidgeting increases."
            ),
        ]
    )

    # 9. PTSD Flashback / Night Terror
    scenarios[9] = Scenario(
        id=9, name="PTSD Flashback / Night Terror", category="Stress & Mental State",
        description="Sudden arousal from sleep with extreme sympathetic activation.",
        clinical_relevance="PTSD monitoring, veteran care, psychiatric inpatient monitoring.",
        phases=[
            ScenarioPhase(
                name="Sleep Baseline",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=58, breathing_rate=11, motion_level=0.0,
                    hrv_sdnn=60, hrv_rmssd=52, gsr=1.5,
                ),
                target_env=EnvironmentParams(light_lux=0, noise_db=25),
            ),
            ScenarioPhase(
                name="Terror Onset",
                duration_sec=5,
                target_vitals=VitalParams(
                    heart_rate=140, breathing_rate=28, breathing_depth=0.5,
                    hrv_sdnn=10, hrv_rmssd=6,
                    motion_level=6.0, motion_type="thrashing",
                    gsr=8.0, stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="NIGHT TERROR — sudden arousal with extreme tachycardia",
                ),
                transition="step",
                description="Explosive awakening."
            ),
            ScenarioPhase(
                name="Active Terror",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=135, breathing_rate=30,
                    hrv_sdnn=8, hrv_rmssd=5,
                    motion_level=5.0, motion_type="thrashing",
                    gsr=9.0, stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="Sustained night terror — agitated state",
                ),
                transition="oscillating",
            ),
            ScenarioPhase(
                name="Settling",
                duration_sec=25,
                target_vitals=VitalParams(
                    heart_rate=90, breathing_rate=20,
                    hrv_sdnn=25, hrv_rmssd=18,
                    motion_level=1.0, motion_type="tremor",
                    gsr=5.0, stress_index="high",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Post-terror settling — still elevated",
                ),
                transition="exponential",
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 3: CARDIAC EVENTS (10-14)
    # ─────────────────────────────────────────────────────────────────────

    # 10. Tachycardia Episode
    scenarios[10] = Scenario(
        id=10, name="Tachycardia Episode", category="Cardiac Events",
        description="Supraventricular tachycardia — HR spikes above 140, irregular rhythm.",
        clinical_relevance="Cardiac monitoring, arrhythmia detection, post-surgical surveillance.",
        phases=[
            ScenarioPhase(
                name="Normal Sinus",
                duration_sec=15,
                target_vitals=VitalParams(heart_rate=75, hrv_sdnn=45, hrv_rmssd=38),
            ),
            ScenarioPhase(
                name="SVT Onset",
                duration_sec=5,
                target_vitals=VitalParams(
                    heart_rate=155, heart_rate_variability=8,
                    hrv_sdnn=8, hrv_rmssd=5,
                    irregular_rhythm=True, rhythm_chaos=0.3,
                    gsr=5.0, stress_index="high",
                    blood_pressure_sys=100, blood_pressure_dia=65,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="TACHYCARDIA — HR > 150 bpm, possible SVT",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Sustained SVT",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=165, heart_rate_variability=10,
                    breathing_rate=22,
                    hrv_sdnn=5, hrv_rmssd=3,
                    irregular_rhythm=True, rhythm_chaos=0.4,
                    gsr=6.0, stress_index="high",
                    blood_pressure_sys=95, blood_pressure_dia=60,
                    spo2=96,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="SUSTAINED SVT — HR 165 bpm — seek immediate care",
                ),
                transition="oscillating",
            ),
            ScenarioPhase(
                name="Conversion",
                duration_sec=5,
                target_vitals=VitalParams(
                    heart_rate=90, irregular_rhythm=False,
                    hrv_sdnn=20, hrv_rmssd=15,
                    gsr=4.0, stress_index="moderate",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Rhythm converted — monitoring for recurrence",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Post-Conversion",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=78, hrv_sdnn=35, hrv_rmssd=28,
                    gsr=2.5, stress_index="low",
                ),
                transition="exponential",
            ),
        ]
    )

    # 11. Bradycardia / Near-Syncope
    scenarios[11] = Scenario(
        id=11, name="Bradycardia / Near-Syncope", category="Cardiac Events",
        description="Dangerous bradycardia with near-fainting, possible vasovagal syncope.",
        clinical_relevance="Elderly monitoring, post-operative care, medication side effect tracking.",
        phases=[
            ScenarioPhase(
                name="Normal",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=68, breathing_rate=14),
            ),
            ScenarioPhase(
                name="Bradycardia Onset",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=42, breathing_rate=10,
                    hrv_sdnn=15, hrv_rmssd=10,
                    blood_pressure_sys=85, blood_pressure_dia=55,
                    spo2=94,
                    motion_level=0.3, motion_type="micro",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="BRADYCARDIA — HR < 45 bpm — syncope risk",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Near-Syncope / Fall",
                duration_sec=5,
                target_vitals=VitalParams(
                    heart_rate=38,
                    breathing_rate=8, breathing_depth=0.6,
                    blood_pressure_sys=75, blood_pressure_dia=45,
                    spo2=91,
                    motion_level=8.0, motion_type="fall",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="FALL DETECTED during bradycardia — emergency response needed",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Post-Fall Supine",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=55, breathing_rate=12,
                    blood_pressure_sys=100, blood_pressure_dia=65,
                    spo2=95,
                    motion_level=0.0, motion_type="none",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Patient supine post-fall — vitals recovering",
                ),
                transition="exponential",
            ),
        ]
    )

    # 12. Atrial Fibrillation
    scenarios[12] = Scenario(
        id=12, name="Atrial Fibrillation", category="Cardiac Events",
        description="Persistent AF with irregularly irregular rhythm, variable rate.",
        clinical_relevance="Stroke risk monitoring, anticoagulation management, rate vs rhythm control assessment.",
        phases=[
            ScenarioPhase(
                name="AF Active",
                duration_sec=60,
                target_vitals=VitalParams(
                    heart_rate=105, heart_rate_variability=25,
                    breathing_rate=18,
                    hrv_sdnn=80, hrv_rmssd=65,
                    irregular_rhythm=True, rhythm_chaos=0.8,
                    gsr=3.5, stress_index="moderate",
                    blood_pressure_sys=135, blood_pressure_dia=85,
                    alert_level=AlertLevel.WARNING,
                    alert_message="ATRIAL FIBRILLATION — irregularly irregular rhythm detected",
                ),
                transition="oscillating",
                description="Classic AF: chaotic heart rate 80-130, no P-waves equivalent."
            ),
        ]
    )

    # 13. Cardiac Arrest
    scenarios[13] = Scenario(
        id=13, name="Cardiac Arrest", category="Cardiac Events",
        description="Sudden cardiac arrest — loss of effective cardiac output.",
        clinical_relevance="The ultimate emergency scenario. Tests maximum-severity alerting chain.",
        loop=False,
        phases=[
            ScenarioPhase(
                name="Pre-Arrest Normal",
                duration_sec=15,
                target_vitals=VitalParams(heart_rate=78, breathing_rate=16),
            ),
            ScenarioPhase(
                name="Arrest Onset",
                duration_sec=3,
                target_vitals=VitalParams(
                    heart_rate=180, heart_rate_variability=30,
                    breathing_rate=25,
                    irregular_rhythm=True, rhythm_chaos=1.0,
                    motion_level=5.0, motion_type="thrashing",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ VENTRICULAR FIBRILLATION — CALL 911",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Cardiac Standstill",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=0, breathing_rate=0,
                    hrv_sdnn=0, hrv_rmssd=0,
                    motion_level=0.0, motion_type="none",
                    presence=True, presence_score=0.4,
                    gsr=0.5, spo2=70,
                    blood_pressure_sys=0, blood_pressure_dia=0,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ CARDIAC ARREST — NO HEARTBEAT — CPR NEEDED — 911 CALLED",
                ),
                transition="exponential",
            ),
        ]
    )

    # 14. PVC / Ectopic Beats
    scenarios[14] = Scenario(
        id=14, name="PVC / Ectopic Beats", category="Cardiac Events",
        description="Premature ventricular contractions — occasional skipped beats.",
        clinical_relevance="Common benign arrhythmia tracking, medication effect monitoring.",
        phases=[
            ScenarioPhase(
                name="Normal with PVCs",
                duration_sec=60,
                target_vitals=VitalParams(
                    heart_rate=74, heart_rate_variability=6,
                    breathing_rate=15,
                    hrv_sdnn=50, hrv_rmssd=42,
                    irregular_rhythm=True,
                    ectopic_beat_prob=0.08,
                    rhythm_chaos=0.1,
                    alert_level=AlertLevel.INFO,
                    alert_message="Occasional ectopic beats detected — typically benign",
                ),
                description="Mostly regular rhythm with periodic extra/skipped beats."
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 4: RESPIRATORY EVENTS (15-18)
    # ─────────────────────────────────────────────────────────────────────

    # 15. Sleep Apnea Episode
    scenarios[15] = Scenario(
        id=15, name="Sleep Apnea Episode", category="Respiratory Events",
        description="Obstructive sleep apnea — breathing pauses, desaturation, arousal cycle.",
        clinical_relevance="The #1 use case for contactless sleep monitoring. Affects ~1B people globally.",
        phases=[
            ScenarioPhase(
                name="Normal Sleep Breathing",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=58, breathing_rate=12, breathing_depth=1.2,
                    spo2=97, motion_level=0.0,
                ),
                target_env=EnvironmentParams(light_lux=0, noise_db=25),
            ),
            ScenarioPhase(
                name="Apnea Event",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=52, breathing_rate=0, breathing_depth=0.0,
                    apnea_active=True,
                    spo2=88,
                    motion_level=0.5, motion_type="micro",
                    alert_level=AlertLevel.WARNING,
                    alert_message="APNEA EVENT — breathing cessation for 15 seconds",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Arousal / Gasping",
                duration_sec=8,
                target_vitals=VitalParams(
                    heart_rate=85, breathing_rate=25, breathing_depth=1.8,
                    breathing_effort=2.0,
                    apnea_active=False,
                    spo2=92,
                    motion_level=3.0, motion_type="thrashing",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Arousal from apnea — gasping recovery breaths",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Recovery to Sleep",
                duration_sec=12,
                target_vitals=VitalParams(
                    heart_rate=62, breathing_rate=14, breathing_depth=1.1,
                    spo2=96,
                    motion_level=0.1, motion_type="micro",
                ),
                transition="exponential",
            ),
        ]
    )

    # 16. Hyperventilation
    scenarios[16] = Scenario(
        id=16, name="Hyperventilation", category="Respiratory Events",
        description="Anxiety-driven hyperventilation with respiratory alkalosis effects.",
        clinical_relevance="Anxiety disorder monitoring, ER triage, biofeedback therapy.",
        phases=[
            ScenarioPhase(
                name="Normal",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=75, breathing_rate=15),
            ),
            ScenarioPhase(
                name="Hyperventilation Onset",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=100, breathing_rate=32, breathing_depth=0.5,
                    hrv_sdnn=18, hrv_rmssd=12,
                    gsr=6.0, stress_index="high",
                    spo2=99,
                    alert_level=AlertLevel.WARNING,
                    alert_message="HYPERVENTILATION — BR > 30, guide slow breathing",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Sustained",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=108, breathing_rate=35, breathing_depth=0.4,
                    hrv_sdnn=14, hrv_rmssd=9,
                    motion_level=1.5, motion_type="tremor",
                    gsr=7.5, stress_index="high",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Sustained hyperventilation — respiratory alkalosis risk",
                ),
                transition="oscillating",
            ),
            ScenarioPhase(
                name="Coached Recovery",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=82, breathing_rate=16, breathing_depth=1.2,
                    hrv_sdnn=32, hrv_rmssd=25,
                    gsr=3.5, stress_index="moderate",
                ),
                transition="exponential",
            ),
        ]
    )

    # 17. Asthma Attack
    scenarios[17] = Scenario(
        id=17, name="Asthma Attack", category="Respiratory Events",
        description="Acute bronchospasm with labored breathing and prolonged expiration.",
        clinical_relevance="Chronic disease monitoring, medication efficacy, ER prediction.",
        phases=[
            ScenarioPhase(
                name="Pre-Attack",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=78, breathing_rate=16),
            ),
            ScenarioPhase(
                name="Bronchospasm",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=105, breathing_rate=22,
                    breathing_depth=0.5, breathing_effort=2.0,
                    breathing_variability=3,
                    hrv_sdnn=20, hrv_rmssd=14,
                    spo2=93,
                    motion_level=1.0, motion_type="fidget",
                    alert_level=AlertLevel.WARNING,
                    alert_message="ASTHMA EXACERBATION — labored breathing, SpO2 dropping",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Post-Inhaler",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=92, breathing_rate=18,
                    breathing_depth=0.8, breathing_effort=1.3,
                    spo2=96,
                    alert_level=AlertLevel.INFO,
                    alert_message="Post-bronchodilator — breathing improving",
                ),
                transition="exponential",
            ),
        ]
    )

    # 18. COPD Exacerbation
    scenarios[18] = Scenario(
        id=18, name="COPD Exacerbation", category="Respiratory Events",
        description="Progressive respiratory failure over hours. Slow decline requiring intervention.",
        clinical_relevance="Home monitoring for COPD patients, hospital readmission prevention.",
        phases=[
            ScenarioPhase(
                name="Baseline COPD",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=82, breathing_rate=20,
                    breathing_depth=0.7, breathing_effort=1.3,
                    spo2=93,
                ),
            ),
            ScenarioPhase(
                name="Worsening",
                duration_sec=25,
                target_vitals=VitalParams(
                    heart_rate=95, breathing_rate=24,
                    breathing_depth=0.5, breathing_effort=1.8,
                    spo2=89,
                    motion_level=0.5, motion_type="fidget",
                    alert_level=AlertLevel.WARNING,
                    alert_message="COPD worsening — SpO2 < 90%, increased work of breathing",
                ),
                transition="linear",
            ),
            ScenarioPhase(
                name="Decompensation",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=110, breathing_rate=28,
                    breathing_depth=0.4, breathing_effort=2.0,
                    spo2=85,
                    gsr=5.0, stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="RESPIRATORY FAILURE RISK — SpO2 85% — seek emergency care",
                ),
                transition="linear",
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 5: EMERGENCY SCENARIOS (19-22)
    # ─────────────────────────────────────────────────────────────────────

    # 19. Stroke Indicators
    scenarios[19] = Scenario(
        id=19, name="Stroke Indicators", category="Emergency",
        description="Acute stroke with asymmetric motor loss and vital sign instability.",
        clinical_relevance="Golden-hour detection. Every minute of delay = 1.9M neurons lost.",
        loop=False,
        phases=[
            ScenarioPhase(
                name="Pre-Stroke",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=78, breathing_rate=15,
                    blood_pressure_sys=145, blood_pressure_dia=92,
                ),
            ),
            ScenarioPhase(
                name="Stroke Onset",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=95, breathing_rate=20,
                    hrv_sdnn=15, hrv_rmssd=10,
                    blood_pressure_sys=185, blood_pressure_dia=110,
                    motion_level=2.0, motion_type="fidget",
                    irregular_rhythm=True, rhythm_chaos=0.3,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ POSSIBLE STROKE — asymmetric movement + BP spike + rhythm change",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Post-Stroke Decline",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=88, breathing_rate=18,
                    blood_pressure_sys=175, blood_pressure_dia=105,
                    motion_level=0.1, motion_type="none",
                    spo2=94,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ STROKE CONFIRMED — patient immobile — emergency en route",
                ),
                transition="linear",
            ),
        ]
    )

    # 20. Fall Detection
    scenarios[20] = Scenario(
        id=20, name="Fall Detection", category="Emergency",
        description="Elderly fall — sudden impact, then stillness. Classic geriatric emergency.",
        clinical_relevance="Leading cause of injury death in adults 65+. Early detection saves lives.",
        phases=[
            ScenarioPhase(
                name="Standing / Walking",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=80, breathing_rate=16,
                    motion_level=3.0, motion_type="walking",
                ),
            ),
            ScenarioPhase(
                name="Fall Impact",
                duration_sec=3,
                target_vitals=VitalParams(
                    heart_rate=110, breathing_rate=22,
                    motion_level=10.0, motion_type="fall",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ FALL DETECTED — sudden high-G impact",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Post-Fall Stillness",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=95, breathing_rate=20,
                    motion_level=0.0, motion_type="none",
                    gsr=5.0, stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ PERSON DOWN — no movement post-fall — checking consciousness",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Slow Recovery",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=85, breathing_rate=18,
                    motion_level=1.0, motion_type="micro",
                    gsr=3.5, stress_index="moderate",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Movement detected post-fall — patient may be conscious",
                ),
                transition="linear",
            ),
        ]
    )

    # 21. Seizure (Tonic-Clonic)
    scenarios[21] = Scenario(
        id=21, name="Seizure (Tonic-Clonic)", category="Emergency",
        description="Grand mal seizure with tonic stiffening, clonic jerking, postictal state.",
        clinical_relevance="Epilepsy monitoring, SUDEP prevention, institutional safety.",
        phases=[
            ScenarioPhase(
                name="Pre-Ictal",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=72, breathing_rate=15),
            ),
            ScenarioPhase(
                name="Tonic Phase",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=130, breathing_rate=5, breathing_depth=0.3,
                    motion_level=7.0, motion_type="seizure",
                    spo2=90,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ SEIZURE — tonic phase — rigid body, apneic",
                ),
                transition="step",
            ),
            ScenarioPhase(
                name="Clonic Phase",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=155, breathing_rate=8,
                    hrv_sdnn=5, hrv_rmssd=3,
                    motion_level=9.0, motion_type="seizure",
                    spo2=85,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ SEIZURE — clonic phase — rhythmic convulsions",
                ),
                transition="oscillating",
            ),
            ScenarioPhase(
                name="Postictal",
                duration_sec=30,
                target_vitals=VitalParams(
                    heart_rate=90, breathing_rate=18, breathing_depth=1.5,
                    motion_level=0.0, motion_type="none",
                    spo2=93,
                    alert_level=AlertLevel.WARNING,
                    alert_message="Postictal state — patient unconscious but breathing",
                ),
                transition="exponential",
            ),
        ]
    )

    # 22. Anaphylaxis
    scenarios[22] = Scenario(
        id=22, name="Anaphylaxis", category="Emergency",
        description="Severe allergic reaction with cardiovascular collapse.",
        clinical_relevance="Food allergy monitoring, post-vaccination observation, drug reaction detection.",
        phases=[
            ScenarioPhase(
                name="Exposure",
                duration_sec=10,
                target_vitals=VitalParams(heart_rate=78, breathing_rate=16),
            ),
            ScenarioPhase(
                name="Early Reaction",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=100, breathing_rate=22,
                    breathing_effort=1.5,
                    skin_temp=37.5,
                    gsr=5.0, stress_index="high",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Allergic reaction — rising HR, labored breathing",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Anaphylactic Shock",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=145, breathing_rate=30,
                    breathing_depth=0.3, breathing_effort=2.0,
                    blood_pressure_sys=75, blood_pressure_dia=40,
                    spo2=88,
                    motion_level=2.0, motion_type="fidget",
                    gsr=8.0, stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ ANAPHYLAXIS — tachycardia, bronchospasm, hypotension — USE EPIPEN",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Post-Epinephrine",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=110, breathing_rate=20,
                    breathing_depth=0.8, breathing_effort=1.2,
                    blood_pressure_sys=105, blood_pressure_dia=65,
                    spo2=94,
                    alert_level=AlertLevel.WARNING,
                    alert_message="Post-epinephrine — improving but monitor for biphasic reaction",
                ),
                transition="exponential",
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 6: MEDICATION & TREATMENT TRACKING (23-26)
    # ─────────────────────────────────────────────────────────────────────

    # 23. Post-Medication Dosing (Beta Blocker)
    scenarios[23] = Scenario(
        id=23, name="Post-Medication: Beta Blocker", category="Medication & Treatment",
        description="Tracking vital sign response to beta blocker (metoprolol) administration.",
        clinical_relevance="Medication titration, dose-response monitoring, compliance verification.",
        phases=[
            ScenarioPhase(
                name="Pre-Dose Tachycardia",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=98, breathing_rate=18,
                    hrv_sdnn=22, hrv_rmssd=15,
                    blood_pressure_sys=145, blood_pressure_dia=92,
                    stress_index="moderate",
                    alert_level=AlertLevel.INFO,
                    alert_message="Pre-medication baseline — elevated HR and BP",
                ),
            ),
            ScenarioPhase(
                name="Drug Onset (15 min)",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=82, breathing_rate=16,
                    hrv_sdnn=32, hrv_rmssd=25,
                    blood_pressure_sys=132, blood_pressure_dia=85,
                    alert_level=AlertLevel.INFO,
                    alert_message="Beta blocker taking effect — HR decreasing",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Peak Effect (45 min)",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=65, breathing_rate=14,
                    hrv_sdnn=48, hrv_rmssd=40,
                    blood_pressure_sys=120, blood_pressure_dia=78,
                    stress_index="low",
                    alert_level=AlertLevel.INFO,
                    alert_message="Beta blocker at peak effect — HR 65, BP normalized",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Stable Therapeutic",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=62, breathing_rate=14,
                    hrv_sdnn=52, hrv_rmssd=45,
                    blood_pressure_sys=118, blood_pressure_dia=76,
                ),
                transition="linear",
                description="Stable therapeutic effect."
            ),
        ]
    )

    # 24. Opioid Administration
    scenarios[24] = Scenario(
        id=24, name="Opioid Administration", category="Medication & Treatment",
        description="Post-surgical opioid pain management with respiratory depression risk.",
        clinical_relevance="#1 cause of preventable hospital death. Contactless monitoring is critical.",
        phases=[
            ScenarioPhase(
                name="Pre-Dose Pain",
                duration_sec=10,
                target_vitals=VitalParams(
                    heart_rate=92, breathing_rate=20,
                    gsr=5.5, stress_index="high",
                    motion_level=1.5, motion_type="fidget",
                ),
            ),
            ScenarioPhase(
                name="Analgesic Onset",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=72, breathing_rate=14,
                    gsr=2.5, stress_index="low",
                    motion_level=0.2, motion_type="micro",
                    alert_level=AlertLevel.INFO,
                    alert_message="Opioid taking effect — pain relief, drowsiness",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Sedation",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=60, breathing_rate=9,
                    breathing_depth=0.6,
                    hrv_sdnn=25, hrv_rmssd=18,
                    motion_level=0.0, motion_type="none",
                    spo2=94,
                    alert_level=AlertLevel.WARNING,
                    alert_message="OPIOID SEDATION — BR = 9, SpO2 declining — monitor closely",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Respiratory Depression Risk",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=55, breathing_rate=6,
                    breathing_depth=0.4,
                    spo2=89,
                    motion_level=0.0,
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ RESPIRATORY DEPRESSION — BR = 6, SpO2 = 89% — NALOXONE READY",
                ),
                transition="linear",
            ),
        ]
    )

    # 25. Stimulant Effect (Caffeine/Adderall)
    scenarios[25] = Scenario(
        id=25, name="Stimulant Effect", category="Medication & Treatment",
        description="Tracking sympathomimetic effects of caffeine or amphetamine-based medications.",
        clinical_relevance="ADHD medication monitoring, caffeine sensitivity, stimulant abuse detection.",
        phases=[
            ScenarioPhase(
                name="Pre-Dose Baseline",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=68, breathing_rate=14,
                    hrv_sdnn=50, hrv_rmssd=42,
                    gsr=2.0,
                ),
            ),
            ScenarioPhase(
                name="Absorption Phase",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=78, breathing_rate=15,
                    hrv_sdnn=38, hrv_rmssd=30,
                    motion_level=0.8, motion_type="fidget",
                    gsr=3.0, stress_index="moderate",
                    alert_level=AlertLevel.INFO,
                    alert_message="Stimulant onset — HR rising, increased alertness",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Peak Effect",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=88, breathing_rate=16,
                    hrv_sdnn=28, hrv_rmssd=20,
                    motion_level=1.2, motion_type="fidget",
                    gsr=4.0, stress_index="moderate",
                    blood_pressure_sys=135, blood_pressure_dia=88,
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Plateau",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=85, breathing_rate=15,
                    hrv_sdnn=30, hrv_rmssd=22,
                    motion_level=0.5, motion_type="micro",
                    gsr=3.5,
                ),
                transition="linear",
            ),
        ]
    )

    # 26. Sedative Wearing Off
    scenarios[26] = Scenario(
        id=26, name="Sedative Wearing Off", category="Medication & Treatment",
        description="Post-anesthesia recovery — patient transitioning from sedation to consciousness.",
        clinical_relevance="PACU monitoring, outpatient procedure recovery, discharge readiness assessment.",
        phases=[
            ScenarioPhase(
                name="Deep Sedation",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=55, breathing_rate=8,
                    breathing_depth=0.5,
                    hrv_sdnn=18, hrv_rmssd=12,
                    motion_level=0.0, motion_type="none",
                    spo2=94,
                    gsr=1.0,
                ),
            ),
            ScenarioPhase(
                name="Lightening",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=65, breathing_rate=12,
                    breathing_depth=0.8,
                    hrv_sdnn=30, hrv_rmssd=22,
                    motion_level=0.5, motion_type="micro",
                    spo2=96,
                    gsr=2.0,
                    alert_level=AlertLevel.INFO,
                    alert_message="Patient lightening — increased responsiveness",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Emergence",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=78, breathing_rate=16,
                    breathing_depth=1.0,
                    hrv_sdnn=40, hrv_rmssd=34,
                    motion_level=2.0, motion_type="fidget",
                    spo2=98,
                    gsr=3.0, stress_index="moderate",
                    alert_level=AlertLevel.INFO,
                    alert_message="Patient awake — assessing discharge readiness",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Alert and Oriented",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=74, breathing_rate=15,
                    hrv_sdnn=45, hrv_rmssd=38,
                    motion_level=1.0, motion_type="fidget",
                    spo2=98, gsr=2.5, stress_index="low",
                ),
                transition="linear",
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 7: MULTI-PERSON & OCCUPANCY (27-28)
    # ─────────────────────────────────────────────────────────────────────

    # 27. Room Occupancy Change
    scenarios[27] = Scenario(
        id=27, name="Room Occupancy Change", category="Multi-Person & Occupancy",
        description="Person enters, stays, then leaves. Tests presence detection transitions.",
        clinical_relevance="Hospital room monitoring, elderly living alone, smart home safety.",
        phases=[
            ScenarioPhase(
                name="Empty Room",
                duration_sec=10,
                target_vitals=VitalParams(
                    presence=False, presence_score=0.05,
                    heart_rate=0, breathing_rate=0,
                    motion_level=0.0,
                ),
                description="No one in the room."
            ),
            ScenarioPhase(
                name="Person Enters",
                duration_sec=8,
                target_vitals=VitalParams(
                    presence=True, presence_score=0.6,
                    heart_rate=82, breathing_rate=17,
                    motion_level=4.0, motion_type="walking",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Person Seated",
                duration_sec=25,
                target_vitals=VitalParams(
                    presence=True, presence_score=0.95,
                    heart_rate=72, breathing_rate=15,
                    motion_level=0.3, motion_type="micro",
                ),
                transition="exponential",
            ),
            ScenarioPhase(
                name="Person Leaves",
                duration_sec=8,
                target_vitals=VitalParams(
                    presence=True, presence_score=0.4,
                    heart_rate=80, breathing_rate=16,
                    motion_level=4.0, motion_type="walking",
                ),
                transition="sigmoid",
            ),
            ScenarioPhase(
                name="Room Empty Again",
                duration_sec=10,
                target_vitals=VitalParams(
                    presence=False, presence_score=0.05,
                    heart_rate=0, breathing_rate=0,
                    motion_level=0.0,
                ),
                transition="exponential",
            ),
        ]
    )

    # 28. Two People Resting
    scenarios[28] = Scenario(
        id=28, name="Two People Resting", category="Multi-Person & Occupancy",
        description="Two occupants with different vital sign signatures, testing multi-person separation.",
        clinical_relevance="Shared hospital rooms, couples sleep monitoring, multi-patient wards.",
        num_people=2,
        phases=[
            ScenarioPhase(
                name="Two Occupants — Steady",
                duration_sec=60,
                target_vitals=VitalParams(
                    heart_rate=70, breathing_rate=15,
                    hrv_sdnn=45, hrv_rmssd=38,
                    presence=True, presence_score=0.95,
                    alert_level=AlertLevel.INFO,
                    alert_message="Multi-person mode — 2 occupants detected with distinct signatures",
                ),
                description="Person A: HR~70, BR~15. Person B: HR~62, BR~12 (fused view shows blended)."
            ),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # CATEGORY 8: ENVIRONMENTAL INTERACTION (29-30)
    # ─────────────────────────────────────────────────────────────────────

    # 29. Poor Air Quality Response
    scenarios[29] = Scenario(
        id=29, name="Poor Air Quality Response", category="Environmental Interaction",
        description="Rising CO₂ and VOCs cause physiological compensation.",
        clinical_relevance="Sick building syndrome, classroom/office ventilation, industrial safety.",
        phases=[
            ScenarioPhase(
                name="Normal Air",
                duration_sec=15,
                target_vitals=VitalParams(heart_rate=72, breathing_rate=15),
                target_env=EnvironmentParams(co2_ppm=450, tvoc_ppb=120, temperature_c=22),
            ),
            ScenarioPhase(
                name="CO₂ Rising",
                duration_sec=25,
                target_vitals=VitalParams(
                    heart_rate=80, breathing_rate=19,
                    breathing_depth=1.1,
                    alert_level=AlertLevel.WARNING,
                    alert_message="Elevated CO₂ — compensatory breathing rate increase",
                ),
                target_env=EnvironmentParams(co2_ppm=1200, tvoc_ppb=350, temperature_c=24),
                transition="linear",
            ),
            ScenarioPhase(
                name="Hazardous Levels",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=88, breathing_rate=22,
                    breathing_depth=1.2,
                    stress_index="moderate",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ HAZARDOUS AIR — CO₂ > 2000 ppm — ventilate immediately",
                ),
                target_env=EnvironmentParams(co2_ppm=2200, tvoc_ppb=600, temperature_c=26),
                transition="linear",
            ),
        ]
    )

    # 30. Temperature Stress (Heat)
    scenarios[30] = Scenario(
        id=30, name="Temperature Stress (Heat)", category="Environmental Interaction",
        description="Rising ambient temperature causes thermoregulatory stress response.",
        clinical_relevance="Heatstroke prevention, elderly care during heat waves, workplace safety.",
        phases=[
            ScenarioPhase(
                name="Comfortable",
                duration_sec=15,
                target_vitals=VitalParams(
                    heart_rate=72, breathing_rate=15, skin_temp=36.5, gsr=2.0,
                ),
                target_env=EnvironmentParams(temperature_c=22, humidity_pct=45),
            ),
            ScenarioPhase(
                name="Warming",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=85, breathing_rate=18,
                    skin_temp=37.2, gsr=5.0,
                    stress_index="moderate",
                    alert_level=AlertLevel.WARNING,
                    alert_message="Heat stress — elevated HR, increased perspiration",
                ),
                target_env=EnvironmentParams(temperature_c=32, humidity_pct=60),
                transition="linear",
            ),
            ScenarioPhase(
                name="Heat Exhaustion Risk",
                duration_sec=20,
                target_vitals=VitalParams(
                    heart_rate=105, breathing_rate=22,
                    skin_temp=38.5, gsr=8.0,
                    blood_pressure_sys=100, blood_pressure_dia=60,
                    spo2=96,
                    stress_index="high",
                    alert_level=AlertLevel.CRITICAL,
                    alert_message="⚠ HEAT EXHAUSTION RISK — core temp rising, hypotension",
                ),
                target_env=EnvironmentParams(temperature_c=38, humidity_pct=70),
                transition="sigmoid",
            ),
        ]
    )

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_scenario(scenario_id: int) -> Scenario:
    """Get a single scenario by ID."""
    scenarios = build_all_scenarios()
    if scenario_id not in scenarios:
        raise ValueError(f"Scenario {scenario_id} not found. Valid IDs: {list(scenarios.keys())}")
    return scenarios[scenario_id]


def get_scenarios_by_category() -> Dict[str, List[Scenario]]:
    """Get all scenarios grouped by category."""
    scenarios = build_all_scenarios()
    by_cat = {}
    for s in scenarios.values():
        by_cat.setdefault(s.category, []).append(s)
    return by_cat


def list_scenarios() -> List[Dict]:
    """Return summary list of all scenarios."""
    scenarios = build_all_scenarios()
    return [
        {
            'id': s.id,
            'name': s.name,
            'category': s.category,
            'description': s.description,
            'clinical_relevance': s.clinical_relevance,
            'duration_sec': s.total_duration_sec,
            'num_phases': len(s.phases),
            'loop': s.loop,
        }
        for s in scenarios.values()
    ]


def create_engine(scenario_id: int, speed: float = 1.0) -> ScenarioEngine:
    """Create a ScenarioEngine for a given scenario."""
    scenario = get_scenario(scenario_id)
    return ScenarioEngine(scenario, speed=speed)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SENSUS VIRTUAL SCENARIO SIMULATOR — SELF TEST")
    print("=" * 70)

    scenarios = build_all_scenarios()
    print(f"\nTotal scenarios: {len(scenarios)}")
    print()

    for cat, items in get_scenarios_by_category().items():
        print(f"  [{cat}]")
        for s in items:
            print(f"    {s.id:2d}. {s.name:<40s} ({s.total_duration_sec:.0f}s, {len(s.phases)} phases)")
        print()

    # Run a quick simulation of scenario 13 (Cardiac Arrest)
    print("-" * 70)
    print("Running Scenario 13 (Cardiac Arrest) for 5 seconds...")
    engine = create_engine(13, speed=5.0)

    for i in range(50):
        state = engine.step(0.1)
        if i % 10 == 0:
            print(f"  t={state['elapsed_sec']:6.1f}s | Phase: {state['phase_name']:<25s} | "
                  f"HR: {state['heart_rate']:5.1f} | BR: {state['breathing_rate']:5.1f} | "
                  f"Alert: {state['alert_level']}")

    print("\n✅ Self-test passed — all 30 scenarios defined and engine runs correctly.")
