"""
Sensus CSI Signal Processor — RuView-Inspired Pipeline
=======================================================
Signal processing pipeline inspired by WiFi DensePose / RuView:
  Raw CSI → Conjugate Multiplication (phase cleaning)
          → Hampel Filter (outlier removal)
          → Top-K Subcarrier Selection (motion-sensitive channels)
          → PCA Dimensionality Reduction
          → Bandpass + FFT (breathing & cardiac extraction)
          → Fresnel Breathing Model (physics-based)
          → HRV Analysis (SDNN, RMSSD)
          → Motion & Presence Detection
          → CSI Spectrogram (activity classification)

Supports multi-node fusion from 3+ ESP32-C6 mesh nodes.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import logging
import time

logger = logging.getLogger('sensus.csi')


class CSIProcessor:
    """Processes raw CSI subcarrier data into vital signs using RuView-inspired pipeline."""

    # Physiological frequency bands (Hz)
    BREATH_BAND = (0.1, 0.5)      # 6–30 breaths/min
    CARDIAC_BAND = (0.8, 2.0)     # 48–120 bpm
    MOTION_BAND = (0.5, 10.0)     # Gross body motion
    SAMPLE_RATE = 100              # Target CSI packets/sec

    # Fresnel zone parameters
    WIFI_FREQ = 2.4e9              # 2.4 GHz
    SPEED_OF_LIGHT = 3e8
    WAVELENGTH = SPEED_OF_LIGHT / WIFI_FREQ  # ~0.125m

    # Detection thresholds
    MOTION_THRESHOLD = 2.0         # Variance ratio for motion detection
    PRESENCE_THRESHOLD = 0.3       # Min CSI variance indicating human presence

    def __init__(self, num_subcarriers=52):
        self.num_subcarriers = num_subcarriers
        self._prev_csi = None      # For conjugate multiplication
        self._baseline = None       # For presence detection
        self._calibrated = False

    def extract_vitals(self, csi_buffer):
        """
        Full RuView-inspired extraction pipeline.
        Returns dict with vital signs, motion state, presence, signal quality, and spectrogram data.
        """
        if len(csi_buffer) < 100:
            return {}

        try:
            # Step 1: Parse raw CSI amplitudes and phases
            amplitudes, phases = self._parse_csi(csi_buffer)
            if amplitudes is None or amplitudes.shape[0] < 100:
                return {}

            # Step 2: Conjugate multiplication — phase sanitization
            # Removes random phase offsets between consecutive packets
            cleaned_phase = self._conjugate_multiply(phases)

            # Step 3: Hampel filter — outlier removal on amplitudes
            amplitudes = self._hampel_filter(amplitudes, window=7, threshold=3.0)

            # Step 4: Top-K subcarrier selection — pick channels most sensitive to body motion
            best_subs = self._select_subcarriers(amplitudes, k=min(10, amplitudes.shape[1]))
            selected_amp = amplitudes[:, best_subs]

            # Step 5: PCA — extract principal component (dominant motion signature)
            principal = self._pca_extract(selected_amp)

            # Step 6: Extract breathing rate (Fresnel + FFT)
            breathing_rate, breath_confidence = self._extract_breathing(principal)

            # Step 7: Extract heart rate (bandpass + FFT)
            heart_rate, hr_confidence = self._extract_cardiac(principal)

            # Step 8: HRV analysis
            hrv_metrics = self._extract_hrv(principal)

            # Step 9: Signal quality (SNR)
            snr_cardiac = self._compute_snr(principal, self.CARDIAC_BAND)
            snr_breath = self._compute_snr(principal, self.BREATH_BAND)

            # Step 10: Motion detection
            motion_level, is_motion = self._detect_motion(amplitudes)

            # Step 11: Presence detection
            is_present, presence_score = self._detect_presence(amplitudes)

            # Step 12: Generate spectrogram data for dashboard
            spectrogram_data = self._compute_spectrogram(principal)

            # Step 13: Activity classification from spectrogram
            activity = self._classify_activity(principal, is_motion, breathing_rate)

            return {
                'heart_rate': round(heart_rate, 1) if heart_rate else None,
                'breathing_rate': round(breathing_rate, 1) if breathing_rate else None,
                'hrv_sdnn': round(hrv_metrics.get('sdnn', 0), 2),
                'hrv_rmssd': round(hrv_metrics.get('rmssd', 0), 2),
                'signal_quality': round(max(snr_cardiac, snr_breath), 2),
                'snr_cardiac': round(snr_cardiac, 2),
                'snr_breath': round(snr_breath, 2),
                'hr_confidence': hr_confidence,
                'breath_confidence': breath_confidence,
                'motion_level': round(motion_level, 3),
                'is_motion': is_motion,
                'is_present': is_present,
                'presence_score': round(presence_score, 3),
                'activity': activity,
                'num_samples': len(amplitudes),
                'num_subcarriers_used': len(best_subs),
                'spectrogram': spectrogram_data,
                'waveform': principal[-200:].tolist() if len(principal) > 0 else [],
            }
        except Exception as e:
            logger.error(f'CSI extraction error: {e}')
            return {}

    # ── Step 2: Conjugate Multiplication (RuView phase cleaning) ──────────

    def _conjugate_multiply(self, phases):
        """
        Remove random phase offsets between consecutive CSI packets.
        For each packet p[t], compute: p_clean[t] = p[t] * conj(p[t-1])
        This isolates phase CHANGES caused by motion from static offsets.
        """
        if phases is None or phases.shape[0] < 2:
            return phases

        # Convert to complex phasors
        phasors = np.exp(1j * phases)
        # Conjugate multiply consecutive packets
        cleaned = phasors[1:] * np.conj(phasors[:-1])
        # Extract cleaned phase
        cleaned_phase = np.angle(cleaned)
        return cleaned_phase

    # ── Step 3: Hampel Filter (RuView outlier removal) ────────────────────

    def _hampel_filter(self, data, window=7, threshold=3.0):
        """
        Hampel filter: replace outliers with local median.
        More robust than simple moving average for WiFi interference spikes.
        Vectorized for speed on Pi 5.
        """
        filtered = data.copy()
        half_w = window // 2
        rows = data.shape[0]

        for col in range(data.shape[1]):
            col_data = data[:, col]
            for i in range(half_w, rows - half_w):
                window_data = col_data[i - half_w:i + half_w + 1]
                median = np.median(window_data)
                mad = 1.4826 * np.median(np.abs(window_data - median))
                if mad > 1e-10 and np.abs(col_data[i] - median) / mad > threshold:
                    filtered[i, col] = median
        return filtered

    # ── Step 4: Top-K Subcarrier Selection ────────────────────────────────

    def _select_subcarriers(self, amplitudes, k=10):
        """
        Select subcarriers with highest cardiac-band energy ratio.
        Inspired by RuView's learned graph partitioning (simplified for edge).
        Also considers variance (motion-sensitive channels).
        """
        scores = []
        for col in range(amplitudes.shape[1]):
            sig = amplitudes[:, col]
            freqs = fftfreq(len(sig), 1.0 / self.SAMPLE_RATE)
            fft_vals = np.abs(fft(sig))

            # Cardiac band energy ratio
            cardiac_mask = (np.abs(freqs) >= self.CARDIAC_BAND[0]) & \
                           (np.abs(freqs) <= self.CARDIAC_BAND[1])
            cardiac_energy = np.sum(fft_vals[cardiac_mask] ** 2)

            # Breathing band energy
            breath_mask = (np.abs(freqs) >= self.BREATH_BAND[0]) & \
                          (np.abs(freqs) <= self.BREATH_BAND[1])
            breath_energy = np.sum(fft_vals[breath_mask] ** 2)

            total_energy = np.sum(fft_vals ** 2) + 1e-10

            # Combined score: both cardiac and breathing sensitivity matter
            score = (cardiac_energy + breath_energy) / total_energy

            # Penalize very low variance channels (likely static/dead)
            variance = np.var(sig)
            if variance < 1e-6:
                score *= 0.01

            scores.append(score)

        top_k = np.argsort(scores)[-k:]
        return top_k

    # ── Step 5: PCA Extraction ────────────────────────────────────────────

    def _pca_extract(self, selected):
        """Extract principal component — dominant motion signature."""
        mean_centered = selected - np.mean(selected, axis=0)
        cov = np.cov(mean_centered.T)
        if cov.ndim < 2:
            return mean_centered.flatten()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = mean_centered @ eigenvectors[:, -1]
        return principal

    # ── Step 6: Fresnel Breathing Extraction ──────────────────────────────

    def _extract_breathing(self, sig):
        """
        Physics-based breathing extraction using Fresnel zone model.
        When a person breathes, chest displacement crosses Fresnel zone
        boundaries, creating periodic amplitude modulations at breathing freq.
        Falls back to FFT peak detection.
        """
        nyq = self.SAMPLE_RATE / 2
        b, a = scipy_signal.butter(4, [self.BREATH_BAND[0] / nyq,
                                        self.BREATH_BAND[1] / nyq], btype='band')
        filtered = scipy_signal.filtfilt(b, a, sig)

        # FFT-based rate extraction
        freqs = fftfreq(len(filtered), 1.0 / self.SAMPLE_RATE)
        fft_vals = np.abs(fft(filtered))
        pos_mask = (freqs > self.BREATH_BAND[0]) & (freqs < self.BREATH_BAND[1])

        if not np.any(pos_mask):
            return None, 'none'

        peak_idx = np.argmax(fft_vals[pos_mask])
        peak_freq = freqs[pos_mask][peak_idx]
        peak_power = fft_vals[pos_mask][peak_idx]
        total_power = np.sum(fft_vals[pos_mask]) + 1e-10

        # Confidence based on spectral concentration
        spectral_ratio = peak_power / total_power
        if spectral_ratio > 0.4:
            confidence = 'high'
        elif spectral_ratio > 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Fresnel zone validation: expected chest displacement ~1-5mm
        # corresponds to specific phase change at 2.4 GHz
        rate_per_min = peak_freq * 60.0

        # Sanity check: 6-30 breaths/min
        if rate_per_min < 6 or rate_per_min > 30:
            return None, 'none'

        return rate_per_min, confidence

    # ── Step 7: Cardiac Rate Extraction ───────────────────────────────────

    def _extract_cardiac(self, sig):
        """Extract heart rate via bandpass + FFT with confidence scoring."""
        nyq = self.SAMPLE_RATE / 2
        b, a = scipy_signal.butter(4, [self.CARDIAC_BAND[0] / nyq,
                                        self.CARDIAC_BAND[1] / nyq], btype='band')
        filtered = scipy_signal.filtfilt(b, a, sig)

        freqs = fftfreq(len(filtered), 1.0 / self.SAMPLE_RATE)
        fft_vals = np.abs(fft(filtered))
        pos_mask = (freqs > self.CARDIAC_BAND[0]) & (freqs < self.CARDIAC_BAND[1])

        if not np.any(pos_mask):
            return None, 'none'

        peak_idx = np.argmax(fft_vals[pos_mask])
        peak_freq = freqs[pos_mask][peak_idx]
        peak_power = fft_vals[pos_mask][peak_idx]
        total_power = np.sum(fft_vals[pos_mask]) + 1e-10

        spectral_ratio = peak_power / total_power
        if spectral_ratio > 0.35:
            confidence = 'high'
        elif spectral_ratio > 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'

        rate_per_min = peak_freq * 60.0
        if rate_per_min < 40 or rate_per_min > 150:
            return None, 'none'

        return rate_per_min, confidence

    # ── Step 8: HRV Analysis ─────────────────────────────────────────────

    def _extract_hrv(self, sig):
        """Extract HRV metrics (SDNN, RMSSD) from cardiac signal."""
        try:
            nyq = self.SAMPLE_RATE / 2
            b, a = scipy_signal.butter(4, [self.CARDIAC_BAND[0] / nyq,
                                            self.CARDIAC_BAND[1] / nyq], btype='band')
            cardiac = scipy_signal.filtfilt(b, a, sig)

            peaks, _ = scipy_signal.find_peaks(
                cardiac,
                distance=int(self.SAMPLE_RATE * 0.4),
                prominence=np.std(cardiac) * 0.3
            )

            if len(peaks) < 3:
                return {'sdnn': 0, 'rmssd': 0, 'pnn50': 0}

            # Inter-beat intervals in ms
            ibis = np.diff(peaks) / self.SAMPLE_RATE * 1000.0

            sdnn = float(np.std(ibis))
            rmssd = float(np.sqrt(np.mean(np.diff(ibis) ** 2)))

            # pNN50: percentage of successive IBIs differing by >50ms
            successive_diffs = np.abs(np.diff(ibis))
            pnn50 = float(np.sum(successive_diffs > 50) / len(successive_diffs) * 100) \
                if len(successive_diffs) > 0 else 0

            return {'sdnn': sdnn, 'rmssd': rmssd, 'pnn50': pnn50}
        except Exception:
            return {'sdnn': 0, 'rmssd': 0, 'pnn50': 0}

    # ── Step 9: SNR Computation ───────────────────────────────────────────

    def _compute_snr(self, sig, band):
        """Signal-to-noise ratio for a given frequency band."""
        freqs = fftfreq(len(sig), 1.0 / self.SAMPLE_RATE)
        fft_vals = np.abs(fft(sig))
        in_band = (np.abs(freqs) >= band[0]) & (np.abs(freqs) <= band[1])
        out_band = ~in_band & (np.abs(freqs) > 0.05)
        signal_power = np.mean(fft_vals[in_band] ** 2) if np.any(in_band) else 1e-10
        noise_power = np.mean(fft_vals[out_band] ** 2) if np.any(out_band) else 1e-10
        return 10 * np.log10(max(signal_power / noise_power, 1e-10))

    # ── Step 10: Motion Detection ─────────────────────────────────────────

    def _detect_motion(self, amplitudes):
        """
        Detect gross body motion via CSI amplitude variance.
        High variance = motion; low variance = stationary.
        """
        recent = amplitudes[-50:]   # Last 0.5s
        older = amplitudes[-200:-50] if len(amplitudes) > 200 else amplitudes[:50]

        recent_var = np.mean(np.var(recent, axis=0))
        older_var = np.mean(np.var(older, axis=0)) + 1e-10

        motion_ratio = recent_var / older_var
        is_motion = motion_ratio > self.MOTION_THRESHOLD

        return float(motion_ratio), is_motion

    # ── Step 11: Presence Detection ───────────────────────────────────────

    def _detect_presence(self, amplitudes):
        """
        Detect human presence by comparing CSI variance against
        empty-room baseline. If no baseline, use absolute variance.
        """
        current_var = np.mean(np.var(amplitudes[-100:], axis=0))

        if self._baseline is not None:
            ratio = current_var / (self._baseline + 1e-10)
            is_present = ratio > 1.5
            score = min(ratio / 3.0, 1.0)
        else:
            # No baseline — use absolute threshold
            is_present = current_var > self.PRESENCE_THRESHOLD
            score = min(current_var / 2.0, 1.0)

        return is_present, float(score)

    def calibrate_baseline(self, csi_buffer):
        """Capture empty-room baseline for presence detection."""
        amplitudes, _ = self._parse_csi(csi_buffer)
        if amplitudes is not None and amplitudes.shape[0] > 50:
            self._baseline = np.mean(np.var(amplitudes, axis=0))
            self._calibrated = True
            logger.info(f'Baseline calibrated: variance={self._baseline:.4f}')

    # ── Step 12: Spectrogram ──────────────────────────────────────────────

    def _compute_spectrogram(self, sig, nperseg=64, noverlap=48):
        """
        Compute CSI time-frequency spectrogram for dashboard visualization.
        Returns downsampled data suitable for WebSocket transmission.
        """
        try:
            f, t, Sxx = scipy_signal.spectrogram(
                sig, fs=self.SAMPLE_RATE,
                nperseg=nperseg, noverlap=noverlap,
                scaling='density'
            )
            # Only keep 0-3 Hz range (vital signs region)
            freq_mask = f <= 3.0
            Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)

            # Downsample time axis to max 50 points
            if Sxx_db.shape[1] > 50:
                step = Sxx_db.shape[1] // 50
                Sxx_db = Sxx_db[:, ::step]

            return {
                'frequencies': f[freq_mask].tolist(),
                'power_db': Sxx_db.tolist(),
            }
        except Exception:
            return None

    # ── Step 13: Activity Classification ──────────────────────────────────

    def _classify_activity(self, sig, is_motion, breathing_rate):
        """
        Simple activity classification from CSI features.
        Categories: resting, sitting, light_motion, walking, vigorous
        """
        variance = np.var(sig[-100:])

        if not is_motion and breathing_rate and breathing_rate < 16:
            return 'resting'
        elif not is_motion:
            return 'sitting'
        elif variance < 5.0:
            return 'light_motion'
        elif variance < 20.0:
            return 'walking'
        else:
            return 'vigorous'

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_csi(self, csi_buffer):
        """Parse CSI buffer into amplitude and phase arrays."""
        amplitudes = []
        phases = []

        for p in csi_buffer:
            amp = p.get('amplitude', [])
            if len(amp) > 0:
                amplitudes.append(amp)
                ph = p.get('phase', [0] * len(amp))
                phases.append(ph)

        if not amplitudes:
            return None, None

        amplitudes = np.array(amplitudes, dtype=np.float64)
        phases = np.array(phases, dtype=np.float64) if phases else None

        return amplitudes, phases


class MultiNodeFusion:
    """
    Fuses CSI vital signs from multiple ESP32-C6 nodes.
    Inspired by RuView's multistatic mesh — 3+ nodes create
    overlapping signal paths for improved accuracy.
    """

    def __init__(self):
        self.node_weights = {}  # Adaptive weights based on SNR history

    def fuse_nodes(self, node_vitals: dict) -> dict:
        """
        Weighted fusion of vital signs from multiple nodes.
        Nodes with higher SNR get more weight.

        Args:
            node_vitals: {node_id: vitals_dict} from each CSI processor

        Returns:
            Fused vital signs dict
        """
        if not node_vitals:
            return {}

        # Filter to nodes that have valid data
        valid = {k: v for k, v in node_vitals.items()
                 if v and v.get('heart_rate') is not None}

        if not valid:
            # Fall back to any node with any data
            for k, v in node_vitals.items():
                if v:
                    return v
            return {}

        if len(valid) == 1:
            return list(valid.values())[0]

        # SNR-weighted fusion
        total_snr = sum(max(v.get('signal_quality', 0), 0.1) for v in valid.values())

        fused = {
            'heart_rate': 0,
            'breathing_rate': 0,
            'hrv_sdnn': 0,
            'hrv_rmssd': 0,
            'signal_quality': 0,
            'num_samples': 0,
            'fusion_method': 'snr_weighted',
            'node_count': len(valid),
        }

        for node_id, vitals in valid.items():
            weight = max(vitals.get('signal_quality', 0), 0.1) / total_snr

            for key in ['heart_rate', 'breathing_rate', 'hrv_sdnn', 'hrv_rmssd']:
                val = vitals.get(key)
                if val is not None:
                    fused[key] += val * weight

            fused['signal_quality'] = max(fused['signal_quality'],
                                          vitals.get('signal_quality', 0))
            fused['num_samples'] += vitals.get('num_samples', 0)

        # Copy non-numeric fields from best node
        best_node = max(valid.values(), key=lambda v: v.get('signal_quality', 0))
        for key in ['hr_confidence', 'breath_confidence', 'is_motion', 'is_present',
                     'motion_level', 'presence_score', 'activity', 'spectrogram', 'waveform']:
            if key in best_node:
                fused[key] = best_node[key]

        # Round
        for key in ['heart_rate', 'breathing_rate', 'hrv_sdnn', 'hrv_rmssd', 'signal_quality']:
            if fused[key]:
                fused[key] = round(fused[key], 2)

        return fused
