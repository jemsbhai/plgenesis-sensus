"""
Sensus — Contactless Health Sensing Platform
=============================================
PL_Genesis Hackathon 2026 Demo Dashboard

Interactive Streamlit application with animated room visualization
showing WiFi CSI sensing in action across 30 clinical scenarios.

Usage:
  streamlit run demo/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pi', 'services'))

from simulator import (
    build_all_scenarios, create_engine, get_scenarios_by_category,
    ScenarioEngine, AlertLevel
)

try:
    from classifier import get_classifier
    _classifier_available = True
except Exception:
    _classifier_available = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'integrations'))

try:
    from impulse_inference import impulse_classify_sync
    _impulse_available = True
except Exception:
    _impulse_available = False

try:
    from hypercerts import HypercertGenerator
    from filecoin_store import FilecoinHealthStore
    from storacha_store import StorachaHealthStore
    _integrations_available = True
except Exception:
    _integrations_available = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sensus — Contactless Health Sensing",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — MINIMAL (heavy styling is inside the HTML component)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 0.5rem; }

/* Remove default streamlit padding for full-width HTML */
.stHtml { width: 100%; }
iframe { border: none !important; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid #1a2744;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #8294b4 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

# Train classifier once (cached)
@st.cache_resource(show_spinner="🧠 Training ML classifier on 30 scenarios...")
def load_classifier():
    if not _classifier_available:
        return None
    clf = get_classifier()
    return clf

clf = load_classifier()

if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'ml_prediction' not in st.session_state:
    st.session_state.ml_prediction = None
if 'impulse_prediction' not in st.session_state:
    st.session_state.impulse_prediction = None
if 'impulse_last_call' not in st.session_state:
    st.session_state.impulse_last_call = 0
if 'session_states' not in st.session_state:
    st.session_state.session_states = []
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'session_states' not in st.session_state:
    st.session_state.session_states = []
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []
if 'br_history' not in st.session_state:
    st.session_state.br_history = []
if 'hrv_history' not in st.session_state:
    st.session_state.hrv_history = []
if 'waveform_history' not in st.session_state:
    st.session_state.waveform_history = []
if 'last_state' not in st.session_state:
    st.session_state.last_state = None
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = 1
if 'speed' not in st.session_state:
    st.session_state.speed = 1.0

MAX_HISTORY = 200


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:16px; padding-top:8px;">
        <div style="font-size:32px; font-weight:900; background:linear-gradient(135deg,#3b82f6,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">SENSUS</div>
        <div style="font-size:10px; color:#8294b4; letter-spacing:2px; text-transform:uppercase;">
            Virtual Scenario Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🎯 Select Scenario")

    scenarios_by_cat = get_scenarios_by_category()
    all_scenarios = build_all_scenarios()

    categories = list(scenarios_by_cat.keys())
    selected_cat = st.selectbox("Category", ["All"] + categories, label_visibility="collapsed")

    if selected_cat == "All":
        available = all_scenarios
    else:
        available = {s.id: s for s in scenarios_by_cat[selected_cat]}

    scenario_options = {f"{s.id:02d}. {s.name}": s.id for s in available.values()}
    selected_name = st.selectbox("Scenario", list(scenario_options.keys()), label_visibility="collapsed")
    selected_id = scenario_options[selected_name]
    scenario = all_scenarios[selected_id]

    st.markdown(f"""
    <div style="background:#111b2e; border:1px solid #1a2744; border-radius:10px; padding:12px; font-size:12px;">
        <div style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:10px;font-weight:700;
                    letter-spacing:0.5px;background:#3b82f618;border:1px solid #3b82f633;color:#93c5fd;
                    text-transform:uppercase;">{scenario.category}</div>
        <div style="color:#8294b4; margin-top:8px; line-height:1.6;">{scenario.description}</div>
        <div style="color:#4b5e80; margin-top:6px; font-size:11px;">
            <strong style="color:#f59e0b;">Clinical:</strong> {scenario.clinical_relevance}
        </div>
        <div style="color:#4b5e80; margin-top:4px; font-size:11px;">
            ⏱ {scenario.total_duration_sec:.0f}s · {len(scenario.phases)} phases · {'🔁 loops' if scenario.loop else '▶ single run'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚡ Playback")

    speed = st.select_slider("Speed", options=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                              value=st.session_state.speed, format_func=lambda x: f"{x}x")
    st.session_state.speed = speed

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Play", use_container_width=True, type="primary"):
            st.session_state.engine = create_engine(selected_id, speed=speed)
            st.session_state.selected_scenario = selected_id
            st.session_state.running = True
            st.session_state.hr_history = []
            st.session_state.br_history = []
            st.session_state.hrv_history = []
            st.session_state.waveform_history = []
            st.session_state.last_state = None
    with col2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False
    with col3:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            st.session_state.engine = None
            st.session_state.last_state = None
            st.session_state.hr_history = []
            st.session_state.br_history = []
            st.session_state.hrv_history = []
            st.session_state.waveform_history = []

    st.markdown("---")
    st.markdown("##### 📋 Phases")
    for i, phase in enumerate(scenario.phases):
        is_current = (st.session_state.last_state and st.session_state.last_state.get('phase_index') == i)
        marker = "▶" if is_current else "○"
        color = "#3b82f6" if is_current else "#4b5e80"
        weight = "700" if is_current else "400"
        st.markdown(f'<div style="font-size:12px;color:{color};padding:2px 0;font-weight:{weight};">'
                    f'{marker} {phase.name} <span style="color:#4b5e80;">({phase.duration_sec:.0f}s)</span></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🏆 Tech Stack")
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:4px;">
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#3b82f610;border:1px solid #3b82f633;color:#93c5fd;">WiFi CSI</span>
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#10b98110;border:1px solid #10b98133;color:#6ee7b7;">ESP32-C6</span>
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#8b5cf610;border:1px solid #8b5cf633;color:#c4b5fd;">Fresnel Model</span>
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#f59e0b10;border:1px solid #f59e0b33;color:#fcd34d;">PCA + FFT</span>
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#ef444410;border:1px solid #ef444433;color:#fca5a5;">HRV Analysis</span>
        <span style="padding:3px 8px;border-radius:5px;font-size:9px;font-weight:700;background:#06b6d410;border:1px solid #06b6d433;color:#67e8f9;">Sensor Fusion</span>
    </div>
    """, unsafe_allow_html=True)

    # Impulse AI Query Button
    if _impulse_available:
        st.markdown("---")
        st.markdown("##### ⚡ Impulse AI Inference")
        if st.button("🔮 Query Impulse AI", use_container_width=True, type="primary",
                     disabled=not st.session_state.running):
            if st.session_state.last_state:
                with st.spinner("Querying Impulse AI..."):
                    result = impulse_classify_sync(st.session_state.last_state)
                    if result and result.get('predicted_class'):
                        st.session_state.impulse_prediction = result
                        st.success(f"Impulse AI: {result['predicted_class']} ({int(result.get('confidence',0)*100)}%)")
                    else:
                        st.warning("No response from Impulse AI")
        if st.session_state.impulse_prediction:
            ip = st.session_state.impulse_prediction
            st.markdown(f"""
            <div style="background:#f59e0b10;border:1px solid #f59e0b33;border-radius:8px;padding:10px;margin-top:8px;">
                <div style="font-size:9px;color:#f59e0b;font-weight:700;text-transform:uppercase;">Last Impulse AI Result</div>
                <div style="font-size:18px;font-weight:800;color:#fcd34d;margin:4px 0;">{ip.get('predicted_class','--')}</div>
                <div style="font-size:11px;color:#4b5e80;">Confidence: {int(ip.get('confidence',0)*100)}%</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION STEP
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.running and st.session_state.engine:
    engine = st.session_state.engine
    engine.speed = st.session_state.speed
    state = engine.step(0.1)
    st.session_state.last_state = state

    st.session_state.hr_history.append(state['heart_rate'])
    st.session_state.br_history.append(state['breathing_rate'])
    st.session_state.hrv_history.append(state['hrv_rmssd'])

    node_csi = state.get('node_csi', {})
    if 'node_1' in node_csi:
        amp = node_csi['node_1']['amplitude']
        top_amp = sorted(amp, reverse=True)[:10]
        st.session_state.waveform_history.append(float(np.mean(top_amp)))

    # ML Prediction (local — every frame)
    if clf is not None and clf.is_trained:
        st.session_state.ml_prediction = clf.predict(state)
    else:
        st.session_state.ml_prediction = None

    # Impulse AI Prediction — triggered by button only (see sidebar)

    st.session_state.session_states.append(state)
    if len(st.session_state.session_states) > 500:
        st.session_state.session_states = st.session_state.session_states[-500:]

    for key in ['hr_history', 'br_history', 'hrv_history', 'waveform_history']:
        if len(st.session_state[key]) > MAX_HISTORY:
            st.session_state[key] = st.session_state[key][-MAX_HISTORY:]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE FULL-PAGE HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def _build_ml_panel(prediction):
    """Build the ML prediction panel HTML."""
    if prediction is None:
        return ''
    
    pred_class = prediction.get('predicted_class', 'Unknown')
    confidence = prediction.get('confidence', 0)
    risk_level = prediction.get('risk_level', 0)
    risk_label = prediction.get('risk_label', 'Unknown')
    probs = prediction.get('probabilities', {})

    risk_colors = {
        0: ('#10b981', '#10b98120'),
        1: ('#f59e0b', '#f59e0b20'),
        2: ('#ef4444', '#ef444420'),
        3: ('#dc2626', '#dc262630'),
    }
    color, bg = risk_colors.get(risk_level, ('#4b5e80', '#4b5e8020'))
    conf_pct = int(confidence * 100)

    # Probability bars
    prob_bars = ''
    for cls, prob in probs.items():
        bar_w = int(prob * 100)
        prob_bars += (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
            f'<span style="font-size:10px;color:#8294b4;width:80px;text-align:right;">{cls}</span>'
            f'<div style="flex:1;height:6px;background:#1a2744;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{bar_w}%;height:100%;background:{color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-size:10px;color:#4b5e80;width:30px;">{bar_w}%</span>'
            f'</div>'
        )

    return (
        f'<div class="panel" style="border-color:{color}44;">'
        f'<h3 style="color:{color};">\U0001f9e0 ML Prediction</h3>'
        f'<div style="text-align:center;margin:8px 0;">'
        f'<div style="font-size:20px;font-weight:800;color:{color};">{pred_class}</div>'
        f'<div style="font-size:11px;color:#4b5e80;margin-top:4px;">'
        f'Confidence: {conf_pct}% · Risk: {risk_label}</div>'
        f'</div>'
        f'<div style="background:{bg};border:1px solid {color}33;border-radius:6px;'
        f'padding:6px 10px;margin:8px 0;">'
        f'<div style="height:4px;background:#1a2744;border-radius:2px;overflow:hidden;">'
        f'<div style="width:{conf_pct}%;height:100%;background:{color};border-radius:2px;"></div>'
        f'</div>'
        f'</div>'
        f'{prob_bars}'
        f'<div style="font-size:9px;color:#4b5e80;margin-top:6px;text-align:center;">'
        f'RandomForest · 30-scenario trained · Real-time inference</div>'
        f'</div>'
    )


def _build_impulse_panel(ml_prediction, impulse_prediction=None):
    """Build the Impulse AI integration panel."""
    has_key = bool(os.environ.get('IMPULSE_API_KEY') or os.environ.get('IMPSDK_API_KEY'))
    conn_color = '#10b981' if has_key else '#4b5e80'
    conn_text = 'Connected' if has_key else 'No API Key'
    conn_dot = 'on' if has_key else 'off'

    # Show LIVE Impulse AI API prediction
    pred_html = ''
    if impulse_prediction and impulse_prediction.get('predicted_class'):
        pred = impulse_prediction['predicted_class']
        conf = int(impulse_prediction.get('confidence', 0) * 100)
        reasoning = impulse_prediction.get('raw_response', '')[:120]
        # Clean markdown from reasoning
        reasoning = reasoning.replace('```json', '').replace('```', '').strip()
        pred_html = (
            f'<div style="background:#f59e0b10;border:1px solid #f59e0b33;border-radius:6px;padding:8px 10px;margin:8px 0;">'
            f'<div style="font-size:9px;color:#f59e0b;font-weight:700;text-transform:uppercase;letter-spacing:1px;">'
            f'\U0001f310 Live API Inference</div>'
            f'<div style="font-size:16px;font-weight:800;color:#fcd34d;margin:4px 0;">{pred}</div>'
            f'<div style="font-size:10px;color:#4b5e80;">{conf}% confidence</div>'
            f'</div>'
        )
    elif has_key:
        pred_html = (
            f'<div style="background:#111b2e;border-radius:6px;padding:6px 10px;margin:6px 0;">'
            f'<div style="font-size:10px;color:#4b5e80;">\U0001f310 Waiting for API response...</div>'
            f'</div>'
        )

    return (
        f'<div class="panel" style="border-color:#f59e0b44;">'
        f'<h3 style="color:#f59e0b;">⚡ Impulse AI</h3>'
        f'<div class="node-item">'
        f'<span class="node-dot {conn_dot}"></span>'
        f'<span style="font-weight:600;color:{conn_color};">{conn_text}</span>'
        f'<span style="color:#4b5e80;margin-left:auto;font-size:10px;">SDK v1</span>'
        f'</div>'
        f'<div style="margin-top:8px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#4b5e80;margin-bottom:3px;">'
        f'<span>Training Dataset</span><span style="color:#f0f4fc;font-weight:600;">18,180 samples</span></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#4b5e80;margin-bottom:3px;">'
        f'<span>Classes</span><span style="color:#f0f4fc;font-weight:600;">30 scenarios → 10 states</span></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#4b5e80;margin-bottom:3px;">'
        f'<span>Model Accuracy</span><span style="color:#10b981;font-weight:600;">98.8%</span></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#4b5e80;margin-bottom:3px;">'
        f'<span>F1 Score</span><span style="color:#10b981;font-weight:600;">0.9883</span></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#4b5e80;margin-bottom:3px;">'
        f'<span>Features</span><span style="color:#f0f4fc;font-weight:600;">15 (HR, BR, HRV, SpO2...)</span></div>'
        f'</div>'
        f'{pred_html}'
        f'<div style="display:flex;gap:4px;margin-top:8px;flex-wrap:wrap;">'
        f'<span style="padding:2px 6px;border-radius:4px;font-size:8px;font-weight:700;'
        f'background:#f59e0b15;border:1px solid #f59e0b33;color:#fcd34d;">CLASSIFICATION</span>'
        f'<span style="padding:2px 6px;border-radius:4px;font-size:8px;font-weight:700;'
        f'background:#ef444415;border:1px solid #ef444433;color:#fca5a5;">ANOMALY DETECT</span>'
        f'<span style="padding:2px 6px;border-radius:4px;font-size:8px;font-weight:700;'
        f'background:#3b82f615;border:1px solid #3b82f633;color:#93c5fd;">TIME-SERIES</span>'
        f'</div>'
        f'<div style="font-size:9px;color:#4b5e80;margin-top:6px;text-align:center;">'
        f'Powered by Impulse AI · impulselabs.ai</div>'
        f'</div>'
    )


def build_dashboard_html(state, hr_hist, br_hist, hrv_hist, wf_hist, ml_prediction=None, impulse_prediction=None):
    """Build the entire dashboard as a single HTML component."""

    if state is None:
        return build_idle_html()

    # Extract values safely
    hr = state.get('heart_rate', 0)
    br = state.get('breathing_rate', 0)
    hrv_rmssd = state.get('hrv_rmssd', 0)
    hrv_sdnn = state.get('hrv_sdnn', 0)
    stress = state.get('stress_index', 'low')
    spo2 = state.get('spo2', 98)
    sys_bp = state.get('blood_pressure_sys', 120)
    dia_bp = state.get('blood_pressure_dia', 80)
    gsr = state.get('gsr', 2.5)
    skin_temp = state.get('skin_temp', 36.5)
    activity = state.get('activity', 'unknown').replace('_', ' ').title()
    motion = state.get('motion_level', 0)
    motion_type = state.get('motion_type', 'none')
    sig_q = state.get('signal_quality', 0)
    is_present = state.get('is_present', False)
    presence_score = state.get('presence_score', 0)
    irregular = state.get('irregular_rhythm', False)
    alert_level = state.get('alert_level', 'normal')
    alert_msg = state.get('alert_message', '')
    hr_conf = state.get('hr_confidence', 'high')
    br_conf = state.get('breath_confidence', 'high')

    scenario_name = state.get('scenario_name', '')
    phase_name = state.get('phase_name', '')
    phase_idx = state.get('phase_index', 0)
    total_phases = state.get('total_phases', 1)
    elapsed = state.get('elapsed_sec', 0)
    total_dur = state.get('total_duration_sec', 1)
    spd = state.get('speed', 1.0)
    overall_pct = (elapsed % total_dur) / total_dur * 100 if total_dur > 0 else 0

    env = state.get('environment', {})
    temp_c = env.get('temperature_c', 22)
    humidity = env.get('humidity_pct', 45)
    co2 = env.get('co2_ppm', 450)
    tvoc = env.get('tvoc_ppb', 120)
    light = env.get('light_lux', 300)
    noise = env.get('noise_db', 35)

    # Alert styling
    alert_html = ""
    if alert_level == "critical":
        alert_html = f'<div class="alert-banner critical">⚠ {alert_msg}</div>'
    elif alert_level == "warning":
        alert_html = f'<div class="alert-banner warning">⚠ {alert_msg}</div>'
    elif alert_level == "info" and alert_msg:
        alert_html = f'<div class="alert-banner info">ℹ {alert_msg}</div>'

    # Card alert classes
    hr_alert = ' critical' if (hr > 140 or (0 < hr < 40)) else ' warning' if (hr > 110 or (0 < hr < 50)) else ''
    br_alert = ' critical' if (br > 30 or (0 < br < 6)) else ' warning' if (br > 24 or (0 < br < 8)) else ''
    spo2_alert = ' critical' if spo2 < 90 else ' warning' if spo2 < 94 else ''
    bp_alert = ' critical' if (sys_bp > 180 or (0 < sys_bp < 80)) else ' warning' if (sys_bp > 150 or (0 < sys_bp < 90)) else ''
    stress_color = '#ef4444' if stress == 'high' else '#f59e0b' if stress == 'moderate' else '#10b981'

    hr_display = f"{hr:.0f}" if hr > 0 else "--"
    br_display = f"{br:.0f}" if br > 0 else "--"

    # Confidence badge helper
    def conf(level):
        colors = {'high': ('#10b98130', '#6ee7b7'), 'medium': ('#f59e0b30', '#fbbf24'),
                  'low': ('#ef444430', '#fca5a5'), 'none': ('#4b5e8030', '#4b5e80')}
        bg, fg = colors.get(level, colors['none'])
        lbl = level if level != 'none' else 'NO SIGNAL'
        return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;text-transform:uppercase;background:{bg};color:{fg};">{lbl}</span>'

    # Multi-person
    num_people = state.get('num_people', 1)

    # Serialize chart data
    wf_json = json.dumps(wf_hist[-200:]) if wf_hist else "[]"
    hr_json = json.dumps(hr_hist[-200:]) if hr_hist else "[]"
    br_json = json.dumps(br_hist[-200:]) if br_hist else "[]"
    hrv_json = json.dumps(hrv_hist[-200:]) if hrv_hist else "[]"

    # Presence indicator
    presence_cls = "present" if is_present else "absent"
    presence_icon = "🧍" if is_present else "👤"
    presence_label = "Person Detected" if is_present else "No Presence"
    presence_color = "#10b981" if is_present else "#4b5e80"

    # Irregular badge
    irreg_html = ""
    if irregular:
        irreg_html = '<div class="irreg-badge">⚡ Irregular Rhythm Detected</div>'

    # Motion type visual parameters for room animation
    person_anim = "none"
    if motion_type == "seizure":
        person_anim = "seizure"
    elif motion_type == "thrashing":
        person_anim = "thrash"
    elif motion_type == "fall":
        person_anim = "fall"
    elif motion_type == "tremor":
        person_anim = "tremor"
    elif motion_type == "walking":
        person_anim = "walk"
    elif motion_type == "fidget":
        person_anim = "fidget"

    # Heart animation speed (pulse CSS)
    heart_anim_dur = f"{60.0/max(hr,30):.2f}s" if hr > 0 else "0s"
    breath_anim_dur = f"{60.0/max(br,3):.2f}s" if br > 0 else "0s"

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',system-ui,sans-serif; background:#06090f; color:#f0f4fc; overflow-x:hidden; }}

/* ─── HEADER ─── */
.header {{
    background:linear-gradient(180deg,#0c1222,#06090f);
    padding:14px 24px; display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid #1a2744;
}}
.brand {{ display:flex; align-items:center; gap:14px; }}
.brand-icon {{
    width:42px; height:42px; border-radius:11px;
    background:linear-gradient(135deg,#3b82f6,#06b6d4);
    display:flex; align-items:center; justify-content:center;
    font-size:20px; font-weight:900; color:#fff;
    box-shadow:0 0 24px #3b82f644;
}}
.brand-title {{
    font-size:26px; font-weight:800;
    background:linear-gradient(135deg,#3b82f6,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.brand-sub {{ font-size:11px; color:#4b5e80; letter-spacing:2px; text-transform:uppercase; }}
.status {{
    display:flex; align-items:center; gap:8px; padding:6px 16px;
    border-radius:20px; background:#10b98118; border:1px solid #10b98133;
    font-size:12px; font-weight:600; color:#10b981;
}}
.status-dot {{
    width:8px; height:8px; border-radius:50%; background:#10b981;
    animation:blink 2s infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

/* ─── ALERT ─── */
.alert-banner {{
    padding:10px 24px; font-size:13px; font-weight:600;
    display:flex; align-items:center; gap:8px;
}}
.alert-banner.critical {{
    background:linear-gradient(90deg,#7f1d1d,#991b1b); color:#fca5a5;
    animation:alertPulse 1.5s infinite;
}}
.alert-banner.warning {{ background:linear-gradient(90deg,#78350f,#92400e); color:#fcd34d; }}
.alert-banner.info {{ background:linear-gradient(90deg,#1e3a5f,#1e40af); color:#93c5fd; }}
@keyframes alertPulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.85}} }}

/* ─── SCENARIO BAR ─── */
.scenario-bar {{
    background:#111b2e; border-bottom:1px solid #1a2744;
    padding:8px 24px; display:flex; justify-content:space-between; align-items:center;
    font-size:12px;
}}
.phase-progress {{
    height:3px; background:#1a2744; border-radius:2px; overflow:hidden; margin-top:2px;
}}
.phase-fill {{
    height:100%; border-radius:2px;
    background:linear-gradient(90deg,#3b82f6,#06b6d4);
    transition:width 0.3s;
}}

/* ─── MAIN GRID ─── */
.main {{
    display:grid; grid-template-columns:1fr 380px; gap:14px;
    padding:14px 24px; min-height:calc(100vh - 140px);
}}
@media(max-width:1200px) {{ .main {{ grid-template-columns:1fr; }} }}

/* ─── VITAL CARDS ─── */
.vitals-grid {{
    display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
}}
.v-card {{
    background:#0c1222; border:1px solid #1a2744; border-radius:12px;
    padding:14px 16px; position:relative; overflow:hidden;
    transition:border-color 0.3s, box-shadow 0.3s;
}}
.v-card:hover {{ border-color:#2563eb44; box-shadow:0 0 20px #2563eb11; }}
.v-card .label {{ font-size:10px; color:#4b5e80; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; margin-bottom:6px; }}
.v-card .value {{ font-size:32px; font-weight:800; line-height:1; letter-spacing:-1px; }}
.v-card .unit {{ font-size:13px; font-weight:400; color:#8294b4; }}
.v-card .sub {{ font-size:10px; color:#4b5e80; margin-top:6px; }}
.v-card::before {{ content:''; position:absolute; top:0; left:16px; right:16px; height:2px; border-radius:0 0 2px 2px; }}
.v-card.hr::before {{ background:#ef4444; }}
.v-card.rr::before {{ background:#06b6d4; }}
.v-card.hrv::before {{ background:#8b5cf6; }}
.v-card.stress::before {{ background:#f59e0b; }}
.v-card.spo2::before {{ background:#3b82f6; }}
.v-card.bp::before {{ background:#10b981; }}
.v-card.warning {{ border-color:#f59e0b !important; box-shadow:0 0 15px #f59e0b22; }}
.v-card.critical {{ border-color:#ef4444 !important; box-shadow:0 0 20px #ef444433; animation:cardCrit 1.5s infinite; }}
@keyframes cardCrit {{ 0%,100%{{box-shadow:0 0 20px #ef444422}} 50%{{box-shadow:0 0 35px #ef444455}} }}

/* ─── ROOM VISUALIZATION ─── */
.room-viz {{
    background:#0c1222; border:1px solid #1a2744; border-radius:12px;
    padding:16px; margin-top:10px; position:relative; overflow:hidden;
}}
.room-viz h3 {{
    font-size:10px; color:#4b5e80; text-transform:uppercase;
    letter-spacing:1.5px; font-weight:700; margin-bottom:10px;
}}
.room-svg {{ width:100%; height:auto; }}

/* ─── WIFI WAVES ─── */
@keyframes wifiPulse {{
    0% {{ r:8; opacity:0.8; }}
    100% {{ r:55; opacity:0; }}
}}
.wifi-wave {{
    fill:none; stroke:#3b82f6; stroke-width:1.5; opacity:0;
    animation:wifiPulse 2s infinite;
}}
.wifi-wave.w2 {{ animation-delay:0.5s; }}
.wifi-wave.w3 {{ animation-delay:1.0s; }}
.wifi-wave.w4 {{ animation-delay:1.5s; }}

/* Signal paths */
@keyframes signalFlow {{
    0% {{ stroke-dashoffset:20; opacity:0.1; }}
    50% {{ opacity:0.5; }}
    100% {{ stroke-dashoffset:0; opacity:0.1; }}
}}
.signal-path {{
    stroke:#3b82f6; stroke-width:1; fill:none;
    stroke-dasharray:4 4; opacity:0.2;
    animation:signalFlow 1.5s linear infinite;
}}
.signal-path.p2 {{ stroke:#06b6d4; animation-delay:0.3s; }}
.signal-path.p3 {{ stroke:#8b5cf6; animation-delay:0.6s; }}

/* ─── PERSON ANIMATIONS ─── */
@keyframes heartbeat {{
    0%,100% {{ transform:scale(1); }}
    15% {{ transform:scale(1.15); }}
    30% {{ transform:scale(1); }}
    45% {{ transform:scale(1.08); }}
    60% {{ transform:scale(1); }}
}}
@keyframes breathe {{
    0%,100% {{ transform:scaleY(1); }}
    50% {{ transform:scaleY(1.06); }}
}}
@keyframes seizureShake {{
    0%,100% {{ transform:translate(0px,0px) rotate(0deg); }}
    10% {{ transform:translate(-3px,1px) rotate(-1.5deg); }}
    20% {{ transform:translate(2px,-1px) rotate(1deg); }}
    30% {{ transform:translate(-2px,2px) rotate(-1deg); }}
    40% {{ transform:translate(3px,-1px) rotate(1.5deg); }}
    50% {{ transform:translate(-1px,1px) rotate(-1deg); }}
    60% {{ transform:translate(2px,-2px) rotate(1deg); }}
    70% {{ transform:translate(-3px,1px) rotate(-0.5deg); }}
    80% {{ transform:translate(1px,-1px) rotate(1deg); }}
    90% {{ transform:translate(-2px,2px) rotate(-1deg); }}
}}
@keyframes thrash {{
    0%,100% {{ transform:translate(0px,0px) rotate(0deg); }}
    25% {{ transform:translate(-4px,2px) rotate(-2deg); }}
    50% {{ transform:translate(3px,-2px) rotate(1.5deg); }}
    75% {{ transform:translate(-3px,1px) rotate(-1deg); }}
}}
@keyframes fallDown {{
    0% {{ transform:translate(0px,0px) rotate(0deg); }}
    40% {{ transform:translate(0px,0px) rotate(0deg); }}
    70% {{ transform:translate(5px,8px) rotate(25deg); }}
    100% {{ transform:translate(8px,14px) rotate(40deg); }}
}}
@keyframes tremor {{
    0%,100% {{ transform:translate(0px,0px); }}
    25% {{ transform:translate(-0.7px,0.3px); }}
    50% {{ transform:translate(0.7px,-0.3px); }}
    75% {{ transform:translate(-0.3px,0.7px); }}
}}
@keyframes fidgetMove {{
    0%,100% {{ transform:translate(0px,0px); }}
    30% {{ transform:translate(0.5px,-0.5px); }}
    60% {{ transform:translate(-0.5px,0.3px); }}
}}
@keyframes walkBob {{
    0%,100% {{ transform:translateY(0px); }}
    50% {{ transform:translateY(-2px); }}
}}

.person-group {{
    animation-fill-mode:forwards;
}}
.person-group.seizure {{ animation:seizureShake 0.18s linear infinite; }}
.person-group.thrash {{ animation:thrash 0.5s ease-in-out infinite; }}
.person-group.fall {{ animation:fallDown 2.5s ease-in forwards; }}
.person-group.tremor {{ animation:tremor 0.12s linear infinite; }}
.person-group.fidget {{ animation:fidgetMove 2.5s ease-in-out infinite; }}
.person-group.walk {{ animation:walkBob 0.6s ease-in-out infinite; }}

.heart-indicator {{
    transform-origin:center center;
    animation:heartbeat {heart_anim_dur} infinite;
}}
.chest-group {{
    transform-origin:center center;
    animation:breathe {breath_anim_dur} infinite;
}}

/* Alert border glow on room */
.room-viz.critical {{ border-color:#ef4444; }}
.room-viz.warning {{ border-color:#f59e0b66; }}

/* ─── CHART AREA ─── */
.chart-panel {{
    background:#0c1222; border:1px solid #1a2744; border-radius:12px;
    padding:14px 16px; margin-top:10px;
}}
.chart-panel h3 {{
    font-size:10px; color:#4b5e80; text-transform:uppercase;
    letter-spacing:1.5px; font-weight:700; margin-bottom:8px;
}}
.chart-canvas {{ width:100%; height:110px; }}

/* ─── ENV STRIP ─── */
.env-strip {{
    display:flex; gap:6px; flex-wrap:wrap; margin-top:10px;
}}
.env-chip {{
    background:#0c1222; border:1px solid #1a2744; border-radius:8px;
    padding:8px 12px; font-size:11px; display:flex; align-items:center;
    gap:6px; flex:1; min-width:100px;
}}
.env-chip .lbl {{ color:#4b5e80; font-weight:600; }}
.env-chip .val {{ color:#f0f4fc; font-weight:700; margin-left:auto; }}

/* ─── SIDEBAR PANEL ─── */
.side-panel {{
    display:flex; flex-direction:column; gap:10px;
}}
.panel {{
    background:#0c1222; border:1px solid #1a2744; border-radius:12px; padding:14px 16px;
}}
.panel h3 {{
    font-size:10px; color:#4b5e80; text-transform:uppercase;
    letter-spacing:1.5px; font-weight:700; margin-bottom:10px;
}}
.presence-ring {{
    width:80px; height:80px; border-radius:50%; margin:0 auto 10px;
    display:flex; align-items:center; justify-content:center; font-size:32px;
    transition:all 0.5s;
}}
.presence-ring.present {{
    background:#10b98118; border:3px solid #10b981; box-shadow:0 0 30px #10b98122;
}}
.presence-ring.absent {{ background:#4b5e8018; border:3px solid #4b5e80; }}

.node-item {{
    display:flex; align-items:center; gap:8px; padding:6px 10px;
    border-radius:6px; background:#111b2e; font-size:11px; margin-bottom:3px;
}}
.node-dot {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
.node-dot.on {{ background:#10b981; box-shadow:0 0 6px #10b98144; }}
.node-dot.off {{ background:#4b5e80; }}

.irreg-badge {{
    background:#ef444418; border:1px solid #ef444433; border-radius:6px;
    padding:6px 10px; text-align:center; font-size:11px; font-weight:700;
    color:#fca5a5; text-transform:uppercase; letter-spacing:0.5px; margin-top:8px;
}}

.mini-chart {{ width:100%; height:50px; }}

/* ─── FOOTER ─── */
.footer {{
    text-align:center; padding:12px; color:#4b5e80; font-size:10px;
    letter-spacing:0.5px; border-top:1px solid #1a2744; margin-top:10px;
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="brand">
        <div class="brand-icon">S</div>
        <div>
            <div class="brand-title">SENSUS</div>
            <div class="brand-sub">Contactless Multi-Modal Health Sensing</div>
        </div>
    </div>
    <div class="status"><div class="status-dot"></div>Virtual Mode — Live</div>
</div>

<!-- ALERT -->
{alert_html}

<!-- SCENARIO BAR -->
<div class="scenario-bar">
    <div>
        <span style="color:#3b82f6;font-weight:700;">{scenario_name}</span>
        <span style="color:#4b5e80;margin:0 6px;">›</span>
        <span style="color:#f0f4fc;font-weight:600;">{phase_name}</span>
        <span style="color:#4b5e80;margin-left:6px;">Phase {phase_idx+1}/{total_phases}</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
        <span style="color:#4b5e80;">{elapsed:.1f}s / {total_dur:.0f}s</span>
        <span style="color:#f59e0b;font-weight:700;">{spd}x</span>
    </div>
</div>
<div class="phase-progress"><div class="phase-fill" style="width:{overall_pct:.1f}%;"></div></div>

<!-- MAIN -->
<div class="main">
<div>
    <!-- VITALS -->
    <div class="vitals-grid">
        <div class="v-card hr{hr_alert}">
            <div class="label">Heart Rate</div>
            <div class="value" style="color:#f0f4fc;">{hr_display}<span class="unit">bpm</span></div>
            <div class="sub">{conf(hr_conf)}</div>
        </div>
        <div class="v-card rr{br_alert}">
            <div class="label">Breathing Rate</div>
            <div class="value" style="color:#f0f4fc;">{br_display}<span class="unit">/min</span></div>
            <div class="sub">{conf(br_conf)}</div>
        </div>
        <div class="v-card hrv">
            <div class="label">HRV (RMSSD)</div>
            <div class="value" style="color:#f0f4fc;">{hrv_rmssd:.1f}<span class="unit">ms</span></div>
            <div class="sub">SDNN: {hrv_sdnn:.1f} ms</div>
        </div>
        <div class="v-card stress">
            <div class="label">Stress Index</div>
            <div class="value" style="color:{stress_color};font-size:26px;">{stress.upper()}</div>
            <div class="sub">GSR: {gsr:.1f} μS</div>
        </div>
        <div class="v-card spo2{spo2_alert}">
            <div class="label">SpO₂</div>
            <div class="value" style="color:#f0f4fc;">{spo2:.0f}<span class="unit">%</span></div>
        </div>
        <div class="v-card bp{bp_alert}">
            <div class="label">Blood Pressure</div>
            <div class="value" style="color:#f0f4fc;font-size:26px;">{sys_bp:.0f}/{dia_bp:.0f}<span class="unit">mmHg</span></div>
            <div class="sub">Skin: {skin_temp:.1f}°C</div>
        </div>
    </div>

    <!-- ROOM VISUALIZATION -->
    <div class="room-viz {alert_level if alert_level in ('critical','warning') else ''}">
        <h3>Sensing Environment — WiFi CSI Mesh Active</h3>
        <svg viewBox="0 0 700 280" class="room-svg" xmlns="http://www.w3.org/2000/svg">
            <!-- Room background -->
            <rect x="10" y="10" width="680" height="260" rx="8" fill="#080c18" stroke="#1a2744" stroke-width="1"/>

            <!-- Grid dots -->
            <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                <circle cx="15" cy="15" r="0.5" fill="#1a274444"/>
            </pattern>
            <rect x="10" y="10" width="680" height="260" fill="url(#grid)"/>

            <!-- Signal paths between nodes and person -->
            <line x1="80" y1="60" x2="{'240' if num_people > 1 else '350'}" y2="140" class="signal-path"/>
            <line x1="620" y1="60" x2="{'240' if num_people > 1 else '350'}" y2="140" class="signal-path p2"/>
            <line x1="350" y1="250" x2="{'240' if num_people > 1 else '350'}" y2="140" class="signal-path p3"/>

            <!-- Fresnel zone ellipses (subtle) -->
            <ellipse cx="215" cy="100" rx="80" ry="25" fill="none" stroke="#3b82f6" stroke-width="0.5" opacity="0.1"/>
            <ellipse cx="485" cy="100" rx="80" ry="25" fill="none" stroke="#06b6d4" stroke-width="0.5" opacity="0.1"/>

            <!-- NODE 1 (top-left) -->
            <g transform="translate(80,60)">
                <circle r="6" fill="#3b82f6" opacity="0.9"/>
                <circle class="wifi-wave"/>
                <circle class="wifi-wave w2"/>
                <circle class="wifi-wave w3"/>
                <text y="-14" text-anchor="middle" fill="#8294b4" font-size="9" font-weight="600">NODE 1</text>
                <text y="20" text-anchor="middle" fill="#4b5e80" font-size="7">C6 · TX</text>
            </g>

            <!-- NODE 2 (top-right) -->
            <g transform="translate(620,60)">
                <circle r="6" fill="#06b6d4" opacity="0.9"/>
                <circle class="wifi-wave" style="stroke:#06b6d4"/>
                <circle class="wifi-wave w2" style="stroke:#06b6d4"/>
                <circle class="wifi-wave w3" style="stroke:#06b6d4"/>
                <text y="-14" text-anchor="middle" fill="#8294b4" font-size="9" font-weight="600">NODE 2</text>
                <text y="20" text-anchor="middle" fill="#4b5e80" font-size="7">C6 · RX-A</text>
            </g>

            <!-- NODE 3 (bottom-center) -->
            <g transform="translate(350,250)">
                <circle r="6" fill="#8b5cf6" opacity="0.9"/>
                <circle class="wifi-wave" style="stroke:#8b5cf6"/>
                <circle class="wifi-wave w2" style="stroke:#8b5cf6"/>
                <circle class="wifi-wave w3" style="stroke:#8b5cf6"/>
                <text y="18" text-anchor="middle" fill="#8294b4" font-size="9" font-weight="600">NODE 3</text>
            </g>

            <!-- ENV SENSOR (bottom-left) -->
            <g transform="translate(80,230)">
                <rect x="-10" y="-6" width="20" height="12" rx="3" fill="#f59e0b" opacity="0.6"/>
                <text y="-12" text-anchor="middle" fill="#4b5e80" font-size="7">ENV</text>
            </g>

            <!-- PERSON (center or left for multi-person) -->
            <g transform="translate({'240' if num_people > 1 else '350'},130)">
            <g class="person-group {person_anim}">
                <!-- Chair -->
                {'<rect x="-25" y="5" width="50" height="40" rx="4" fill="#1a2744" stroke="#2a3a5a" stroke-width="0.5"/>' if motion_type != 'walking' else ''}

                <!-- Body (chest area for breathing) -->
                <g class="chest-group">
                    <!-- Torso -->
                    <ellipse cx="0" cy="0" rx="14" ry="20" fill="#2a4a7a" opacity="0.7"/>
                    <!-- Head -->
                    <circle cx="0" cy="-28" r="10" fill="#3a5a8a" opacity="0.8"/>
                    <!-- Arms -->
                    <line x1="-14" y1="-5" x2="-28" y2="15" stroke="#3a5a8a" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
                    <line x1="14" y1="-5" x2="28" y2="15" stroke="#3a5a8a" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
                </g>

                <!-- Heart indicator -->
                <g class="heart-indicator" style="{'animation-duration:'+heart_anim_dur if hr > 0 else 'animation:none'}">
                    <circle cx="-3" cy="-5" r="4" fill="#ef4444" opacity="0.7"/>
                    <text x="-3" y="-2" text-anchor="middle" fill="#fff" font-size="5">♥</text>
                </g>

                <!-- Vital readout below person -->
                <text x="0" y="55" text-anchor="middle" fill="#8294b4" font-size="9" font-weight="600">
                    {'♥ '+str(int(hr))+' bpm' if hr > 0 else '♥ --'}
                </text>
                <text x="0" y="66" text-anchor="middle" fill="#4b5e80" font-size="8">
                    {'🫁 '+str(int(br))+'/min' if br > 0 else '🫁 --'}
                </text>

                {'<text x="0" y="-45" text-anchor="middle" fill="#ef4444" font-size="10" font-weight="800">⚠ ALERT</text>' if alert_level == 'critical' else ''}
            </g>
            </g>

            {'<!-- PERSON 2 (right side) -->' if num_people > 1 else ''}
            {'<g transform="translate(520,135)"><g><rect x="-22" y="5" width="44" height="36" rx="4" fill="#1a2744" stroke="#2a3a5a" stroke-width="0.5"/><g class="chest-group" style="animation-duration:5s;"><ellipse cx="0" cy="0" rx="12" ry="18" fill="#1a5a4a" opacity="0.7"/><circle cx="0" cy="-25" r="9" fill="#2a6a5a" opacity="0.8"/><line x1="-12" y1="-5" x2="-24" y2="12" stroke="#2a6a5a" stroke-width="3.5" stroke-linecap="round" opacity="0.6"/><line x1="12" y1="-5" x2="24" y2="12" stroke="#2a6a5a" stroke-width="3.5" stroke-linecap="round" opacity="0.6"/></g><g class="heart-indicator" style="animation-duration:0.97s;"><circle cx="-3" cy="-5" r="3.5" fill="#06b6d4" opacity="0.7"/><text x="-3" y="-2" text-anchor="middle" fill="#fff" font-size="4">♥</text></g><text x="0" y="52" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="600">♥ 62 bpm</text><text x="0" y="62" text-anchor="middle" fill="#4b5e80" font-size="7">🫁 12/min</text></g></g>' if num_people > 1 else ''}

            {'<!-- Extra signal paths for person 2 -->' if num_people > 1 else ''}
            {'<line x1="620" y1="60" x2="520" y2="135" class="signal-path" style="stroke:#10b981;"/><line x1="350" y1="250" x2="520" y2="135" class="signal-path p2" style="stroke:#10b981;"/>' if num_people > 1 else ''}

            <!-- Status text -->
            <text x="350" y="24" text-anchor="middle" fill="#4b5e80" font-size="8" letter-spacing="2">
                ACTIVITY: {activity.upper()}
            </text>

            <!-- Router icon (top center) -->
            <g transform="translate(350,45)">
                <rect x="-12" y="-5" width="24" height="10" rx="2" fill="#1a2744" stroke="#2a3a5a" stroke-width="0.5"/>
                <line x1="-6" y1="-5" x2="-10" y2="-14" stroke="#2a3a5a" stroke-width="1"/>
                <line x1="6" y1="-5" x2="10" y2="-14" stroke="#2a3a5a" stroke-width="1"/>
                <text y="18" text-anchor="middle" fill="#4b5e80" font-size="6">TP-Link AX1500</text>
            </g>
        </svg>
    </div>

    <!-- WAVEFORM CHART -->
    <div class="chart-panel">
        <h3>WiFi CSI Signal — Real-Time Processing Pipeline</h3>
        <canvas id="wfChart" class="chart-canvas"></canvas>
    </div>

    <!-- VITALS TREND -->
    <div class="chart-panel">
        <h3>Vital Signs Trend</h3>
        <canvas id="trendChart" style="width:100%;height:160px;"></canvas>
    </div>

    <!-- ENV STRIP -->
    <div class="env-strip">
        <div class="env-chip">🌡 <span class="lbl">Temp</span><span class="val">{temp_c:.1f}°C</span></div>
        <div class="env-chip">💧 <span class="lbl">Humid</span><span class="val">{humidity:.0f}%</span></div>
        <div class="env-chip">☁ <span class="lbl">CO₂</span><span class="val">{co2:.0f} ppm</span></div>
        <div class="env-chip">🍃 <span class="lbl">TVOC</span><span class="val">{tvoc:.0f} ppb</span></div>
        <div class="env-chip">💡 <span class="lbl">Light</span><span class="val">{light:.0f} lux</span></div>
        <div class="env-chip">🔊 <span class="lbl">Noise</span><span class="val">{noise:.0f} dB</span></div>
    </div>
</div>

<!-- RIGHT SIDEBAR -->
<div class="side-panel">
    <!-- Presence -->
    <div class="panel" style="text-align:center;">
        <h3>Presence Detection</h3>
        <div class="presence-ring {presence_cls}">{presence_icon}</div>
        <div style="font-size:14px;font-weight:700;color:{presence_color};">{presence_label}</div>
        <div style="font-size:11px;color:#4b5e80;margin-top:4px;">Score: {presence_score:.3f}</div>
    </div>

    <!-- Activity -->
    <div class="panel" style="text-align:center;">
        <h3>Activity Classification</h3>
        <div style="font-size:22px;font-weight:800;margin:6px 0;">{activity}</div>
        <div style="font-size:11px;color:#4b5e80;">Motion: {motion:.2f} · Signal: {sig_q:.1f} dB</div>
        {irreg_html}
    </div>

    <!-- Nodes -->
    <div class="panel">
        <h3>Sensor Mesh</h3>
        <div class="node-item"><span class="node-dot on"></span><span style="font-weight:600;">node_1</span><span style="color:#4b5e80;margin-left:auto;font-size:10px;">C6 · CSI TX</span></div>
        <div class="node-item"><span class="node-dot on"></span><span style="font-weight:600;">node_2</span><span style="color:#4b5e80;margin-left:auto;font-size:10px;">C6 · CSI RX-A</span></div>
        <div class="node-item"><span class="node-dot on"></span><span style="font-weight:600;">node_3</span><span style="color:#4b5e80;margin-left:auto;font-size:10px;">C6 · CSI RX-B</span></div>
        <div class="node-item"><span class="node-dot on"></span><span style="font-weight:600;">env_sensor</span><span style="color:#4b5e80;margin-left:auto;font-size:10px;">Giga · ENV</span></div>
        <div class="node-item"><span class="node-dot off"></span><span style="font-weight:600;">audio_1</span><span style="color:#4b5e80;margin-left:auto;font-size:10px;">S3 · Audio</span></div>
    </div>

    <!-- Clinical Note -->
    <div class="panel" style="border-color:#1e3a5f;">
        <h3 style="color:#3b82f6;">📋 Clinical Context</h3>
        <div style="font-size:12px;color:#8294b4;line-height:1.7;">
            {state.get('phase_description', scenario.phases[phase_idx].description if phase_idx < len(scenario.phases) else '')}
        </div>
    </div>

    <!-- ML PREDICTION -->
    {_build_ml_panel(ml_prediction)}

    <!-- IMPULSE AI -->
    {_build_impulse_panel(ml_prediction, impulse_prediction)}
</div>
</div>

<!-- FOOTER -->
<div class="footer">
    SENSUS v2.0 — PL_Genesis Hackathon 2026 — WiFi CSI + Sensor Fusion — ESP32-C6 Mesh + Raspberry Pi 5 — ML by Impulse AI
</div>

<!-- CHARTS via Canvas API (lightweight, no library needed) -->
<script>
function drawWaveform() {{
    const canvas = document.getElementById('wfChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;

    const data = {wf_json};
    if (data.length < 2) return;

    const min = Math.min(...data), max = Math.max(...data);
    const range = max - min || 1;

    // Background
    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(26,39,68,0.3)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {{
        const y = (i / 4) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }}

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < data.length; i++) {{
        const x = (i / (data.length - 1)) * w;
        const y = h - ((data[i] - min) / range) * h * 0.85 - h * 0.05;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Fill
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(59,130,246,0.06)';
    ctx.fill();
}}

function drawTrend() {{
    const canvas = document.getElementById('trendChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;

    const hrData = {hr_json};
    const brData = {br_json};
    const hrvData = {hrv_json};

    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, w, h);

    function drawLine(data, color, minV, maxV) {{
        if (data.length < 2) return;
        const range = maxV - minV || 1;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < data.length; i++) {{
            const x = (i / (data.length - 1)) * w;
            const y = h - ((data[i] - minV) / range) * h * 0.28 - h * 0.02;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
    }}

    function drawLineInBand(data, color, band, bandIdx) {{
        if (data.length < 2) return;
        const bh = h / 3;
        const yOff = bandIdx * bh;
        const min = Math.min(...data) - 2;
        const max = Math.max(...data) + 2;
        const range = max - min || 1;

        // Band separator
        if (bandIdx > 0) {{
            ctx.strokeStyle = 'rgba(26,39,68,0.4)';
            ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(0, yOff); ctx.lineTo(w, yOff); ctx.stroke();
        }}

        // Label
        ctx.fillStyle = color;
        ctx.font = '600 9px Inter, sans-serif';
        ctx.fillText(band, 4, yOff + 12);

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < data.length; i++) {{
            const x = (i / (data.length - 1)) * w;
            const y = yOff + bh - ((data[i] - min) / range) * (bh - 18) - 4;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();

        // Latest value
        if (data.length > 0) {{
            const last = data[data.length - 1];
            ctx.fillStyle = '#f0f4fc';
            ctx.font = '700 10px Inter, sans-serif';
            ctx.fillText(last.toFixed(1), w - 40, yOff + 12);
        }}
    }}

    drawLineInBand(hrData, '#ef4444', 'HR', 0);
    drawLineInBand(brData, '#06b6d4', 'BR', 1);
    drawLineInBand(hrvData, '#8b5cf6', 'HRV', 2);
}}

drawWaveform();
drawTrend();
</script>
</body></html>"""


def build_idle_html():
    """Build idle state HTML."""
    return """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',sans-serif; background:#06090f; color:#f0f4fc; }
.header {
    background:linear-gradient(180deg,#0c1222,#06090f);
    padding:14px 24px; display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid #1a2744;
}
.brand { display:flex; align-items:center; gap:14px; }
.brand-icon {
    width:42px; height:42px; border-radius:11px;
    background:linear-gradient(135deg,#3b82f6,#06b6d4);
    display:flex; align-items:center; justify-content:center;
    font-size:20px; font-weight:900; color:#fff;
    box-shadow:0 0 24px #3b82f644;
}
.brand-title {
    font-size:26px; font-weight:800;
    background:linear-gradient(135deg,#3b82f6,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.brand-sub { font-size:11px; color:#4b5e80; letter-spacing:2px; text-transform:uppercase; }

@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes wifiPulse {
    0% { r:10; opacity:0.6; }
    100% { r:60; opacity:0; }
}
.wifi-wave { fill:none; stroke:#3b82f6; stroke-width:1.5; opacity:0; animation:wifiPulse 2s infinite; }
.wifi-wave.w2 { animation-delay:0.7s; }
.wifi-wave.w3 { animation-delay:1.4s; }
</style>
</head><body>
<div class="header">
    <div class="brand">
        <div class="brand-icon">S</div>
        <div>
            <div class="brand-title">SENSUS</div>
            <div class="brand-sub">Contactless Multi-Modal Health Sensing</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;padding:6px 16px;border-radius:20px;
                background:#4b5e8018;border:1px solid #4b5e8033;font-size:12px;color:#4b5e80;">
        ● Standby
    </div>
</div>
<div style="text-align:center; padding:60px 20px;">
    <svg width="200" height="200" viewBox="0 0 200 200" style="animation:float 3s ease-in-out infinite;">
        <!-- Central person -->
        <circle cx="100" cy="70" r="15" fill="#2a4a7a" opacity="0.7"/>
        <ellipse cx="100" cy="105" rx="18" ry="28" fill="#2a4a7a" opacity="0.5"/>
        <!-- Heart -->
        <circle cx="97" cy="98" r="5" fill="#ef4444" opacity="0.5"/>
        <text x="97" y="101" text-anchor="middle" fill="#fff" font-size="6">♥</text>
        <!-- WiFi waves from 3 points -->
        <g transform="translate(30,50)">
            <circle r="4" fill="#3b82f6" opacity="0.8"/>
            <circle class="wifi-wave"/>
            <circle class="wifi-wave w2"/>
        </g>
        <g transform="translate(170,50)">
            <circle r="4" fill="#06b6d4" opacity="0.8"/>
            <circle class="wifi-wave" style="stroke:#06b6d4"/>
            <circle class="wifi-wave w2" style="stroke:#06b6d4"/>
        </g>
        <g transform="translate(100,180)">
            <circle r="4" fill="#8b5cf6" opacity="0.8"/>
            <circle class="wifi-wave" style="stroke:#8b5cf6"/>
            <circle class="wifi-wave w2" style="stroke:#8b5cf6"/>
        </g>
        <!-- Signal lines -->
        <line x1="30" y1="50" x2="100" y2="100" stroke="#3b82f6" stroke-width="0.5" opacity="0.2" stroke-dasharray="4 4"/>
        <line x1="170" y1="50" x2="100" y2="100" stroke="#06b6d4" stroke-width="0.5" opacity="0.2" stroke-dasharray="4 4"/>
        <line x1="100" y1="180" x2="100" y2="100" stroke="#8b5cf6" stroke-width="0.5" opacity="0.2" stroke-dasharray="4 4"/>
    </svg>
    <div style="font-size:24px; font-weight:700; color:#f0f4fc; margin-top:20px;">
        Select a scenario and press Play
    </div>
    <div style="font-size:14px; color:#8294b4; max-width:550px; margin:12px auto 0; line-height:1.7;">
        Sensus uses WiFi Channel State Information (CSI) to extract vital signs
        contactlessly. Choose from <strong style="color:#3b82f6;">30 clinical scenarios</strong> to see how
        the system detects everything from normal resting to cardiac arrest —
        all without touching the patient.
    </div>
    <div style="display:flex;justify-content:center;gap:8px;margin-top:24px;flex-wrap:wrap;">
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">🫀 Cardiac Events</span>
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">🫁 Respiratory</span>
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">🧠 Neuro/Stress</span>
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">💊 Medication</span>
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">🚨 Emergency</span>
        <span style="padding:5px 12px;border-radius:6px;font-size:11px;font-weight:600;background:#0c1222;border:1px solid #1a2744;color:#8294b4;">🌡 Environmental</span>
    </div>
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════

state = st.session_state.last_state
html = build_dashboard_html(
    state,
    st.session_state.hr_history,
    st.session_state.br_history,
    st.session_state.hrv_history,
    st.session_state.waveform_history,
    st.session_state.ml_prediction,
    st.session_state.impulse_prediction,
)

components.html(html, height=1200, scrolling=True)

# Auto-refresh
if st.session_state.running:
    time.sleep(0.1)
    st.rerun()
