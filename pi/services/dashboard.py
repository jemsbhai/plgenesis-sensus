"""
Sensus Web Dashboard — PL_Genesis 2026
====================================
Real-time contactless health monitoring dashboard.
Features: CSI waveform viz, vital signs, AI interpretation, node status.
Inspired by RuView's sensing visualization.
"""

import json
import os
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import threading
import logging
import time

logger = logging.getLogger('sensus.dashboard')

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sensus — Contactless Health Sensing</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #06090f;
  --bg-card: #0c1222;
  --bg-elevated: #111b2e;
  --border: #1a2744;
  --border-glow: #2563eb33;
  --text-primary: #f0f4fc;
  --text-secondary: #8294b4;
  --text-muted: #4b5e80;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-green: #10b981;
  --accent-amber: #f59e0b;
  --accent-red: #ef4444;
  --accent-purple: #8b5cf6;
  --gradient-main: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',system-ui,-apple-system,sans-serif; background:var(--bg-primary); color:var(--text-primary); min-height:100vh; overflow-x:hidden; }

/* ─── HEADER ─── */
.header { background:linear-gradient(180deg, #0c1222 0%, var(--bg-primary) 100%); padding:16px 24px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); position:sticky; top:0; z-index:100; backdrop-filter:blur(12px); }
.brand { display:flex; align-items:center; gap:12px; }
.brand-icon { width:36px; height:36px; border-radius:10px; background:var(--gradient-main); display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:800; color:#fff; box-shadow:0 0 20px #3b82f644; }
.brand h1 { font-size:22px; font-weight:700; background:var(--gradient-main); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.5px; }
.brand-sub { font-size:11px; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:1px; }
.header-right { display:flex; align-items:center; gap:16px; }
.status-badge { display:flex; align-items:center; gap:8px; padding:6px 14px; border-radius:20px; background:#10b98118; border:1px solid #10b98133; font-size:12px; font-weight:500; color:var(--accent-green); }
.status-badge .dot { width:8px; height:8px; border-radius:50%; background:var(--accent-green); animation:blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.status-badge.disconnected { background:#ef444418; border-color:#ef444433; color:var(--accent-red); }
.status-badge.disconnected .dot { background:var(--accent-red); animation:none; }
.node-count { font-size:12px; color:var(--text-muted); }

/* ─── ALERT BANNER ─── */
.alert-banner { background:linear-gradient(90deg,#7f1d1d,#991b1b); color:#fca5a5; padding:10px 24px; text-align:center; font-size:13px; font-weight:600; display:none; letter-spacing:0.5px; }
.alert-banner.active { display:block; animation:alertSlide 0.3s ease; }
@keyframes alertSlide { from{transform:translateY(-100%)} to{transform:translateY(0)} }

/* ─── CONTROLS BAR ─── */
.controls { display:flex; gap:10px; padding:12px 24px; background:var(--bg-card); border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; }
.ctrl-label { color:var(--text-muted); font-size:11px; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-right:6px; }
.toggle-btn { padding:7px 14px; border-radius:8px; border:1px solid var(--border); background:var(--bg-elevated); color:var(--text-secondary); cursor:pointer; font-size:12px; font-weight:500; font-family:inherit; transition:all 0.2s; }
.toggle-btn:hover { border-color:var(--accent-blue); }
.toggle-btn.on { background:#065f46; border-color:var(--accent-green); color:#6ee7b7; box-shadow:0 0 12px #10b98122; }
.ask-btn { background:linear-gradient(135deg,#1e3a5f,#1e40af); border:1px solid #3b82f6; color:#93c5fd; font-weight:600; }
.ask-btn:hover { box-shadow:0 0 16px #3b82f633; }

/* ─── MAIN LAYOUT ─── */
.main { display:grid; grid-template-columns:1fr 320px; gap:16px; padding:16px 24px; max-width:1600px; margin:0 auto; }
@media(max-width:1100px) { .main { grid-template-columns:1fr; } }

/* ─── VITALS GRID ─── */
.vitals-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
@media(max-width:800px) { .vitals-grid { grid-template-columns:repeat(2,1fr); } }

.v-card { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; position:relative; overflow:hidden; transition:border-color 0.3s, box-shadow 0.3s; }
.v-card:hover { border-color:#2563eb44; box-shadow:0 0 20px #2563eb11; }
.v-card .label { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:10px; }
.v-card .value { font-size:38px; font-weight:800; color:var(--text-primary); line-height:1; letter-spacing:-1px; }
.v-card .unit { font-size:14px; font-weight:400; color:var(--text-secondary); margin-left:3px; }
.v-card .sub { font-size:11px; color:var(--text-muted); margin-top:8px; }
.v-card .confidence { display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-top:6px; }
.conf-high { background:#10b98120; color:#6ee7b7; }
.conf-medium { background:#f59e0b20; color:#fbbf24; }
.conf-low { background:#ef444420; color:#fca5a5; }

.v-card.warning { border-color:var(--accent-amber); }
.v-card.critical { border-color:var(--accent-red); box-shadow:0 0 20px #ef444422; animation:critPulse 1.5s infinite; }
@keyframes critPulse { 0%,100%{box-shadow:0 0 20px #ef444422} 50%{box-shadow:0 0 30px #ef444444} }

/* Glow bar at top of vital card */
.v-card::before { content:''; position:absolute; top:0; left:20px; right:20px; height:2px; border-radius:0 0 2px 2px; }
.v-card.hr::before { background:var(--accent-red); }
.v-card.rr::before { background:var(--accent-cyan); }
.v-card.hrv::before { background:var(--accent-purple); }
.v-card.stress::before { background:var(--accent-amber); }
.v-card.gsr::before { background:var(--accent-green); }
.v-card.activity::before { background:var(--accent-blue); }

/* ─── CSI WAVEFORM CHART ─── */
.chart-panel { grid-column:1/-1; background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; margin-top:4px; }
.chart-panel h3 { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:12px; }
.chart-container { position:relative; height:160px; }

/* ─── ENV BAR ─── */
.env-strip { grid-column:1/-1; display:flex; gap:10px; flex-wrap:wrap; }
.env-chip { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:10px 16px; font-size:12px; display:flex; align-items:center; gap:8px; flex:1; min-width:120px; }
.env-chip .icon { font-size:16px; }
.env-chip .lbl { color:var(--text-muted); font-weight:500; }
.env-chip .val { color:var(--text-primary); font-weight:700; margin-left:auto; }

/* ─── RIGHT SIDEBAR ─── */
.sidebar { display:flex; flex-direction:column; gap:12px; }

/* AI Panel */
.ai-panel { background:var(--bg-card); border:1px solid #1e3a5f; border-radius:14px; padding:18px 20px; flex:1; }
.ai-panel h3 { font-size:11px; color:var(--accent-blue); text-transform:uppercase; letter-spacing:1.5px; font-weight:700; margin-bottom:10px; display:flex; align-items:center; gap:8px; }
.ai-panel h3::before { content:''; width:8px; height:8px; border-radius:50%; background:var(--accent-blue); }
.ai-text { font-size:13px; line-height:1.7; color:var(--text-secondary); }

/* Node Status */
.nodes-panel { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; }
.nodes-panel h3 { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:12px; }
.node-list { display:flex; flex-direction:column; gap:8px; }
.node-item { display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:8px; background:var(--bg-elevated); font-size:12px; }
.node-dot { width:8px; height:8px; border-radius:50%; }
.node-dot.online { background:var(--accent-green); box-shadow:0 0 6px #10b98144; }
.node-dot.offline { background:var(--text-muted); }
.node-name { font-weight:600; color:var(--text-primary); }
.node-type { color:var(--text-muted); margin-left:auto; font-size:11px; }

/* Presence Panel */
.presence-panel { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; text-align:center; }
.presence-panel h3 { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:12px; }
.presence-indicator { width:80px; height:80px; border-radius:50%; margin:0 auto 12px; display:flex; align-items:center; justify-content:center; font-size:32px; transition:all 0.5s; }
.presence-indicator.present { background:#10b98118; border:2px solid var(--accent-green); box-shadow:0 0 30px #10b98122; }
.presence-indicator.absent { background:#4b5e8018; border:2px solid var(--text-muted); }
.presence-label { font-size:13px; font-weight:600; }
.presence-score { font-size:11px; color:var(--text-muted); margin-top:4px; }

/* ─── FOOTER ─── */
.footer { text-align:center; padding:20px; color:var(--text-muted); font-size:11px; letter-spacing:0.5px; border-top:1px solid var(--border); margin-top:16px; }
.footer a { color:var(--accent-blue); text-decoration:none; }

/* ─── PRIZE BADGES ─── */
.prize-strip { grid-column:1/-1; display:flex; gap:6px; flex-wrap:wrap; padding:4px 0; }
.prize-badge { padding:4px 10px; border-radius:6px; font-size:10px; font-weight:600; letter-spacing:0.5px; border:1px solid; }
.prize-badge.gemini { background:#4285f410; border-color:#4285f433; color:#8ab4f8; }
.prize-badge.eleven { background:#6366f110; border-color:#6366f133; color:#a5b4fc; }
.prize-badge.mongo { background:#10b98110; border-color:#10b98133; color:#6ee7b7; }
.prize-badge.snow { background:#06b6d410; border-color:#06b6d433; color:#67e8f9; }
.prize-badge.auth { background:#f59e0b10; border-color:#f59e0b33; color:#fcd34d; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="brand">
    <div class="brand-icon">S</div>
    <div>
      <h1>SENSUS</h1>
      <div class="brand-sub">Contactless Multi-Modal Health Sensing</div>
    </div>
  </div>
  <div class="header-right">
    <span class="node-count" id="nodeCount">0 sources</span>
    <div class="status-badge" id="statusBadge">
      <span class="dot" id="statusDot"></span>
      <span id="statusText">Connecting...</span>
    </div>
  </div>
</div>

<!-- ALERT BANNER -->
<div class="alert-banner" id="alertBanner"></div>

<!-- CONTROLS -->
<div class="controls">
  <span class="ctrl-label">AI Controls</span>
  <button class="toggle-btn" id="geminiToggle" onclick="toggleGemini()">Gemini: OFF</button>
  <button class="toggle-btn" id="elevenlabsToggle" onclick="toggleEL()">ElevenLabs: OFF</button>
  <button class="toggle-btn ask-btn" onclick="askGemini()">&#9889; Ask Gemini Now</button>
  <span style="margin-left:auto;font-size:11px;color:var(--text-muted)" id="clock"></span>
</div>

<!-- MAIN LAYOUT -->
<div class="main">
  <div>
    <!-- Prize Badges -->
    <div class="prize-strip">
      <span class="prize-badge gemini">Gemini API</span>
      <span class="prize-badge eleven">ElevenLabs TTS</span>
      <span class="prize-badge mongo">MongoDB Atlas</span>
      <span class="prize-badge snow">Snowflake</span>
      <span class="prize-badge auth">Auth0</span>
    </div>

    <!-- Vital Signs Grid -->
    <div class="vitals-grid">
      <div class="v-card hr" id="hrCard">
        <div class="label">Heart Rate</div>
        <div class="value" id="hrVal">--<span class="unit">bpm</span></div>
        <div class="sub">Ground truth: <span id="hrGT">--</span> bpm</div>
        <div class="confidence" id="hrConf"></div>
      </div>
      <div class="v-card rr" id="rrCard">
        <div class="label">Breathing Rate</div>
        <div class="value" id="rrVal">--<span class="unit">/min</span></div>
        <div class="confidence" id="rrConf"></div>
      </div>
      <div class="v-card hrv">
        <div class="label">HRV (RMSSD)</div>
        <div class="value" id="hrvVal">--<span class="unit">ms</span></div>
        <div class="sub">SDNN: <span id="hrvSdnn">--</span> ms | pNN50: <span id="hrvPnn50">--</span>%</div>
      </div>
      <div class="v-card stress">
        <div class="label">Stress Index</div>
        <div class="value" id="stressVal">--</div>
        <div class="sub">via HRV + GSR fusion</div>
      </div>
      <div class="v-card gsr">
        <div class="label">Skin Conductance</div>
        <div class="value" id="gsrVal">--<span class="unit">&micro;S</span></div>
      </div>
      <div class="v-card activity">
        <div class="label">Activity</div>
        <div class="value" id="activityVal" style="font-size:24px">--</div>
        <div class="sub">Motion: <span id="motionLevel">--</span> | Signal: <span id="sigQuality">--</span> dB</div>
      </div>
    </div>

    <!-- CSI Waveform -->
    <div class="chart-panel">
      <h3>WiFi CSI Waveform — Real-Time Signal Processing Pipeline</h3>
      <div class="chart-container">
        <canvas id="waveformChart"></canvas>
      </div>
    </div>

    <!-- Environmental Strip -->
    <div class="env-strip">
      <div class="env-chip"><span class="icon">&#127777;</span><span class="lbl">Temp</span><span class="val" id="envTemp">--°C</span></div>
      <div class="env-chip"><span class="icon">&#128167;</span><span class="lbl">Humidity</span><span class="val" id="envHum">--%</span></div>
      <div class="env-chip"><span class="icon">&#9729;</span><span class="lbl">CO₂</span><span class="val" id="envCo2">-- ppm</span></div>
      <div class="env-chip"><span class="icon">&#127811;</span><span class="lbl">TVOC</span><span class="val" id="envTvoc">-- ppb</span></div>
      <div class="env-chip"><span class="icon">&#128101;</span><span class="lbl">Occupancy</span><span class="val" id="envOcc">--</span></div>
      <div class="env-chip"><span class="icon">&#127897;</span><span class="lbl">Audio</span><span class="val" id="audioVal">quiet</span></div>
    </div>
  </div>

  <!-- RIGHT SIDEBAR -->
  <div class="sidebar">
    <!-- Presence Detection -->
    <div class="presence-panel">
      <h3>Presence Detection</h3>
      <div class="presence-indicator absent" id="presenceRing">&#128100;</div>
      <div class="presence-label" id="presenceLabel">No signal</div>
      <div class="presence-score" id="presenceScore">Score: --</div>
    </div>

    <!-- Node Status -->
    <div class="nodes-panel">
      <h3>Sensor Mesh</h3>
      <div class="node-list" id="nodeList">
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_1</span><span class="node-type">C6 · CSI</span></div>
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_2</span><span class="node-type">C6 · CSI</span></div>
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_3</span><span class="node-type">C6 · CSI</span></div>
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_s3_1</span><span class="node-type">S3 · Audio</span></div>
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_s3_2</span><span class="node-type">S3 · Audio</span></div>
        <div class="node-item"><span class="node-dot offline"></span><span class="node-name">node_s3_3</span><span class="node-type">S3 · Audio</span></div>
      </div>
    </div>

    <!-- AI Interpretation -->
    <div class="ai-panel">
      <h3>Gemini Health AI</h3>
      <div class="ai-text" id="aiText">Toggle Gemini above or click "Ask Gemini Now" for AI-powered clinical interpretation of real-time vital signs.</div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<div class="footer">
  SENSUS v1.0 &mdash; PL_Genesis 2026 &mdash; WiFi CSI + BLE + Environmental + Audio Fusion
  &mdash; Powered by <a href="#">ESP32-C6/S3 Mesh</a> + Raspberry Pi 5
</div>

<script>
// ─── SOCKET.IO ───
const socket = io();
let geminiOn = false, elOn = false;
const waveformData = [];
const MAX_WAVEFORM = 200;
let activeNodes = new Set();

// ─── CLOCK ───
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}, 1000);

// ─── CHART SETUP ───
const ctx = document.getElementById('waveformChart').getContext('2d');
const waveformChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: Array(MAX_WAVEFORM).fill(''),
    datasets: [{
      data: Array(MAX_WAVEFORM).fill(0),
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      borderWidth: 1.5,
      fill: true,
      pointRadius: 0,
      tension: 0.3,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    scales: {
      x: { display: false },
      y: { display: true, grid: { color: '#1a274422' }, ticks: { color: '#4b5e80', font: { size: 10 } } }
    },
    plugins: { legend: { display: false } },
    interaction: { enabled: false }
  }
});

// ─── TOGGLE FUNCTIONS ───
function toggleGemini() {
  geminiOn = !geminiOn;
  const btn = document.getElementById('geminiToggle');
  btn.textContent = 'Gemini: ' + (geminiOn ? 'ON' : 'OFF');
  btn.classList.toggle('on', geminiOn);
  fetch('/api/control/gemini', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:geminiOn})});
}
function toggleEL() {
  elOn = !elOn;
  const btn = document.getElementById('elevenlabsToggle');
  btn.textContent = 'ElevenLabs: ' + (elOn ? 'ON' : 'OFF');
  btn.classList.toggle('on', elOn);
  fetch('/api/control/elevenlabs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:elOn})});
}
function askGemini() {
  document.getElementById('aiText').textContent = 'Analyzing real-time vitals with Gemini...';
  fetch('/api/control/gemini_once', {method:'POST'});
}

// ─── CONFIDENCE BADGE ───
function confBadge(level) {
  if (!level || level === 'none') return '';
  const cls = level === 'high' ? 'conf-high' : level === 'medium' ? 'conf-medium' : 'conf-low';
  return '<span class="confidence ' + cls + '">' + level + '</span>';
}

// ─── SOCKET EVENTS ───
socket.on('connect', () => {
  const badge = document.getElementById('statusBadge');
  badge.classList.remove('disconnected');
  document.getElementById('statusDot').style.background = '';
  document.getElementById('statusText').textContent = 'Live';
});

socket.on('vitals', (d) => {
  // Heart rate
  if (d.heart_rate) {
    document.getElementById('hrVal').innerHTML = Math.round(d.heart_rate) + '<span class="unit">bpm</span>';
    document.getElementById('hrConf').outerHTML = confBadge(d.hr_confidence);
  }
  if (d.ground_truth_hr) document.getElementById('hrGT').textContent = d.ground_truth_hr;

  // Breathing
  if (d.breathing_rate) {
    document.getElementById('rrVal').innerHTML = Math.round(d.breathing_rate) + '<span class="unit">/min</span>';
    document.getElementById('rrConf').outerHTML = confBadge(d.breath_confidence);
  }

  // HRV
  if (d.hrv_rmssd) document.getElementById('hrvVal').innerHTML = d.hrv_rmssd.toFixed(1) + '<span class="unit">ms</span>';
  if (d.hrv_sdnn) document.getElementById('hrvSdnn').textContent = d.hrv_sdnn.toFixed(1);
  if (d.hrv_pnn50 !== undefined) document.getElementById('hrvPnn50').textContent = d.hrv_pnn50.toFixed(0);

  // Stress
  if (d.stress_index) {
    const el = document.getElementById('stressVal');
    el.textContent = d.stress_index;
    el.style.color = d.stress_index === 'high' ? 'var(--accent-red)' : d.stress_index === 'moderate' ? 'var(--accent-amber)' : 'var(--accent-green)';
  }

  // GSR
  if (d.gsr) document.getElementById('gsrVal').innerHTML = d.gsr.toFixed(1) + '<span class="unit">&micro;S</span>';

  // Activity
  if (d.activity) document.getElementById('activityVal').textContent = d.activity;
  if (d.motion_level !== undefined) document.getElementById('motionLevel').textContent = d.motion_level.toFixed(2);
  if (d.signal_quality) document.getElementById('sigQuality').textContent = d.signal_quality.toFixed(1);

  // Environment
  const env = d.environment || {};
  if (env.temperature_c) document.getElementById('envTemp').textContent = env.temperature_c.toFixed(1) + '°C';
  if (env.humidity_pct) document.getElementById('envHum').textContent = env.humidity_pct.toFixed(0) + '%';
  if (env.co2_ppm) document.getElementById('envCo2').textContent = env.co2_ppm + ' ppm';
  if (env.tvoc_ppb) document.getElementById('envTvoc').textContent = env.tvoc_ppb + ' ppb';
  if (d.occupancy !== undefined) document.getElementById('envOcc').textContent = d.occupancy;
  if (d.audio_events && d.audio_events.length > 0) {
    document.getElementById('audioVal').textContent = d.audio_events.map(e => e.type).join(', ');
  } else {
    document.getElementById('audioVal').textContent = 'quiet';
  }

  // Presence
  const ring = document.getElementById('presenceRing');
  const label = document.getElementById('presenceLabel');
  const score = document.getElementById('presenceScore');
  if (d.is_present) {
    ring.className = 'presence-indicator present';
    label.textContent = 'Person Detected';
    label.style.color = 'var(--accent-green)';
  } else {
    ring.className = 'presence-indicator absent';
    label.textContent = 'No presence';
    label.style.color = 'var(--text-muted)';
  }
  if (d.presence_score !== undefined) score.textContent = 'Score: ' + d.presence_score.toFixed(3);

  // Waveform
  if (d.waveform && d.waveform.length > 0) {
    const newData = d.waveform.slice(-MAX_WAVEFORM);
    waveformChart.data.datasets[0].data = newData;
    waveformChart.data.labels = Array(newData.length).fill('');
    waveformChart.update();
  }

  // Alert banners
  const hrCard = document.getElementById('hrCard');
  const rrCard = document.getElementById('rrCard');
  hrCard.classList.remove('warning', 'critical');
  rrCard.classList.remove('warning', 'critical');
  const banner = document.getElementById('alertBanner');
  if (d.alert_level === 'critical') {
    hrCard.classList.add('critical');
    banner.textContent = '⚠ CRITICAL — Vital signs outside safe range — immediate attention recommended';
    banner.classList.add('active');
  } else if (d.alert_level === 'warning') {
    hrCard.classList.add('warning');
    banner.textContent = '⚠ WARNING — Elevated vital signs detected';
    banner.classList.add('active');
  } else {
    banner.classList.remove('active');
  }

  // Node count
  if (d.data_sources) {
    document.getElementById('nodeCount').textContent = d.data_sources.length + ' active sources';
    // Update node dots
    d.data_sources.forEach(src => {
      if (src.startsWith('csi_')) activeNodes.add(src.replace('csi_', ''));
    });
    updateNodeDots();
  }
});

socket.on('ai_interpretation', (d) => {
  document.getElementById('aiText').textContent = d.text;
});

socket.on('disconnect', () => {
  const badge = document.getElementById('statusBadge');
  badge.classList.add('disconnected');
  document.getElementById('statusText').textContent = 'Disconnected';
});

function updateNodeDots() {
  const items = document.querySelectorAll('.node-item');
  items.forEach(item => {
    const name = item.querySelector('.node-name').textContent;
    const dot = item.querySelector('.node-dot');
    if (activeNodes.has(name)) {
      dot.classList.remove('offline');
      dot.classList.add('online');
    }
  });
}

// Update active nodes periodically (timeout after 30s)
setInterval(() => {
  // Reset and let next vitals message re-populate
  activeNodes.clear();
  document.querySelectorAll('.node-dot').forEach(d => { d.classList.remove('online'); d.classList.add('offline'); });
}, 30000);
</script>
</body>
</html>
"""


def create_app(buffers, mqtt_handler):
    """Create Flask + SocketIO dashboard application."""
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route('/api/vitals')
    def api_vitals():
        return jsonify({'status': 'ok', 'service': 'sensus', 'version': '1.0'})

    @app.route('/api/health')
    def api_health():
        return jsonify({'service': 'sensus', 'status': 'running',
                        'uptime': time.time()})

    @app.route('/api/control/gemini', methods=['POST'])
    def control_gemini():
        data = request.get_json()
        enabled = data.get('enabled', False)
        mqtt_handler.client.publish('sensus/control/gemini',
                                    'on' if enabled else 'off')
        return jsonify({'gemini': enabled})

    @app.route('/api/control/elevenlabs', methods=['POST'])
    def control_elevenlabs():
        data = request.get_json()
        enabled = data.get('enabled', False)
        mqtt_handler.client.publish('sensus/control/elevenlabs',
                                    'on' if enabled else 'off')
        return jsonify({'elevenlabs': enabled})

    @app.route('/api/control/gemini_once', methods=['POST'])
    def gemini_once():
        mqtt_handler.client.publish('sensus/control/gemini', 'on')
        def auto_off():
            time.sleep(15)
            mqtt_handler.client.publish('sensus/control/gemini', 'off')
        threading.Thread(target=auto_off, daemon=True).start()
        return jsonify({'gemini': 'one_shot'})

    @app.route('/api/calibrate', methods=['POST'])
    def calibrate():
        """Capture empty-room baseline for presence detection."""
        mqtt_handler.client.publish('sensus/control/calibrate', 'start')
        return jsonify({'calibration': 'started'})

    # ─── WebSocket Bridge: MQTT → SocketIO ───
    def mqtt_to_ws():
        client = mqtt.Client(client_id='sensus-ws-bridge')
        def on_message(c, u, msg):
            try:
                data = json.loads(msg.payload.decode())
                topic = msg.topic
                if 'vitals' in topic:
                    socketio.emit('vitals', data)
                elif 'interpretation' in topic:
                    socketio.emit('ai_interpretation', data)
            except Exception:
                pass
        client.on_message = on_message
        client.connect('localhost', 1883)
        client.subscribe('sensus/fused/#')
        client.subscribe('sensus/ai/#')
        client.loop_forever()

    ws_thread = threading.Thread(target=mqtt_to_ws, daemon=True)
    ws_thread.start()

    return app
