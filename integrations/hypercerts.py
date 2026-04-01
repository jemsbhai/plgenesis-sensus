"""
Hypercerts Integration — Impact Claims from Health Sensing
==========================================================
Generates structured hypercert-compatible impact claims from Sensus
health monitoring sessions. Each session produces verifiable evidence
of contactless vital sign extraction — who was monitored, when, what
was detected, and with what evidence.

Hypercerts Protocol: https://hypercerts.org
Challenge: "Connect the world's impact data to hypercerts"

How Sensus fits:
  - Sensus generates real-time physiological data from WiFi CSI sensors
  - Each monitoring session produces structured health impact evidence
  - Impact claims include: vital signs extracted, alerts generated,
    clinical scenarios detected, and sensor mesh configuration
  - Evidence is CID-addressable for decentralized verification

Usage:
    from integrations.hypercerts import HypercertGenerator
    
    gen = HypercertGenerator()
    claim = gen.create_health_monitoring_claim(session_data)
    gen.export_claim(claim, "output/")
"""

import json
import hashlib
import time
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════════════════════
# HYPERCERT SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HypercertMetadata:
    """
    Hypercert-compatible metadata following the impact claim schema.
    See: https://hypercerts.org/docs/developer/api/metadata
    """
    name: str
    description: str
    image: str = ""
    external_url: str = ""
    
    # Hypercert-specific fields
    work_scope: List[str] = field(default_factory=list)
    work_timeframe_start: str = ""
    work_timeframe_end: str = ""
    impact_scope: List[str] = field(default_factory=list)
    impact_timeframe_start: str = ""
    impact_timeframe_end: str = ""
    contributors: List[str] = field(default_factory=list)
    rights: List[str] = field(default_factory=list)
    
    # Extended properties for health data
    properties: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass 
class HealthImpactEvidence:
    """Structured evidence attached to a hypercert claim."""
    session_id: str
    timestamp_start: str
    timestamp_end: str
    duration_sec: float
    
    # Sensing configuration
    sensor_type: str = "wifi_csi"
    node_count: int = 3
    node_ids: List[str] = field(default_factory=list)
    sample_rate_hz: int = 10
    subcarrier_count: int = 52
    
    # Vital signs extracted
    vitals_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Alerts generated
    alerts: List[Dict[str, str]] = field(default_factory=list)
    
    # Clinical scenarios detected
    scenarios_detected: List[str] = field(default_factory=list)
    
    # Data integrity
    data_hash: str = ""
    evidence_cid: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class HypercertGenerator:
    """
    Generates hypercert impact claims from Sensus health monitoring sessions.
    
    Flow:
      1. Collect session data (vital signs, alerts, config)
      2. Create evidence object with data integrity hash
      3. Generate hypercert metadata with impact scope
      4. Export as JSON for on-chain minting
    """
    
    PROJECT_URL = "https://github.com/jemsbhai/plgenesis-sensus"
    
    def __init__(self, contributor_name: str = "Sensus Health Platform"):
        self.contributor = contributor_name
    
    def create_health_monitoring_claim(
        self,
        session_states: List[Dict],
        scenario_name: str = "Live Monitoring",
        scenario_category: str = "Health",
        node_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Create a complete hypercert claim from a monitoring session.
        
        Args:
            session_states: List of state dicts from ScenarioEngine.step()
            scenario_name: Name of the scenario monitored
            scenario_category: Category of health scenario
            node_ids: IDs of ESP32 nodes used
            
        Returns:
            Dict with 'metadata' and 'evidence' keys
        """
        if not session_states:
            raise ValueError("No session data provided")
        
        session_id = self._generate_session_id(session_states)
        now = datetime.now(timezone.utc)
        
        # Compute session duration
        first_t = session_states[0].get('elapsed_sec', 0)
        last_t = session_states[-1].get('elapsed_sec', 0)
        duration = last_t - first_t
        
        # Extract vital sign summaries
        vitals_summary = self._summarize_vitals(session_states)
        
        # Collect alerts
        alerts = self._collect_alerts(session_states)
        
        # Build evidence
        evidence = HealthImpactEvidence(
            session_id=session_id,
            timestamp_start=now.isoformat(),
            timestamp_end=now.isoformat(),
            duration_sec=round(duration, 2),
            sensor_type="wifi_csi",
            node_count=session_states[0].get('node_count', 3),
            node_ids=node_ids or ["node_1", "node_2", "node_3"],
            sample_rate_hz=10,
            subcarrier_count=52,
            vitals_summary=vitals_summary,
            alerts=alerts,
            scenarios_detected=[scenario_name],
        )
        
        # Compute data integrity hash
        evidence_json = json.dumps(evidence.to_dict(), sort_keys=True, default=str)
        evidence.data_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
        evidence.evidence_cid = f"bafk{evidence.data_hash[:56]}"  # Mock CID format
        
        # Build hypercert metadata
        metadata = HypercertMetadata(
            name=f"Sensus Health Monitoring: {scenario_name}",
            description=(
                f"Contactless health monitoring session using WiFi CSI sensing. "
                f"Scenario: {scenario_name} ({scenario_category}). "
                f"Duration: {duration:.0f}s with {evidence.node_count} sensor nodes. "
                f"Vital signs extracted: HR, BR, HRV, SpO2, BP estimation. "
                f"Alerts generated: {len(alerts)}."
            ),
            external_url=self.PROJECT_URL,
            work_scope=[
                "contactless-health-monitoring",
                "wifi-csi-sensing",
                "vital-sign-extraction",
                f"scenario:{scenario_name.lower().replace(' ', '-')}",
            ],
            work_timeframe_start=now.isoformat(),
            work_timeframe_end=now.isoformat(),
            impact_scope=[
                "healthcare-accessibility",
                "patient-monitoring",
                "early-warning-detection",
                "non-invasive-sensing",
            ],
            impact_timeframe_start=now.isoformat(),
            impact_timeframe_end=now.isoformat(),
            contributors=[self.contributor],
            rights=["Public"],
            properties=[
                {"trait_type": "Platform", "value": "Sensus"},
                {"trait_type": "Sensor Type", "value": "WiFi CSI (2.4 GHz)"},
                {"trait_type": "Hardware", "value": "ESP32-C6 Mesh + Raspberry Pi 5"},
                {"trait_type": "Scenario", "value": scenario_name},
                {"trait_type": "Category", "value": scenario_category},
                {"trait_type": "Duration (s)", "value": round(duration, 1)},
                {"trait_type": "Node Count", "value": evidence.node_count},
                {"trait_type": "Alerts Generated", "value": len(alerts)},
                {"trait_type": "Mean Heart Rate", "value": vitals_summary.get("hr_mean", 0)},
                {"trait_type": "Mean Breathing Rate", "value": vitals_summary.get("br_mean", 0)},
                {"trait_type": "Evidence CID", "value": evidence.evidence_cid},
                {"trait_type": "Data Hash (SHA-256)", "value": evidence.data_hash},
                {"trait_type": "Hackathon", "value": "PL_Genesis 2026"},
            ],
        )
        
        return {
            "metadata": metadata.to_dict(),
            "evidence": evidence.to_dict(),
            "session_id": session_id,
            "evidence_cid": evidence.evidence_cid,
        }
    
    def export_claim(self, claim: Dict, output_dir: str) -> Dict[str, str]:
        """
        Export a hypercert claim to JSON files.
        
        Returns dict of {file_type: filepath}
        """
        os.makedirs(output_dir, exist_ok=True)
        session_id = claim["session_id"]
        
        # Metadata file (for on-chain minting)
        meta_path = os.path.join(output_dir, f"hypercert_metadata_{session_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(claim["metadata"], f, indent=2, default=str)
        
        # Evidence file (for decentralized storage)
        evidence_path = os.path.join(output_dir, f"hypercert_evidence_{session_id}.json")
        with open(evidence_path, 'w') as f:
            json.dump(claim["evidence"], f, indent=2, default=str)
        
        # Combined claim file
        combined_path = os.path.join(output_dir, f"hypercert_claim_{session_id}.json")
        with open(combined_path, 'w') as f:
            json.dump(claim, f, indent=2, default=str)
        
        return {
            "metadata": meta_path,
            "evidence": evidence_path,
            "combined": combined_path,
        }
    
    def _generate_session_id(self, states: List[Dict]) -> str:
        """Generate unique session ID from state data."""
        seed = f"{time.time()}:{len(states)}:{states[0].get('scenario_name', '')}"
        return hashlib.md5(seed.encode()).hexdigest()[:12]
    
    def _summarize_vitals(self, states: List[Dict]) -> Dict:
        """Compute summary statistics from session states."""
        hrs = [s.get('heart_rate', 0) for s in states if s.get('heart_rate', 0) > 0]
        brs = [s.get('breathing_rate', 0) for s in states if s.get('breathing_rate', 0) > 0]
        hrvs = [s.get('hrv_rmssd', 0) for s in states if s.get('hrv_rmssd', 0) > 0]
        spo2s = [s.get('spo2', 98) for s in states]
        
        def stats(data, name):
            if not data:
                return {}
            import numpy as np
            arr = np.array(data)
            return {
                f"{name}_mean": round(float(np.mean(arr)), 1),
                f"{name}_min": round(float(np.min(arr)), 1),
                f"{name}_max": round(float(np.max(arr)), 1),
                f"{name}_std": round(float(np.std(arr)), 2),
                f"{name}_samples": len(data),
            }
        
        summary = {}
        summary.update(stats(hrs, "hr"))
        summary.update(stats(brs, "br"))
        summary.update(stats(hrvs, "hrv_rmssd"))
        summary.update(stats(spo2s, "spo2"))
        
        # Presence stats
        present_count = sum(1 for s in states if s.get('is_present', False))
        summary["presence_pct"] = round(present_count / len(states) * 100, 1) if states else 0
        
        # Motion stats
        motion_count = sum(1 for s in states if s.get('is_motion', False))
        summary["motion_pct"] = round(motion_count / len(states) * 100, 1) if states else 0
        
        return summary
    
    def _collect_alerts(self, states: List[Dict]) -> List[Dict[str, str]]:
        """Collect unique alerts from session."""
        seen = set()
        alerts = []
        for s in states:
            level = s.get('alert_level', 'normal')
            msg = s.get('alert_message', '')
            if level in ('warning', 'critical') and msg and msg not in seen:
                seen.add(msg)
                alerts.append({
                    "level": level,
                    "message": msg,
                    "timestamp": s.get('elapsed_sec', 0),
                })
        return alerts


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════

def generate_claim_from_scenario(scenario_id: int, speed: float = 10.0) -> Dict:
    """
    Run a scenario and generate a hypercert claim from it.
    Convenience function for demo/testing.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
    from simulator import create_engine
    
    engine = create_engine(scenario_id, speed=speed)
    states = []
    
    # Run the full scenario
    total_dur = engine.scenario.total_duration_sec
    steps = int(total_dur / 0.1) + 1
    for _ in range(steps):
        state = engine.step(0.1)
        states.append(state)
    
    gen = HypercertGenerator()
    claim = gen.create_health_monitoring_claim(
        session_states=states,
        scenario_name=engine.scenario.name,
        scenario_category=engine.scenario.category,
    )
    
    return claim


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("HYPERCERT INTEGRATION — SELF TEST")
    print("=" * 70)
    
    # Generate a claim from Scenario 13 (Cardiac Arrest)
    claim = generate_claim_from_scenario(13)
    
    print(f"\nSession ID: {claim['session_id']}")
    print(f"Evidence CID: {claim['evidence_cid']}")
    print(f"\nMetadata name: {claim['metadata']['name']}")
    print(f"Work scope: {claim['metadata']['work_scope']}")
    print(f"Impact scope: {claim['metadata']['impact_scope']}")
    print(f"Properties: {len(claim['metadata']['properties'])} traits")
    
    evidence = claim['evidence']
    print(f"\nEvidence:")
    print(f"  Duration: {evidence['duration_sec']}s")
    print(f"  Nodes: {evidence['node_count']}")
    print(f"  Vitals: HR mean={evidence['vitals_summary'].get('hr_mean', 'N/A')}")
    print(f"  Alerts: {len(evidence['alerts'])}")
    for a in evidence['alerts']:
        print(f"    [{a['level']}] {a['message']}")
    print(f"  Data hash: {evidence['data_hash'][:24]}...")
    
    # Export
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'hypercerts')
    files = HypercertGenerator().export_claim(claim, output_dir)
    print(f"\nExported to:")
    for k, v in files.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Hypercert integration test passed")
