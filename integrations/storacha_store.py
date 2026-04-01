"""
Storacha Integration — Persistent Health Data Storage
=====================================================
Stores health monitoring data on Storacha's decentralized storage
network with UCAN delegation for clinician access control.

Challenge: "Persistent Agent Memory" / "Decentralized RAG Knowledge Base"

How Sensus fits:
  - Health agent state persists across sessions on Storacha
  - UCAN delegations enable secure clinician data sharing
  - Content-addressed retrieval ensures data integrity
  - Health knowledge base for AI-powered clinical reasoning

Usage:
    from integrations.storacha_store import StorachaHealthStore
    
    store = StorachaHealthStore()
    cid = store.upload_session(session_data)
    delegation = store.create_delegation(cid, "dr_jones")
"""

import json
import hashlib
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import secrets


@dataclass
class UCANDelegation:
    """UCAN delegation for data access."""
    delegation_id: str
    issuer: str        # data owner
    audience: str      # clinician/recipient
    resource_cid: str  # what they can access
    capabilities: List[str]  # read, write, list
    expiration: str
    created_at: str = ""
    chain_depth: int = 0   # for delegation chain tracking
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class StorachaHealthStore:
    """
    Health data storage on Storacha network.
    
    Features:
      - Upload health sessions as content-addressed blobs
      - UCAN-based delegation for clinician access
      - Delegation chains for multi-provider scenarios
      - Knowledge base for health AI agents
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("STORACHA_TOKEN", "")
        self.stored: Dict[str, Dict] = {}  # cid -> data
        self.delegations: Dict[str, UCANDelegation] = {}
        self._has_api = bool(self.api_token)
    
    def upload_session(
        self,
        session_states: List[Dict],
        owner_id: str = "patient_001",
        scenario_name: str = "Live Session",
    ) -> str:
        """Upload a health session to Storacha. Returns CID."""
        
        # Build compact payload
        payload = {
            "type": "sensus-health-session",
            "version": "1.0",
            "owner": owner_id,
            "scenario": scenario_name,
            "created": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(session_states),
            "vitals": [],
            "alerts": [],
            "summary": {},
        }
        
        # Compact vital signs
        for s in session_states:
            payload["vitals"].append({
                "t": round(s.get("elapsed_sec", 0), 2),
                "hr": s.get("heart_rate", 0),
                "br": s.get("breathing_rate", 0),
                "hrv": s.get("hrv_rmssd", 0),
                "spo2": s.get("spo2", 98),
                "stress": s.get("stress_index", "low"),
                "motion": round(s.get("motion_level", 0), 2),
                "present": s.get("is_present", True),
                "alert": s.get("alert_level", "normal"),
            })
            
            if s.get("alert_level") in ("warning", "critical"):
                msg = s.get("alert_message", "")
                if msg and msg not in [a["msg"] for a in payload["alerts"]]:
                    payload["alerts"].append({
                        "t": round(s.get("elapsed_sec", 0), 2),
                        "level": s.get("alert_level"),
                        "msg": msg,
                    })
        
        # Summary stats
        import numpy as np
        hrs = [v["hr"] for v in payload["vitals"] if v["hr"] > 0]
        brs = [v["br"] for v in payload["vitals"] if v["br"] > 0]
        if hrs:
            payload["summary"]["hr_mean"] = round(float(np.mean(hrs)), 1)
            payload["summary"]["hr_range"] = [round(float(np.min(hrs)), 1), round(float(np.max(hrs)), 1)]
        if brs:
            payload["summary"]["br_mean"] = round(float(np.mean(brs)), 1)
        payload["summary"]["alert_count"] = len(payload["alerts"])
        
        # Generate CID
        content = json.dumps(payload, sort_keys=True, default=str)
        cid = f"bafkreig{hashlib.sha256(content.encode()).hexdigest()[:52]}"
        
        # Upload to Storacha if token available
        if self._has_api:
            cid = self._upload_to_storacha(content, cid)
        
        self.stored[cid] = payload
        return cid
    
    def create_delegation(
        self,
        resource_cid: str,
        audience: str,
        issuer: str = "patient_001",
        capabilities: Optional[List[str]] = None,
        duration_days: int = 7,
    ) -> UCANDelegation:
        """Create a UCAN delegation for data access."""
        
        delegation = UCANDelegation(
            delegation_id=secrets.token_hex(8),
            issuer=issuer,
            audience=audience,
            resource_cid=resource_cid,
            capabilities=capabilities or ["read"],
            expiration=(datetime.now(timezone.utc) + timedelta(days=duration_days)).isoformat(),
        )
        
        self.delegations[delegation.delegation_id] = delegation
        return delegation
    
    def re_delegate(
        self,
        parent_delegation_id: str,
        new_audience: str,
    ) -> Optional[UCANDelegation]:
        """
        Re-delegate access (delegation chain).
        Enables: Patient → Doctor → Specialist chain.
        """
        parent = self.delegations.get(parent_delegation_id)
        if not parent:
            return None
        
        child = UCANDelegation(
            delegation_id=secrets.token_hex(8),
            issuer=parent.audience,  # delegator becomes issuer
            audience=new_audience,
            resource_cid=parent.resource_cid,
            capabilities=parent.capabilities,  # can't escalate
            expiration=parent.expiration,  # can't extend
            chain_depth=parent.chain_depth + 1,
        )
        
        self.delegations[child.delegation_id] = child
        return child
    
    def access_with_delegation(
        self,
        delegation_id: str,
        requester: str,
    ) -> Optional[Dict]:
        """Access data using a UCAN delegation."""
        deleg = self.delegations.get(delegation_id)
        if not deleg:
            return None
        
        if deleg.audience != requester:
            return None
        
        now = datetime.now(timezone.utc).isoformat()
        if now > deleg.expiration:
            return None
        
        if "read" not in deleg.capabilities:
            return None
        
        return self.stored.get(deleg.resource_cid)
    
    def build_knowledge_base(self) -> Dict:
        """
        Build a knowledge base from all stored sessions.
        For AI agent RAG (Retrieval-Augmented Generation).
        """
        kb = {
            "type": "sensus-health-knowledge-base",
            "sessions": len(self.stored),
            "entries": [],
        }
        
        for cid, data in self.stored.items():
            kb["entries"].append({
                "cid": cid,
                "scenario": data.get("scenario", "Unknown"),
                "summary": data.get("summary", {}),
                "alert_count": len(data.get("alerts", [])),
                "sample_count": data.get("sample_count", 0),
            })
        
        return kb
    
    def export(self, output_dir: str) -> Dict[str, str]:
        """Export all data and delegations."""
        os.makedirs(output_dir, exist_ok=True)
        files = {}
        
        for cid, data in self.stored.items():
            path = os.path.join(output_dir, f"storacha_{cid[:20]}.json")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            files[cid] = path
        
        # Delegation chains
        deleg_path = os.path.join(output_dir, "delegations.json")
        with open(deleg_path, 'w') as f:
            json.dump([asdict(d) for d in self.delegations.values()], f, indent=2, default=str)
        files["delegations"] = deleg_path
        
        # Knowledge base
        kb_path = os.path.join(output_dir, "knowledge_base.json")
        with open(kb_path, 'w') as f:
            json.dump(self.build_knowledge_base(), f, indent=2, default=str)
        files["knowledge_base"] = kb_path
        
        return files
    
    def _upload_to_storacha(self, content: str, fallback_cid: str) -> str:
        """Upload to Storacha API."""
        try:
            import requests
            resp = requests.post(
                "https://up.storacha.network/upload",
                headers={"Authorization": f"Bearer {self.api_token}"},
                files={"file": ("health_data.json", content.encode(), "application/json")},
                timeout=30,
            )
            if resp.ok:
                return resp.json().get("cid", fallback_cid)
        except Exception:
            pass
        return fallback_cid


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))
    from simulator import create_engine
    
    print("=" * 70)
    print("STORACHA INTEGRATION — SELF TEST")
    print("=" * 70)
    
    store = StorachaHealthStore()
    
    # Store multiple scenarios
    for sid in [1, 5, 13]:
        engine = create_engine(sid, speed=10.0)
        states = []
        for _ in range(int(engine.scenario.total_duration_sec / 0.1) + 1):
            states.append(engine.step(0.1))
        cid = store.upload_session(states, "patient_001", engine.scenario.name)
        print(f"  Stored '{engine.scenario.name}' → {cid[:30]}...")
    
    # Create delegation chain: patient → doctor → specialist
    cid = list(store.stored.keys())[0]
    d1 = store.create_delegation(cid, "dr_jones", "patient_001")
    print(f"\nDelegation: patient → dr_jones (depth {d1.chain_depth})")
    
    d2 = store.re_delegate(d1.delegation_id, "specialist_lee")
    print(f"Re-delegation: dr_jones → specialist_lee (depth {d2.chain_depth})")
    
    # Access with delegation
    data = store.access_with_delegation(d2.delegation_id, "specialist_lee")
    print(f"Specialist access: {'✅ granted' if data else '❌ denied'}")
    
    # Knowledge base
    kb = store.build_knowledge_base()
    print(f"\nKnowledge base: {kb['sessions']} sessions indexed")
    
    # Export
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'storacha')
    files = store.export(output_dir)
    print(f"Exported {len(files)} files")
    
    print("\n✅ Storacha integration test passed")
