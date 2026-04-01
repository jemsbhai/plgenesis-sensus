"""
Impulse AI Live Inference via /api/chat endpoint
==================================================
Sends vital signs to Impulse AI's chat-based ML gateway
for real-time health state classification.

The user uploads the CSV to Impulse AI console and trains a model.
This module sends inference requests through the /api/chat SSE endpoint.
"""

import os
import asyncio
import aiohttp
import json
import re
from typing import Dict, Optional


IMPULSE_API_URL = "https://api.impulselabs.ai/api/chat"


async def impulse_classify_async(vital_signs: Dict) -> Optional[Dict]:
    """
    Send vital signs to Impulse AI for classification via /api/chat.
    
    Args:
        vital_signs: Dict with keys like heart_rate, breathing_rate, etc.
    
    Returns:
        Dict with classification result, or None on failure.
    """
    api_key = os.environ.get("IMPULSE_API_KEY") or os.environ.get("IMPSDK_API_KEY")
    if not api_key:
        return None

    # Build a structured prompt for classification
    stress_val = {'low': 0, 'moderate': 1, 'high': 2}.get(vital_signs.get('stress_index', 'low'), 0)
    hr = vital_signs.get('heart_rate', 0)
    br = vital_signs.get('breathing_rate', 0)
    spo2 = vital_signs.get('spo2', 98)
    motion = vital_signs.get('motion_level', 0)
    irreg = vital_signs.get('irregular_rhythm', False)

    prompt = (
        f"You are a clinical health classifier. Given these vital signs from a contactless WiFi CSI sensor, "
        f"classify the health state into exactly ONE of these categories: "
        f"Normal, Sleep, Relaxation, Stress, Medication, Environmental, "
        f"Cardiac Alert, Respiratory Alert, Emergency, Cardiac Emergency.\n\n"
        f"Vital signs:\n"
        f"- Heart Rate: {hr} bpm\n"
        f"- Breathing Rate: {br} /min\n"
        f"- HRV RMSSD: {vital_signs.get('hrv_rmssd', 0)} ms\n"
        f"- HRV SDNN: {vital_signs.get('hrv_sdnn', 0)} ms\n"
        f"- SpO2: {spo2}%\n"
        f"- GSR: {vital_signs.get('gsr', 2.5)} uS\n"
        f"- Motion Level: {motion}\n"
        f"- Blood Pressure: {vital_signs.get('blood_pressure_sys', 120)}/{vital_signs.get('blood_pressure_dia', 80)} mmHg\n"
        f"- Skin Temp: {vital_signs.get('skin_temp', 36.5)} C\n"
        f"- Irregular Rhythm: {irreg}\n\n"
        f"Respond with ONLY a JSON object, nothing else: "
        '{"predicted_class": "...", "confidence": 0.XX, "reasoning": "brief explanation"}'
    )

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                IMPULSE_API_URL, 
                headers=headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None

                # Parse SSE stream — collect delta chunks
                full_content = ""
                raw_text = await resp.text()
                for line in raw_text.split('\n'):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        ctype = chunk.get("type", "")
                        if ctype == "delta":
                            content = chunk.get("data", {}).get("content", "")
                            if content:
                                full_content += content
                        elif ctype == "complete":
                            c = chunk.get("data", {}).get("content", "")
                            if c:
                                full_content = c
                            break
                        elif ctype == "done":
                            break
                    except json.JSONDecodeError:
                        pass

                if not full_content:
                    return None

                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    json_match = re.search(r'\{[^}]+\}', full_content)
                    if json_match:
                        result = json.loads(json_match.group())
                        return {
                            "source": "impulse_ai",
                            "predicted_class": result.get("predicted_class", "Unknown"),
                            "confidence": float(result.get("confidence", 0)),
                            "raw_response": full_content[:200],
                        }
                except (json.JSONDecodeError, ValueError):
                    pass

                # Fallback: return raw response
                return {
                    "source": "impulse_ai",
                    "predicted_class": "See response",
                    "confidence": 0,
                    "raw_response": full_content[:300],
                }

    except Exception as e:
        return {"source": "impulse_ai", "error": str(e)}


def impulse_classify_sync(vital_signs: Dict) -> Optional[Dict]:
    """Synchronous wrapper for Streamlit."""
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(impulse_classify_async(vital_signs))
        loop.close()
        return result
    except Exception as e:
        return {"source": "impulse_ai", "error": str(e)}


async def impulse_health_check() -> Dict:
    """Check if Impulse AI is reachable and authenticated."""
    api_key = os.environ.get("IMPULSE_API_KEY") or os.environ.get("IMPSDK_API_KEY")
    result = {
        "has_key": bool(api_key),
        "api_reachable": False,
        "gateway_version": None,
    }
    if not api_key:
        return result

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.impulselabs.ai/health",
                headers={"X-API-Key": api_key},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result["api_reachable"] = True
                    result["gateway_version"] = data.get("version", "unknown")
    except:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

    print("=" * 70)
    print("IMPULSE AI LIVE INFERENCE TEST")
    print("=" * 70)

    # Health check
    health = asyncio.run(impulse_health_check())
    print(f"\nHealth check:")
    print(f"  API key present: {health['has_key']}")
    print(f"  API reachable:   {health['api_reachable']}")
    print(f"  Gateway version: {health['gateway_version']}")

    if not health['has_key']:
        print("\n❌ Set IMPULSE_API_KEY environment variable")
        exit(1)

    # Quick raw test to see what the API returns
    print("\n--- Raw API response test ---")
    api_key = os.environ.get("IMPULSE_API_KEY") or os.environ.get("IMPSDK_API_KEY")
    
    async def raw_test():
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": 
                "I uploaded a CSV dataset called sensus-health-classification. "
                "Can you list my uploaded datasets?"
            }]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(IMPULSE_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                raw = await resp.text()
                print(f"  Status: {resp.status}")
                # Show each SSE line
                for line in raw.split('\n'):
                    line = line.strip()
                    if line:
                        print(f"  >> {line[:200]}")
    
    import aiohttp
    asyncio.run(raw_test())

    # Test with a cardiac arrest scenario
    print("\n--- Test 1: Cardiac Arrest vitals ---")
    cardiac_arrest = {
        "heart_rate": 0, "breathing_rate": 0,
        "hrv_rmssd": 0, "hrv_sdnn": 0,
        "spo2": 70, "gsr": 0.5,
        "motion_level": 0, "blood_pressure_sys": 0,
        "blood_pressure_dia": 0, "skin_temp": 35.0,
        "stress_index": "high", "is_motion": False,
        "is_present": True, "irregular_rhythm": False,
        "signal_quality": 0,
    }
    result = asyncio.run(impulse_classify_async(cardiac_arrest))
    print(f"  Result: {json.dumps(result, indent=2)}")

    # Test with normal resting
    print("\n--- Test 2: Normal resting vitals ---")
    normal = {
        "heart_rate": 72, "breathing_rate": 15,
        "hrv_rmssd": 40, "hrv_sdnn": 48,
        "spo2": 98, "gsr": 2.0,
        "motion_level": 0.1, "blood_pressure_sys": 118,
        "blood_pressure_dia": 76, "skin_temp": 36.5,
        "stress_index": "low", "is_motion": False,
        "is_present": True, "irregular_rhythm": False,
        "signal_quality": 15,
    }
    result = asyncio.run(impulse_classify_async(normal))
    print(f"  Result: {json.dumps(result, indent=2)}")

    print("\n✅ Impulse AI inference test complete")
