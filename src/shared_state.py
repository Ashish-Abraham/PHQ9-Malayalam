import json
import os
import time
import fcntl
from collections import Counter

STATE_FILE = "data/dashboard_state.json"

def init_shared_state():
    """Initialize the shared state file if it doesn't exist."""
    current_time = time.time()
    if not os.path.exists(STATE_FILE):
        state = {
            "patient": None, # Will store {name, age, gender, id}
            "symptoms": {},  # Will store {Question: Score}
            "external_factors": {}, # {Factor: Level}
            "top_emotions": {},
            "suicide_risk": {
                "label": "Supportive",
                "score": 0,
                "alerts": []
            },
            "last_updated": current_time,
            "message_count": 0
        }
        _write_state(state)

def clear_state():
    """Clear the shared state file."""
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass

def update_patient_data(patient_info):
    """Update patient demographics."""
    state = _read_state()
    if not state:
        init_shared_state()
        state = _read_state()
        
    state["patient"] = patient_info
    state["last_updated"] = time.time()
    _write_state(state)

def update_symptoms(symptoms):
    """Update symptom scores."""
    state = _read_state()
    if not state:
        init_shared_state()
        state = _read_state()
    
    state["symptoms"] = symptoms
    state["last_updated"] = time.time()
    print(f"[DEBUG] Updated symptoms with: {symptoms}")
    _write_state(state)

def update_external_factors(factors):
    """Update external factors."""
    state = _read_state()
    if not state:
        init_shared_state()
        state = _read_state()
        
    # Merge with existing
    current = state.get("external_factors", {})
    current.update(factors)
    state["external_factors"] = current
    state["last_updated"] = time.time()
    _write_state(state)

def _write_state(state):
    """Write state to file with lock."""
    with open(STATE_FILE, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(state, f)
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            from src.debug_utils import log_debug
            log_debug(f"Error writing state: {e}")

def _read_state():
    """Read state from file with lock."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            state = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return state
    except Exception as e:
        from src.debug_utils import log_debug
        log_debug(f"Error reading state: {e}")
        return None

def update_emotion(emotion):
    """Update emotion stats."""
    state = _read_state()
    if not state:
        init_shared_state()
        state = _read_state()
    
    # Update counts
    emotions = Counter(state.get("top_emotions", {}))
    emotions[emotion] += 1
    state["top_emotions"] = dict(emotions)
    state["last_updated"] = time.time()
    state["message_count"] += 1
    
    _write_state(state)

def update_suicide_risk(alert_data):
    """Update suicide risk alerts."""
    state = _read_state()
    if not state: 
        init_shared_state()
        state = _read_state()
        
    current_alerts = state.get("suicide_risk", {}).get("alerts", [])
    if alert_data.get("alert"):
        # Add new alert
        current_alerts.append({
            "message": "Suicidal language detected",
            "timestamp": time.time(),
            "details": alert_data
        })
        
    state["suicide_risk"]["alerts"] = current_alerts
    state["last_updated"] = time.time()
    _write_state(state)

def get_dashboard_state():
    """Get current state for dashboard."""
    return _read_state()
