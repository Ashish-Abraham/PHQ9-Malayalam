import json
import os
import time
import fcntl
from collections import Counter

STATE_FILE = "/tmp/dashboard_state.json"

def init_shared_state():
    """Initialize the shared state file if it doesn't exist."""
    if not os.path.exists(STATE_FILE):
        state = {
            "top_emotions": {},
            "suicide_risk": {
                "label": "Supportive",
                "score": 0,
                "alerts": []
            },
            "last_updated": time.time(),
            "message_count": 0
        }
        _write_state(state)

def _write_state(state):
    """Write state to file with lock."""
    with open(STATE_FILE, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(state, f)
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

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
    except Exception:
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
