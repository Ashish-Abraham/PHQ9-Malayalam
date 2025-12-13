import datetime

LOG_FILE = "data/debug.log"

def log_debug(message):
    try:
        timestamp = datetime.datetime.now().isoformat()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass
