import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)
HISTORY_FILE = Path("chat_sessions.json")
file_lock = Lock()

def load_sessions() -> dict:
    """Load sessions from file securely."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        with file_lock:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat sessions from file: {e}")
        return {}

def save_sessions(sessions: dict):
    """Save sessions to file securely using a file lock."""
    try:
        with file_lock:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chat sessions to file: {e}")
