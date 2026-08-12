import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
SETTINGS_FILE = Path("settings.json")

def load_settings() -> dict:
    """Load settings from settings.json."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return {}

def save_settings(settings_dict: dict):
    """Save settings to settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
