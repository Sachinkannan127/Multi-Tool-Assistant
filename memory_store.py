import json
import logging
import uuid
from pathlib import Path
from threading import Lock
from datetime import datetime

logger = logging.getLogger(__name__)
MEMORY_FILE = Path("memories.json")
file_lock = Lock()

def load_memories() -> list[dict]:
    """Load memories from file securely."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with file_lock:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("memories", [])
    except Exception as e:
        logger.error(f"Error loading memories from file: {e}")
        return []

def save_memories(memories: list[dict]):
    """Save memories to file securely using a file lock."""
    try:
        with file_lock:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"memories": memories}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving memories to file: {e}")

def add_memory(content: str) -> dict:
    """Add a new memory to the store."""
    memories = load_memories()
    new_memory = {
        "id": f"mem_{uuid.uuid4().hex[:8]}",
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    memories.append(new_memory)
    save_memories(memories)
    return new_memory

def delete_memory(memory_id: str) -> bool:
    """Delete a memory by its ID."""
    memories = load_memories()
    initial_len = len(memories)
    memories = [m for m in memories if m["id"] != memory_id]
    if len(memories) < initial_len:
        save_memories(memories)
        return True
    return False
