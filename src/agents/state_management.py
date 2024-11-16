from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

# from internal modules
from utils.generic_utils import read_from_json_file, save_to_json_file


class StateManager:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path: Path = storage_path
        self.states: Dict[str, Dict] = self._load_states()

    def _load_states(self) -> Dict[str, Dict]:
        """
        Load states from file if storage path is provided w/t read_from_json_file,
        which has internal validation already.
        """
        if self.storage_path:
            try:
                read_from_json_file(self.storage_path)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        return {}

    def _persist_states(self):
        """Save states to file if storage path is provided."""
        if self.storage_path:
            save_to_json_file(self.storage_path)

    def get_state(self, user_id: str) -> Dict:
        """Retrieve the state for a user. Initialize if not found."""
        if user_id not in self.states:
            self.states[user_id] = {
                "thought_index": 0,
                "sub_thought_index": 0,
                "current_question": None,
                "last_updated": datetime.now().isoformat(),
                "other_metadata": {},
            }
        return self.states[user_id]

    def update_state(self, user_id: str, **updates):
        """Update the state for a user."""
        state = self.get_state(user_id)
        state.update(updates)
        state["last_updated"] = datetime.utcnow().isoformat()
        self._persist_states()

    def reset_state(self, user_id: str):
        """Reset the state for a user."""
        if user_id in self.states:
            del self.states[user_id]
        self._persist_states()
