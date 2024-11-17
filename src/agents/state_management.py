from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

# from internal modules
from utils.generic_utils import read_from_json_file, save_to_json_file


class StateManager:
    """
    Manages user states for the interview bot.
    Automatically persists states to a file after every modification.

    Attributes:
        - storage_path (str): The file path where states are persisted as JSON.
        - states (dict): A dictionary mapping user IDs to their respective UserState models.

    *Key methods:
        - `get_state`: Retrieves or initializes a user's state.
        - `update_state`: Updates user states with new data and timestamps.
        - `reset_state`: Deletes a user's state from the file.

    !Sample Interview State File Output:
    {
        "user_123": {
            "thought_index": 1,
            "sub_thought_index": 2,
            "current_question": "What strategies have you implemented to improve team communication?",
            "last_updated": "2024-11-17T12:34:56.789Z"
        },
        "user_456": {
            "thought_index": 0,
            "sub_thought_index": 0,
            "current_question": null,
            "last_updated": "2024-11-16T10:20:30.456Z"
        }
    }

    """

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

    def persist_states(self):
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
        state["last_updated"] = datetime.now().isoformat()
        self.persist_states()

    def reset_state(self, user_id: str):
        """Reset the state for a user."""
        if user_id in self.states:
            del self.states[user_id]
        self.persist_states()
