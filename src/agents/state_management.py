from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from pydantic import ValidationError
import json
import logging
import logging_config

# from internal modules
from utils.generic_utils import read_from_json_file, save_to_json_file
from models.user_state_models import UserState
from models.evaluation_models import EvaluationCriteria

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages user states for the interview bot (managing high level conversation flow)
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
            "current_question": "What strategies have you implemented to improve team
            communication?",
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
        logger.debug(
            f"Initializing StateManager with storage_path: {storage_path}"
        )  # TODO: debugging; remove later

        self.storage_path: Path = storage_path if storage_path else None
        self.states: Dict[str, UserState] = self._load_states()

        logger.debug(
            f"StateManager initialized. Loaded states: {self.states}"
        )  # TODO: debugging; remove later

    def _load_states(self) -> Dict[str, UserState]:
        """
        Load states from file if storage path is provided w/t read_from_json_file,
        which has internal validation already.
        """
        if self.storage_path:
            try:
                raw_states = read_from_json_file(self.storage_path)
                # Validate and convert raw states to UserState instances
                return {
                    user_id: UserState(**state) for user_id, state in raw_states.items()
                }
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        return {}

    def get_state(self, user_id: str) -> UserState:
        logger.debug(f"Retrieving state for user {user_id}")

        # Check if the user's state exists
        if user_id not in self.states:
            logger.debug(f"User {user_id} not found, initializing default state.")
            self.states[user_id] = UserState(
                thought_index=0,
                sub_thought_index=0,
                current_question=None,
                last_updated=datetime.now().isoformat(),
            )
            self.persist_states()  # Persist the newly created state
            logger.debug(f"Initialized state for user {user_id}")

        # Return the user's state
        return self.states[user_id]

    def update_state(self, user_id: str, **updates):
        logger.debug(f"Updating state for user {user_id} with updates: {updates}")

        state = self.get_state(user_id)

        logger.debug(f"Current state before update: {state.model_dump()}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        # Automatically append evaluations if provided
        # *keep it flexible for now w/t the extra if "current_evaluation"
        # (current_evaluation may be be fixed as the ultimate key)
        if "current_evaluation" in updates:
            if isinstance(updates["current_evaluation"], EvaluationCriteria):
                state.evaluations.append(updates["current_evaluation"])
                logger.debug(
                    f"Appended evaluation for user {user_id}: {updates['current_evaluation'].model_dump()}"
                )
            else:
                logger.warning(
                    f"Invalid evaluation format for user {user_id}. Expected 'EvaluationCriteria', "
                    f"got {type(updates['current_evaluation'])}."
                )

        state.last_updated = datetime.now().isoformat()
        self.persist_states()
        logger.debug(f"State successfully updated for user {user_id}.")

    def persist_states(self):
        """Save states to file if storage path is provided."""
        if self.storage_path:
            try:
                raw_states = {
                    user_id: state.model_dump(
                        mode="json"
                    )  # Use `model_dump` with `json` mode
                    for user_id, state in self.states.items()
                }
                save_to_json_file(raw_states, self.storage_path)
                logger.info(f"States persisted to {self.storage_path}.")
            except Exception as e:
                logger.error(f"Error while persisting states: {e}")
                raise

    def reset_state(self, user_id: str):
        """Reset the state for a user."""
        if user_id in self.states:
            del self.states[user_id]
        self.persist_states()

    # TODO: fix later and update
    # def update_state_with_eval(self, user_id: str, **updates):
    #     """Update the state for a user."""
    #     logger.debug(f"Updating state for user {user_id} with updates: {updates}")

    #     # Retrieve the existing state
    #     state = self.get_state(user_id)

    #     logger.debug(
    #         f"Current state before update for user {user_id}: {state.model_dump()}"
    #     )  # TODO: debugging; remove later

    #     # Validate nested fields like `current_evaluation` or `evaluation` if provided
    #     if "current_evaluation" in updates:

    #         logger.debug(
    #             f"Validating current evaluation for user {user_id}: {updates['current_evaluation']}"
    #         )  # Debugging
    #         updates["current_evaluation"] = EvaluationCriteria(
    #             **updates["current_evaluation"]
    #         )

    #     if "evaluation" in updates:

    #         logger.debug(
    #             f"Appending new evaluation for user {user_id}: {updates['evaluation']}"
    #         )  # Debugging

    #         evaluation = EvaluationCriteria(**updates["evaluation"])
    #         state.evaluations.append(evaluation)
    #         logger.debug(
    #             f"Updated evaluations list for user {user_id}: {[eval.model_dump() for eval in state.evaluations]}"
    #         )

    #         del updates["evaluation"]

    #     # Apply other updates using `model_copy`
    #     try:
    #         updated_state = state.model_copy(update=updates)
    #         logger.debug(f"Updated state: {updated_state}")
    #         self.states[user_id] = updated_state
    #         self.persist_states()

    #         logger.debug(
    #             f"Updating state for user {user_id} with updates: {updates}"
    #         )  # Debugging

    #     except ValidationError as e:
    #         logger.error(f"Validation error while updating state for {user_id}: {e}")
    #         raise
    #     except Exception as e:
    #         logger.error(f"Unexpected error: {e}")
    #         raise
