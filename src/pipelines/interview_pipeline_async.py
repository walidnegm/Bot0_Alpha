"""
Pipeline to manage a conversation about a single topic (multiple sub-topics)
"""

from pathlib import Path
import json
from typing import Optional, Union, List
import aiofiles
import logging
import logging_config

from thought_generation.thought_reader import IndexedThoughtReader
from agents.facilitator_agent_async import FacilitatorAgentAsync
from agents.state_management import StateManager
from utils.generic_utils import pretty_print_json

from project_config import INTERVIEW_STATES_FILE

# Setup logger
logger = logging.getLogger(__name__)


async def interview_pipeline_async(
    thought_data_file: Union[Path, str],
    user_id: str,
    memory_file: Union[Path, str],
    interview_state_file: Union[Path, str],
    target_thought_indexes: Optional[List[int]] = None,
):
    """
    Pipeline function to initialize and run the facilitator agent with data from a JSON file.

    Args:
        thought_data_file (Union[Path, str]): Path to the JSON file containing indexed thoughts data.
        user_id (str): Identifier for the user.
        memory_file (Union[Path, str]): Path to the file for saving conversation memory.
        target_thought_index (Optional[int]): Specific thought index to focus on, if any.

    Returns:
        None
    """
    logger.info(f"Start interview_pipeline_async.")

    # Step 1. Read a main thought and its sub-thoughts from JSON
    try:
        # Step 1: Initialize IndexedThoughtReader and validate the JSON file
        thought_reader = IndexedThoughtReader(thought_data_file)
        indexed_idea_model = thought_reader.idea_instance  # Extract the validated model

        # Step 2: Initialize StateManager
        state_manager = StateManager(
            storage_path=interview_state_file
        )  # storage is where to save the states data file

        # Step 3: Instantiate FacilitatorAgentAsync with the validated model
        facilitator_agent = FacilitatorAgentAsync(
            user_id=user_id,
            idea_data=indexed_idea_model,
            state_manager=state_manager,
            llm_provider="openai",
            model_id="gpt-4-turbo",
            temperature=0.3,
            max_tokens=1056,
            memory_file=memory_file,
        )

        # Step 4: Process conversation
        await facilitator_agent.coordinate_conversation(
            thought_indexes=target_thought_indexes
        )

    except Exception as e:
        logger.error(f"Pipeline encountered an error: {e}")
