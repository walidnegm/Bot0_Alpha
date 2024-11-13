"""
Pipeline to manage a conversation about a single topic (multiple sub-topics)
"""

import json
import aiofiles
import logging
import logging_config

from agents.facilitator_agent_async import FacilitatorAgentAsync
from utils.generic_utils import pretty_print_json

logger = logging.getLogger(__name__)


async def run_topic_conversation_pipeline_async(sub_thought_file, memory_file):

    # Step 1. Read a main thought and its sub-thoughts from JSON
    try:
        async with aiofiles.open(sub_thought_file, "r") as f:
            data = await f.read()  # need to use async version for all the steps
        data = json.loads(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading concepts file: {e}")
        return

    # Step 2. Instantiate Facilitator Agent Async clss
    agent = FacilitatorAgentAsync(data)

    for sub_concept in agent.sub_concepts:
        sub_concept_name = sub_concept["name"]
        sub_concept_description = sub_concept.get("description", "")

        question = await agent.generate_question(
            sub_concept_name, sub_concept_description
        )
        logger.info(f"Question for '{sub_concept_name}': {question}")

    # await agent.start_conversation()
    # await agent.save_conversation_memory()
