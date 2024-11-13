"""
Pipeline to manage a conversation about a single topic (multiple sub-topics)
"""

import json
import logging
import logging_config

from agents.facilitator_agent_async import FacilitatorAgentAsync
from thought_generation.thought_reader import get_thoughts, get_sub_thoughts
from utils.generic_utils import pretty_print_json, read_from_json_file

logger = logging.getLogger(__name__)


def run_topic_conversation_pipeline(thought_file, memory_file):
    """TBA"""

    # Step 1. Read a main thought and its sub-thoughts from JSON
    data = read_from_json_file(json_file=thought_file)
    main_thoughts = get_thoughts(data)
    print(type(main_thoughts))
    print(main_thoughts)

    # # Step 2. Instantiate Facilitator Agent Async clss
    # agent = FacilitatorAgentAsync(data)

    # for sub_concept in agent.sub_concepts:
    #     sub_concept_name = sub_concept["name"]
    #     sub_concept_description = sub_concept.get("description", "")

    #     question = await agent.generate_question(
    #         sub_concept_name, sub_concept_description
    #     )
    #     logger.info(f"Question for '{sub_concept_name}': {question}")

    # await agent.start_conversation()
    # await agent.save_conversation_memory()
