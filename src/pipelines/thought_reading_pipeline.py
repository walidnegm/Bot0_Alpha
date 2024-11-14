"""
TBA
"""

from pathlib import Path
from typing import Optional, Union
import logging
import logging_config

from models.thought_models import IdeaJSONModel
from models.indexed_thought_models import IndexedIdeaJSONModel
from thought_generation.thought_reader import ThoughtReader
from agents.agent_utils import transform_to_indexed_idea_model

from utils.generic_utils import (
    read_from_json_file,
    pretty_print_json,
    save_to_json_file,
)

logger = logging


def thought_reading_pipeline(json_file: Union[Path, str]):
    """TBA"""
    # "Embedded Software Development" idea file location
    json_file = r"C:\github\Bot0_Release1\backend\input_output\thought_generation\openai_output\array_of_thoughts_output_openai_embedded_software_development.json"

    # Initialize ThoughtReader with your JSON file
    thought_reader = ThoughtReader(json_file)

    # Fetch the main idea or theme of the thoughts data
    main_idea = thought_reader.get_idea()
    print("\nMain Idea:", main_idea)
    # Example Output: Main Idea: embedded software development

    # Get a list of thought names only
    thought_names = thought_reader.get_thoughts()
    print("\nThought Names:")
    for thought in thought_names:
        print(thought)

    # Retrieve thoughts along with their descriptions
    thoughts_with_descriptions = thought_reader.get_thoughts_and_descriptions()
    print("\nThoughts and Descriptions:")
    for thought in thoughts_with_descriptions:
        print(f"{thought['thought']}: {thought['description']}")
    # Example Output:
    # Definition and Scope: This cluster focuses on basic concepts and definitions...
    # Hardware-Software Integration: This cluster focuses on...

    # Get sub-thoughts for a specific thought
    thought_name = "Definition and Scope"
    sub_thoughts = thought_reader.get_sub_thoughts_for_thought(thought_name)

    # LLM friendly version
    print("\nsub_thoughts - list of dics", sub_thoughts)

    # Human readable version
    print(f"\nSub-Thoughts for '{thought_name}':")
    for sub in sub_thoughts:
        print(f"{sub['name']}: {sub['description']} (Importance: {sub['importance']})")


def thought_reading_with_index_pipeline(json_file: Union[Path, str] = None):
    """TBA"""
    # "Embedded Software Development" idea file location
    json_file = r"C:\github\Bot0_Release1\backend\input_output\thought_generation\openai_output\array_of_thoughts_output_openai_embedded_software_development.json"

    data = read_from_json_file(json_file)
    # with open(self.json_file, "r") as f:
    #     data = json.load(f)

    # TODO: debugging; delete later
    # Debugging step: Print or log the structure of the loaded data
    logger.debug("Loaded data: %s", data)

    # Check if data is None or not a dictionary with expected keys
    if not isinstance(data, dict):
        logger.error("Data is not a dictionary in file: %s", json_file)
        raise ValueError(
            "Expected data to be a dictionary with 'idea' and 'thoughts' keys"
        )

    # Ensure `data` has the 'idea' and 'thoughts' keys
    if "idea" not in data or "thoughts" not in data:
        logger.error(
            "Data structure mismatch: expected a dictionary with 'idea' and 'thoughts'"
        )
        raise ValueError(
            "Expected data to be a dictionary with 'idea' and 'thoughts' keys"
        )

    # Directly instantiate IdeaJSONModel with the data
    original_idea = IdeaJSONModel(**data)

    # Transform to IndexedIdeaModel
    indexed_idea_model = transform_to_indexed_idea_model(original_idea)

    # Now `indexed_idea` contains indexed thoughts and sub-thoughts without modifying the original model
    indexed_idea_data = indexed_idea_model.model_dump()

    save_to_json_file(data=indexed_idea_data, file_path="indexed_idea.json")
