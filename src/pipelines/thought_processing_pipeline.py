"""
TBA
"""

from pathlib import Path
from typing import Callable, Union, List, Dict
import logging
import logging_config

from models.thought_models import IdeaJSONModel
from models.indexed_thought_models import IndexedIdeaJSONModel
from thought_generation.thought_reader import ThoughtReader, IndexedThoughtReader
from thought_generation.thought_utils import generate_indexed_model_file

from utils.generic_utils import (
    read_from_json_file,
    pretty_print_json,
    save_to_json_file,
)

logger = logging.getLogger(__name__)


def unindexed_thought_reading_pipeline(json_file: Union[Path, str]):
    """"""

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


def indexed_thought_reading_pipeline(
    unindexed_model_file: Union[Path, str],
    indexed_model_file: Union[Path, str],
    thought_index: int = 0,
    make_indexed_file: Callable[
        [Union[Path, str], Union[Path, str]], None
    ] = generate_indexed_model_file,
    reader: Callable[[Union[Path, str]], "IndexedThoughtReader"] = IndexedThoughtReader,
) -> List[Dict[str, Union[str, int, None]]]:
    """
    A pipeline function that reads an unindexed model file, creates an indexed model file,
    and retrieves thought and sub-thought details for a specified thought index.

    Args:
        unindexed_model_file (Union[Path, str]): Path to the JSON file containing the unindexed model.
        indexed_model_file (Union[Path, str]): Path where the indexed model JSON file will be saved.
        thought_index (int, optional): The index of the thought to retrieve sub-thoughts for. Defaults to 0.
        make_indexed_file (Callable[[Union[Path, str], Union[Path, str]], None], optional):
            A callable that generates the indexed model file from the unindexed file.
            Defaults to `generate_indexed_model_file`.
        reader (Callable[[Union[Path, str]], IndexedThoughtReader], optional):
            A callable to instantiate the reader that can access the indexed model.
            Defaults to `IndexedThoughtReader`.

    Returns:
        List[Dict[str, Union[str, int, None]]]: A list of dictionaries containing details for each
        sub-thought in the specified thought, with fields for `sub_thought_index`, `name`,
        `description`, `importance`, and `connection_to_next`.

    Workflow:
        - Converts paths to Path objects for consistency.
        - Uses `make_indexed_file` to generate the indexed model file from the unindexed model.
        - Instantiates the `reader` to access the indexed model file.
        - Retrieves the main idea, list of thoughts, descriptions, and sub-thoughts for the specified index.
    """
    unindexed_model_file, indexed_model_file = Path(unindexed_model_file), Path(
        indexed_model_file
    )

    # Create the indexed idea model file
    make_indexed_file(unindexed_model_file, indexed_model_file)

    # Instantiate the reader (IndexedThoughtReader)
    thought_reader = reader(indexed_model_file)

    # Now `indexed_idea` contains indexed thoughts and sub-thoughts without modifying the original model
    # Get the main idea
    print(thought_reader.get_idea())

    # Get a list of indexed thoughts
    print(thought_reader.get_thoughts())

    # Get thoughts with descriptions
    print(thought_reader.get_thoughts_and_descriptions())

    # Get sub-thoughts for a specific thought by index
    sub_thoughts = thought_reader.get_sub_thoughts_for_thought(
        thought_index=thought_index
    )
    print(sub_thoughts)

    return sub_thoughts
