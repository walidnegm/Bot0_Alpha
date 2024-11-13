"""
Filename: thought_reader.py

Utilities or helpers for agent classes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError

from models.json_validation_models import IdeaJSONModel, ThoughtJSONModel
from utils.generic_utils import read_from_json_file


from typing import List, Optional
from pathlib import Path
import json
from pydantic import ValidationError
import logging

# Ensure your Pydantic models are imported
# from your_models_file import IdeaJSONModel, ThoughtJSONModel, SubThoughtJSONModel

logger = logging.getLogger(__name__)


class ThoughtReader:
    """
    A class for reading and interacting with structured thoughts data from a JSON file.

    The ThoughtReader class loads and validates a JSON file, initializing it as an instance of
    IdeaJSONModel. It provides methods to retrieve the main idea, list of thoughts, descriptions,
    and detailed information on sub-thoughts for a specific thought. This class is useful for
    applications that need to sequentially process or reference structured thought data.

    Attributes:
        json_file (Path): The path to the JSON file containing thoughts data.
        idea_instance (IdeaJSONModel): An instance of IdeaJSONModel initialized with loaded data.
    """

    def __init__(self, json_file: Union[Path, str]):
        """
        Initializes the ThoughtReader with the path to a JSON file and loads the data.

        Args:
            json_file (Union[Path, str]): The path to the JSON file containing the data.
        """
        self.json_file = Path(json_file)  # if json_file is str., turn into Path obj.
        self.idea_instance: IdeaJSONModel = self._load_and_validate_data()

    def _load_and_validate_data(self):
        """
        Loads the JSON data from the specified file path and initializes it as an IdeaJSONModel.

        Returns:
            IdeaJSONModel: The validated IdeaJSONModel instance containing thoughts data.

        Raises:
            ValueError: If the data structure is invalid or does not meet expected schema.
            Exception: For any other errors during file reading or data processing.
        """
        try:
            data = read_from_json_file(self.json_file)
            # with open(self.json_file, "r") as f:
            #     data = json.load(f)

            # TODO: debugging; delete later
            # Debugging step: Print or log the structure of the loaded data
            logger.debug("Loaded data: %s", data)

            # Check if data is None or not a dictionary with expected keys
            if not isinstance(data, dict):
                logger.error("Data is not a dictionary in file: %s", self.json_file)
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
            idea_instance = IdeaJSONModel(
                **data
            )  #! DO NOT USE PYD'S MODEL_VALIDATE METHOD (ALWAYS USE DIRECT INSTANTIATION!)
            logger.info("Data loaded and initialized successfully.")
            return idea_instance

        except ValidationError as e:
            # Log the validation error details for debugging
            logger.error(
                "Data validation failed: %s", e.json()
            )  # Shows error details in JSON format
            raise ValueError("Data validation failed") from e
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            raise e

    def get_idea(self) -> Optional[str]:
        """Fetches the main idea/theme of the data."""
        if self.idea_instance:
            return self.idea_instance.idea
        logger.warning("Data not loaded. Call load_data() first.")
        return None

    def get_thoughts(self) -> List[dict]:
        """Gets a list of thoughts (name only)"""
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        thought_names = [thought.thought for thought in self.idea_instance.thoughts]
        return thought_names

    def get_thoughts_and_descriptions(self) -> List[dict]:
        """
        Gets a list of thoughts along with their descriptions.

        Returns:
            List[dict]: A list of dictionaries, each containing a thought's name and description.

        Example:
            thought_reader = ThoughtReader("path/to/file.json")
            thoughts_data = thought_reader.get_thoughts_and_descriptions()
            for thought in thoughts_data:
                print(f"{thought['thought']}: {thought['description']}")
        """
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        thoughts_data = [
            {"thought": thought.thought, "description": thought.description}
            for thought in self.idea_instance.thoughts
        ]
        return thoughts_data

    def get_sub_thoughts_for_thought(self, thought_name: str) -> List[dict]:
        """
        Gets a list of all sub-thoughts and their descriptions for a specific thought.
        The order of sub-thoughts is preserved as in the JSON data.

        Args:
            thought_name (str): The name of the thought for which to retrieve sub-thoughts.

        Returns:
            List[dict]: A list of dictionaries, each containing a sub-thought's details.
            Each dictionary contains the name, description, importance, and connection to next.

        Example:
            thought_reader = ThoughtReader("path/to/file.json")
            sub_thoughts = thought_reader.get_sub_thoughts_for_thought("Definition and Scope")
            for sub in sub_thoughts:
                print(f"{sub['name']}: {sub['description']} (Importance: {sub['importance']})")
        """
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        for thought in self.idea_instance.thoughts:
            if thought.thought == thought_name:
                return [
                    {
                        "name": sub_thought.name,
                        "description": sub_thought.description,
                        "importance": sub_thought.importance,
                        "connection_to_next": sub_thought.connection_to_next,
                    }
                    for sub_thought in thought.sub_thoughts or []
                ]

        logger.warning(f"Thought '{thought_name}' not found.")
        return []


def read_thoughts(json_file: Union[Path, str]) -> ThoughtJSONModel:

    # Ensure json_file path is changed to Path obj.
    json_file = Path(json_file)

    # Read JSON file
    # with open(json_file) as f:
    #     json_data = json.load(
    #         f
    #     )  # No need to do more error handling b/c the function has robust validation already
    json_data = read_from_json_file(json_file)
    print(json_data)

    try:
        # Parse the JSON data into the IdeaJSONModel
        idea_instance = IdeaJSONModel.model_validate(json_data)
        return idea_instance  # This will show the loaded and validated model instance
    except ValidationError as e:
        raise ValueError("Validation Error:", e)


def get_idea(data: List[Dict[str, Any]]) -> List[str]:
    """
    Retrieve idea (main concepts or theme) from the JSON data.

    Args:
        data (List[Dict[str, Any]]): The JSON data containing main thoughts and sub-thoughts.

    Returns:
        List[str]: A list of main thought names (main concepts).
    """
    return [item["thought"] for item in data]


def get_thoughts(data: List[Dict[str, Any]]) -> List[str]:
    """
    Retrieve a list of main thoughts (concepts) from the JSON data.

    Args:
        data (List[Dict[str, Any]]): The JSON data containing main thoughts and sub-thoughts.

    Returns:
        List[str]: A list of main thought names (main concepts).
    """
    return [item["thought"] for item in data]


def get_sub_thoughts(
    data: List[Dict[str, Any]], main_thought: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve the list of sub-thoughts (sub-concepts) for a specified main thought.

    Args:
        - data (List[Dict[str, Any]]): The JSON data containing main thoughts and sub-thoughts.
        - main_thought (str): The main thought (main concept) to retrieve sub-thoughts for.

    Returns:
        Optional[List[Dict[str, Any]]]: List of sub-thoughts or None if main thought is not found.
    """
    for item in data:
        if item["concept"] == main_thought:
            return item.get("sub_concepts")
    return None


def get_next_sub_thought(
    sub_thoughts: List[Dict[str, Any]], current_index: int
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Retrieve the next sub-thought in the list of sub-thoughts.

    Args:
        sub_thoughts (List[Dict[str, Any]]): The list of sub-thoughts.
        current_index (int): The current index of the sub-thought.

    Returns:
        Tuple[Optional[Dict[str, Any]], int]:
            The next sub-thought dictionary and the updated index.
            If at the end, returns None and -1.
    """
    if current_index + 1 < len(sub_thoughts):
        return sub_thoughts[current_index + 1], current_index + 1
    return None, -1  # End of sub-thoughts


# Example usage:
# Assuming `data` is loaded from JSON in the provided format
# main_thoughts = get_main_thoughts(data)
# selected_main_thought = main_thoughts[0]  # Select the first main thought
# sub_thoughts = get_sub_thoughts(data, selected_main_thought)
# if sub_thoughts:
#     next_sub_thought, next_index = get_next_sub_thought(sub_thoughts, 0)

# Example usage:
# organized_thoughts = organize_thoughts(data)  # Assuming data is loaded and organized as shown
# main_thoughts = get_main_thoughts(organized_thoughts)
# selected_main_thought = main_thoughts[0]  # Picking the first main thought as an example
# sub_thoughts = get_sub_thoughts(organized_thoughts, selected_main_thought)
# next_sub_thought = get_next_sub_thought(sub_thoughts, 0)  # Get first sub-thought


# Example usage:
# data = load_thoughts("array_of_thoughts_output.json")
# organized_thoughts = organize_thoughts(data)
# next_sub_thought, next_index = get_next_sub_thought(organized_thoughts, "System Core Management", 0)
