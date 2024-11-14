"""
Filename: thought_reader.py

Utilities or helpers for agent classes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError
from pathlib import Path
import json
import logging
import logging_config

from models.thought_models import IdeaJSONModel, ThoughtJSONModel
from models.indexed_thought_models import (
    IndexedIdeaJSONModel,
)  # Import the new indexed model
from utils.generic_utils import read_from_json_file


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


class IndexedThoughtReader:
    """
    A class for reading and interacting with indexed thoughts data from a JSON file.

    The IndexedThoughtReader class loads and validates a JSON file, initializing it as an instance of
    IndexedIdeaModel. It provides methods to retrieve the main idea, list of indexed thoughts, descriptions,
    and detailed information on indexed sub-thoughts for a specific thought.

    Attributes:
        json_file (Path): The path to the JSON file containing thoughts data.
        idea_instance (IndexedIdeaModel): An instance of IndexedIdeaModel initialized with loaded data.
    """

    def __init__(self, json_file: Union[Path, str]):
        """
        Initializes the IndexedThoughtReader with the path to a JSON file and loads the data.

        Args:
            json_file (Union[Path, str]): The path to the JSON file containing the data.
        """
        self.json_file = Path(json_file)
        self.idea_instance: IndexedIdeaModel = self._load_and_validate_data()

    def _load_and_validate_data(self) -> IndexedIdeaJSONModel:
        """
        Loads the JSON data from the specified file path and initializes it as an IndexedIdeaModel.

        Returns:
            IndexedIdeaModel: The validated IndexedIdeaModel instance containing thoughts data.

        Raises:
            ValueError: If the data structure is invalid or does not meet the expected schema.
            Exception: For any other errors during file reading or data processing.
        """
        try:
            data = read_from_json_file(self.json_file)

            logger.debug("Loaded data: %s", data)

            if not isinstance(data, dict):
                logger.error("Data is not a dictionary in file: %s", self.json_file)
                raise ValueError(
                    "Expected data to be a dictionary with 'idea' and 'thoughts' keys"
                )

            if "idea" not in data or "thoughts" not in data:
                logger.error(
                    "Data structure mismatch: expected a dictionary with 'idea' and 'thoughts'"
                )
                raise ValueError(
                    "Expected data to be a dictionary with 'idea' and 'thoughts' keys"
                )

            idea_instance = IndexedIdeaJSONModel(**data)
            logger.info("Data loaded and initialized successfully.")
            return idea_instance

        except ValidationError as e:
            logger.error("Data validation failed: %s", e.json())
            raise ValueError("Data validation failed") from e
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            raise e

    def get_idea(self) -> Optional[str]:
        """Fetches the main idea/theme of the data."""
        if self.idea_instance:
            return self.idea_instance.idea
        logger.warning("Data not loaded.")
        return None

    def get_thoughts(self) -> List[dict]:
        """Gets a list of thoughts with their indices and names."""
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        thoughts_data = [
            {"thought_index": thought.thought_index, "thought": thought.thought}
            for thought in self.idea_instance.thoughts
        ]
        return thoughts_data

    def get_thoughts_and_descriptions(self) -> List[dict]:
        """
        Gets a list of thoughts along with their descriptions and indices.

        Returns:
            List[dict]: A list of dictionaries, each containing a thought's index, name, and description.
        """
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        thoughts_data = [
            {
                "thought_index": thought.thought_index,
                "thought": thought.thought,
                "description": thought.description,
            }
            for thought in self.idea_instance.thoughts
        ]
        return thoughts_data

    def get_sub_thoughts_for_thought(self, thought_index: int) -> List[dict]:
        """
        Gets a list of all sub-thoughts and their descriptions for a specific thought based on index.

        Args:
            thought_index (int): The index of the thought for which to retrieve sub-thoughts.

        Returns:
            List[dict]: A list of dictionaries, each containing a sub-thought's index, name, description,
                        importance, and connection to the next sub-thought.
        """
        if not self.idea_instance or not self.idea_instance.thoughts:
            logger.warning("Data not loaded or no thoughts available.")
            return []

        for thought in self.idea_instance.thoughts:
            if thought.thought_index == thought_index:
                return [
                    {
                        "sub_thought_index": sub_thought.sub_thought_index,
                        "name": sub_thought.name,
                        "description": sub_thought.description,
                        "importance": sub_thought.importance,
                        "connection_to_next": sub_thought.connection_to_next,
                    }
                    for sub_thought in thought.sub_thoughts or []
                ]

        logger.warning(f"Thought with index '{thought_index}' not found.")
        return []
