# transformation_utils.py

from pathlib import Path
from typing import List, Union
from pydantic import ValidationError
import logging
import logging_config

from models.thought_models import IdeaJSONModel  # Import the original IdeaJSONModel
from models.indexed_thought_models import (
    IndexedIdeaJSONModel,
    IndexedThoughtJSONModel,
    IndexedSubThoughtJSONModel,
)
from utils.generic_utils import read_from_json_file, save_to_json_file


# Set logger
logger = logging.getLogger(__name__)


def transform_to_indexed_idea_model(idea_model: IdeaJSONModel) -> IndexedIdeaJSONModel:
    """
    Transforms an IdeaJSONModel (w/o indices) instance to an IndexedIdeaJSONModel with
    indices added.

    Args:
        idea_model (IdeaJSONModel): The original IdeaJSONModel instance.

    Returns:
        IndexedIdeaJSONModel: A new IndexedIdeaJSONModel with thought and
        sub-thought indices added.

    *Workflow:
        * Unpacking:
            - The function first unpacks the original IdeaJSONModel instance, which contains
            a dictionary-like structure with 'idea' (a string) and 'thoughts' (a list of
            ThoughtJSONModel instances).

        * Iterating through Thoughts (a list):
            - It loops through the 'thoughts' list, treating each thought as
            a ThoughtJSONModel instance.
            - While iterating, it adds a `thought_index` to each thought using the `enumerate`
            function, which provides an internal index for each list item.

        * Unpacking Sub-Thoughts:
            - For each thought, it unpacks the 'sub_thoughts' list, which contains
            SubThoughtJSONModel instances (or dictionaries if converted).
            - It loops through 'sub_thoughts', adding a
            `sub_thought_index` to each one in the same way, using the 'enumerate' function
            to track the internal index of each sub-thought.

        * Re-assembling with Indices:
            - After adding indices, it re-assembles the data into a new IndexedIdeaJSONModel
            structure, with each thought and sub-thought now including their respective
            'thought_index' and `sub_thought_index`.
            - This re-assembly preserves the original data structure while adding useful
            metadata (indices) for tracking and sequential processing.
    """

    # Create indexed thoughts and sub-thoughts
    indexed_thoughts = []
    for thought_index, thought in enumerate(idea_model.thoughts or []):
        indexed_sub_thoughts = [
            IndexedSubThoughtJSONModel(
                sub_thought_index=sub_index,
                name=sub_thought.name,
                description=sub_thought.description,
                importance=sub_thought.importance,
                connection_to_next=sub_thought.connection_to_next,
            )
            for sub_index, sub_thought in enumerate(thought.sub_thoughts or [])
        ]
        indexed_thoughts.append(
            IndexedThoughtJSONModel(
                thought_index=thought_index,
                thought=thought.thought,
                description=thought.description,
                sub_thoughts=indexed_sub_thoughts,
            )
        )

    # Create and return the IndexedIdeaModel
    return IndexedIdeaJSONModel(idea=idea_model.idea, thoughts=indexed_thoughts)


def generate_indexed_model_file(
    unindexed_model_file: Union[Path, str],
    indexed_model_file: Union[Path, str],
    to_update: bool = False,
) -> None:
    """
    Reads an unindexed JSON model file, transforms it to an indexed model,
    and saves the indexed version to a new JSON file.

    Args:
        - unindexed_model_file (Union[Path, str]): Path to the JSON file containing
        the unindexed model.
        - indexed_model_file (Union[Path, str]): Path where the indexed model JSON file
        will be saved.
        - to_update (bool): Determine whether to update the output file (indexed_model_file)
            Default to False.
            When the output file exists already,
                if to_update is True: replace it with the new file
                if False: early return -> skip

    Raises:
        - ValueError: If the data structure in the input file is invalid or does not meet
        the required schema.
        - FileNotFoundError: If the specified input file is not found.
        - Exception: For any unexpected errors during file reading, transformation,
        or saving.
    """
    try:
        # Ensure the paths are Path objects
        unindexed_model_file, indexed_model_file = Path(unindexed_model_file), Path(
            indexed_model_file
        )

        # Check if output file exist already
        if indexed_model_file.exists() and not to_update:
            logger.info(
                f"Indexed model file ({indexed_model_file}) already exists. Skip pipeline."
            )
            return

        # Check if input file exists
        if not unindexed_model_file.exists():
            logger.info("Input file not found: %s", unindexed_model_file)
            raise FileNotFoundError(f"Input file '{unindexed_model_file}' not found.")

        # Read data from JSON file
        data = read_from_json_file(unindexed_model_file)
        if data is None:
            logger.error("No data found in file: %s", unindexed_model_file)
            raise ValueError("No data found in the specified JSON file.")

        # Check if data is a dictionary with expected keys
        if not isinstance(data, dict):
            logger.error("Data is not a dictionary in file: %s", unindexed_model_file)
            raise ValueError(
                "Expected data to be a dictionary with 'idea' and 'thoughts' keys"
            )

        if "idea" not in data or "thoughts" not in data:
            logger.error(
                "Data structure mismatch: expected 'idea' and 'thoughts' keys in file: %s",
                unindexed_model_file,
            )
            raise ValueError(
                "Data structure mismatch: expected a dictionary with 'idea' and 'thoughts' keys"
            )

        logger.info(
            "Unindexed idea model (%s) read successfully.", unindexed_model_file
        )

        # Instantiate IdeaJSONModel to validate the unindexed data structure
        try:
            unindexed_idea_model = IdeaJSONModel(
                **data
            )  #! do not use model_validate - direct instantiation only!
            logger.info("Data validated successfully as IdeaJSONModel.")
        except ValidationError as e:
            logger.error("Data validation failed: %s", e.json())
            raise ValueError(
                "Data validation failed: input JSON does not match IdeaJSONModel schema."
            ) from e

        # Transform the validated model to an indexed model
        indexed_idea_model = transform_to_indexed_idea_model(unindexed_idea_model)
        logger.info("Transformation to IndexedIdeaJSONModel successful.")

        # Ensure the output file's directory (directory for the indexed model file) exists
        if not indexed_model_file.parent.exists():
            logger.info(
                "Directory for indexed model file does not exist. Creating: %s",
                indexed_model_file.parent,
            )
            indexed_model_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the indexed model to JSON
        try:
            save_to_json_file(
                data=indexed_idea_model.model_dump(), file_path=indexed_model_file
            )
            logger.info("Indexed model saved to %s successfully.", indexed_model_file)
        except Exception as e:
            logger.error(
                "Failed to save the indexed model to %s: %s", indexed_model_file, str(e)
            )
            raise Exception(
                f"Failed to save the indexed model to {indexed_model_file}"
            ) from e

    except FileNotFoundError as e:
        logger.error("FileNotFoundError: %s", str(e))
        raise
    except ValueError as e:
        logger.error("ValueError: %s", str(e))
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        raise
