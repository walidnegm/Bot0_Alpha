import os
from pathlib import Path
import logging
import logging_config
import json
from typing import Optional, Union
from thought_generation.thought_generator import ThoughtGenerator
from utils.generic_utils import read_from_json_file

from config import (
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    CLAUDE_HAIKU,
    GPT_35_TURBO,
    GPT_35_TURBO_16K,
    GPT_4,
    GPT_4_TURBO,
    GPT_4_TURBO_32K,
    GPT_4O,
)

logger = logging.getLogger(__name__)


def vertical_thought_wt_openai_pipeline(
    input_json_file: Union[str, Path],
    output_json_file: Union[Path, str],
    llm_provider: str = "openai",
    model_id: str = GPT_4_TURBO,
    to_update: bool = False,
) -> None:
    """
    *OpenAI pipeline
    Pipeline to generate "vertical thoughts" based on a main concept and
    save the results to a JSON file.

    This function generates a specified number of sub-thoughts for a given main concept following a
    vertical thought process. It ensures the output is saved to a specified JSON file. The function
    uses an external thought generation model (e.g., GPT-4 turbo) to create the sub-thoughts,
    considering the context provided, and checks for the existence of the target file directory
    before proceeding.

    Args:
        - input_json_file (str.): JSON file w/t topics/concepts generated in the horizontal thought
        creation process from the OpenAI input output data pipeline.
        - output_json_file (str): output JSON file path from the OpenAI input output data pipeline.
        - to_update (bool): Determine whether to update the output file (indexed_model_file)
          Default to False.
          When the output file exists already,
            if to_update is True: replace it with the new file
            if False: early return -> skip

    Return: None

    Raises:
        FileNotFoundError: If the specified directory for the JSON file does not exist.

    Example:
        vertical_thought_pipeline(
            input_json_file = "path to horizontal thought json file")
            output_json_file = "path to array of thoughts json file to be saved")
        )
    """
    logger.info(f"Starting vertical thoughts generation pipeline with {llm_provider}.")

    input_json_file, output_json_file = Path(input_json_file), Path(output_json_file)

    # Check if the output file path already exist
    if output_json_file.exists() and not to_update:
        logger.info(f"Output file {output_json_file} already exists. Skip pipeline.")
        return  # Early return

    # Check if the input file paths exist or not
    if not input_json_file.exists():
        raise FileNotFoundError(f"Input data file {input_json_file} does not exist.")

    # Read horizontal sub thoughts from JSON
    thoughts_data = read_from_json_file(input_json_file)

    logger.info(f"Data is read from JSON file:\n{thoughts_data}")

    # Instantiate thought_generator class and process the method to create sub thoughts
    # by iterating through the already created horizontal/parallel thoughts
    thought_generator = ThoughtGenerator(
        llm_provider=llm_provider, model_id=model_id, temperature=0.8
    )
    array_of_thoughts = thought_generator.generate_array_of_thoughts(
        input_data=thoughts_data,
    )

    logger.info(f"sub_thought_list: \n{array_of_thoughts}")  # debugging

    # Save results to file if json_file is provided
    if output_json_file:
        thought_generator.save_results(array_of_thoughts, output_json_file)

    logger.info(f"Finished vertical thoughts generation pipeline with {llm_provider}.")


def vertical_thought_wt_claude_pipeline(
    input_json_file: Union[Path, str],
    output_json_file: Union[Path, str] = None,
    llm_provider: str = "claude",
    model_id: str = CLAUDE_SONNET,
    to_update: bool = True,
) -> None:
    """
    *Claude pipeline
    Pipeline to generate "vertical thoughts" based on a main concept and
    save the results to a JSON file.

    This function generates a specified number of sub-thoughts for a given main concept following a
    vertical thought process. It ensures the output is saved to a specified JSON file. The function
    uses an external thought generation model (e.g., GPT-4 turbo) to create the sub-thoughts,
    considering the context provided, and checks for the existence of the target file directory
    before proceeding.

    Args:
        - input_json_file (str.): JSON file w/t topics/concepts generated in the horizontal thought
        creation process from the claude input output data pipeline.
        - output_json_file (str): output JSON file path from the claude input output data pipeline.
        - to_update (bool): Determine whether to update the output file (indexed_model_file)
          Default to False.
          When the output file exists already,
            if to_update is True: replace it with the new file
            if False: early return -> skip

    Return: None

    Raises:
        FileNotFoundError: If the specified directory for the JSON file does not exist.

    Example:
        vertical_thought_pipeline(
            input_json_file = "path to horizontal thought json file")
            output_json_file = "path to array of thoughts json file to be saved")
        )
    """
    logger.info(f"Starting vertical thoughts generation pipeline with {llm_provider}.")

    input_json_file, output_json_file = Path(input_json_file), Path(output_json_file)

    # Check if the output file path already exist
    if output_json_file.exists():
        logger.info(f"Output file {output_json_file} already exists. Skip pipeline.")
        return  # Early return

    # Check if input file paths exist or not
    if not input_json_file.exists() and not to_update:
        raise FileNotFoundError(f"Input data file {input_json_file} does not exist.")

    # Read horizontal sub thoughts from JSON
    thoughts_data = read_from_json_file(input_json_file)

    logger.info(f"Data is read from JSON file:\n{thoughts_data}")

    # Instantiate thought_generator class and process the method to create sub thoughts
    # by iterating through the already created horizontal/parallel thoughts
    thought_generator = ThoughtGenerator(
        llm_provider=llm_provider, model_id=model_id, temperature=0.8
    )
    array_of_thoughts = thought_generator.generate_array_of_thoughts(
        input_data=thoughts_data,
    )

    logger.info(f"sub_thought_list: \n{array_of_thoughts}")  # debugging

    # Save results to file if json_file is provided
    if output_json_file:
        thought_generator.save_results(array_of_thoughts, output_json_file)

    logger.info(f"Finished vertical thoughts generation pipeline with {llm_provider}.")
