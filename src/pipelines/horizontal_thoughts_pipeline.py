"""Pipeline to generate parallell (Horizontal) sub topics based on a given main topic."""

import os
from pathlib import Path
from typing import Union
import logging
import logging_config
from thought_generation.thought_generator import ThoughtGenerator

from project_config import (
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


# Set up logger
logger = logging.getLogger(__name__)


def parellel_thoughts_generation_wt_openai_pipeline(
    idea: str,
    num_of_thoughts: int,
    json_file: Union[Path, str],
    llm_provider: str = "openai",
    model_id: str = GPT_4_TURBO,  # default to 4-turbo (a lot cheaper than 4 but better than 3.5 turbo)
    to_update: bool = False,
) -> None:
    """
    Generates "horizontal thoughts" (sub-topics) for a given idea using
    OpenAI's models and saves the results to a JSON file.

    Horizontal thoughts are sub-topics or concepts generated from the main topic (idea)
    based on clustering and ranking logic.

    Args:
        idea (str): The main idea or topic for which horizontal thoughts will be generated.
        num_thoughts (int): Number of horizontal thoughts to generate.
        json_file (Union[Path, str]): Path to the output JSON file where results will be saved.
        llm_provider (str): The LLM provider to use (default is "openai").
        to_update (bool, optional): If True, overwrite the existing output file. Defaults to False.
            - If `json_file` exists:
                - When `to_update=True`, the file is overwritten.
                - When `to_update=False`, the pipeline is skipped (early return).

    Raises:
        FileNotFoundError: If the specified directory for the output file does not exist.
        ValueError: If an unexpected error occurs during thought generation or file saving.

    Workflow:
        - Verifies if the output file exists; skips processing if `to_update=False`.
        - Ensures the directory for the output file exists.
        - Configures thought generation parameters (clustering and ranking).
        - Instantiates the `ThoughtGenerator` class with OpenAI's models and parameters.
        - Generates horizontal thoughts using the `process_horizontal_thought_generation` method.
        - Saves the results to the specified JSON file.

    Example Usage:
        parellel_thoughts_generation_wt_openai_pipeline(
            idea="Sustainable Energy",
            num_thoughts=10,
            json_file="output/sustainable_energy_thoughts.json"
        )
    """
    logger.info(
        f"Start running horizontal thoughts generation pipeline with {llm_provider}."
    )

    # Ensure json file is Path obj.
    json_file = Path(json_file)

    # Check if the output file path already exist
    if json_file.exists() and not to_update:
        logger.info(f"Output file {json_file} already exists. Skip pipeline.")
        return  # Early return

    # Check if output file's directory exist
    directory = os.path.dirname(json_file)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The file or directory at {directory} does not exist.")

    # Set the thought reduction numbers (cluster, ranking)
    num_clusters = int(round(10 * 0.7))  # reclustered sub-thoughts
    top_n = round(num_clusters * 0.7)  # number of top sub-thoughts to display

    # Process to generate sub-topics/concepts
    thought_generator = ThoughtGenerator(
        llm_provider=llm_provider, model_id=model_id, temperature=0.8
    )  # Instantiate thought_generator class and set the llm parameters

    idea_model = thought_generator.process_horizontal_thought_generation(
        thought=idea,
        num_sub_thoughts=num_of_thoughts,
        num_clusters=num_clusters,
        top_n=top_n,
    )

    logger.info(f"Thoughts created: {idea_model}")

    # Save results to json file
    thought_generator.save_results(idea_model, json_file)

    logger.info(
        f"Finished running horizontal thoughts generation pipeline with {llm_provider}."
    )


# parellel_thoughts_generation_wt_claude_pipeline
def parellel_thoughts_generation_wt_claude_pipeline(
    idea: str,
    num_thoughts: int,
    json_file: Union[Path, str],
    llm_provider: str = "claude",
    model_id: str = CLAUDE_SONNET,
    to_update: bool = False,
) -> None:
    """
    Generates "horizontal thoughts" (sub-topics) for a given idea using
    Anthropic Claude's models and saves the results to a JSON file.

    Horizontal thoughts are sub-topics or concepts generated from
    the main topic (idea) based on clustering and ranking logic.

    Args:
        idea (str): The main idea or topic for which horizontal thoughts will be generated.
        num_thoughts (int): Number of horizontal thoughts to generate.
        json_file (Union[Path, str]): Path to the output JSON file where results will be saved.
        llm_provider (str): The LLM provider to use (default is "claude").
        to_update (bool, optional): If True, overwrite the existing output file. Defaults to False.
            - If `json_file` exists:
                - When `to_update=True`, the file is overwritten.
                - When `to_update=False`, the pipeline is skipped (early return).

    Raises:
        FileNotFoundError: If the specified directory for the output file does not exist.
        ValueError: If an unexpected error occurs during thought generation or file saving.

    Workflow:
        - Verifies if the output file exists; skips processing if `to_update=False`.
        - Ensures the directory for the output file exists.
        - Configures thought generation parameters (clustering and ranking).
        - Instantiates the `ThoughtGenerator` class with Anthropic Claude's models and parameters.
        - Generates horizontal thoughts using the `process_horizontal_thought_generation` method.
        - Saves the results to the specified JSON file.

    Example Usage:
        parellel_thoughts_generation_wt_claude_pipeline(
            idea="Artificial Intelligence",
            num_thoughts=10,
            json_file="output/ai_thoughts.json"
        )
    """
    logger.info(
        f"Start running horizontal thoughts generation pipeline with {llm_provider}."
    )

    # Ensure json file is Path obj.
    json_file = Path(json_file)

    # Check if the output file path already exist
    if json_file.exists() and not to_update:
        logger.info(f"Output file {json_file} already exists. Skip pipeline.")
        return  # Early return

    # Check if output file's directory exist
    directory = os.path.dirname(json_file)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The file or directory at {directory} does not exist.")

    # Set the thought reduction numbers (cluster, ranking)
    num_clusters = int(round(10 * 0.7))  # reclustered sub-thoughts
    top_n = round(num_clusters * 0.7)  # number of top sub-thoughts to display

    # Process to generate sub-topics/concepts
    thought_generator = ThoughtGenerator(
        llm_provider=llm_provider, model_id=model_id, temperature=0.8
    )
    sub_topics = thought_generator.process_horizontal_thought_generation(
        thought=idea,
        num_sub_thoughts=num_thoughts,
        num_clusters=num_clusters,
        top_n=top_n,
    )  # ideally, top_n should be around 4

    logger.info(sub_topics)
    thought_generator.save_results(sub_topics, json_file)

    logger.info(
        f"Finished running horizontal thoughts generation pipeline with {llm_provider}."
    )
