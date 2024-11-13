"""Pipeline to generate parallell (Horizontal) sub topics based on a given main topic."""

import os
from pathlib import Path
from typing import Union
import logging
import logging_config
from thought_generation.thought_generator import ThoughtGenerator

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


# Set up logger
logger = logging.getLogger(__name__)


def parellel_thoughts_generation_wt_openai_pipeline(
    idea: str,
    num_thoughts: int,
    json_file: Union[Path, str],
    llm_provider: str = "openai",
):
    """Pipeline to generate "horizontal thoughts"""
    logger.info(
        f"Start running horizontal thoughts generation pipeline with {llm_provider}."
    )

    # Ensure json file is Path obj.
    json_file = Path(json_file)

    # Check if the output file path already exist
    if json_file.exists():
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
        llm_provider=llm_provider, model_id=GPT_4_TURBO, temperature=0.8
    )  # Instantiate thought_generator class and set the llm parameters

    idea_model = thought_generator.process_horizontal_thought_generation(
        thought=idea, num_sub_thoughts=num_thoughts, num_clusters=6, top_n=top_n
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
):
    """Pipeline to generate "horizontal thoughts"""
    logger.info(
        f"Start running horizontal thoughts generation pipeline with {llm_provider}."
    )

    # Ensure json file is Path obj.
    json_file = Path(json_file)

    # Check if the output file path already exist
    if json_file.exists():
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
        llm_provider=llm_provider, model_id=CLAUDE_SONNET, temperature=0.8
    )
    sub_topics = thought_generator.process_horizontal_thought_generation(
        thought=idea, num_sub_thoughts=num_thoughts, num_clusters=6, top_n=top_n
    )  # ideally, top_n should be around 4

    logger.info(sub_topics)
    thought_generator.save_results(sub_topics, json_file)

    logger.info(
        f"Finished running horizontal thoughts generation pipeline with {llm_provider}."
    )
