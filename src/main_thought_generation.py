"""
* main_thought_generation.py.

TODO: This is temp... To be integrated into main.py later...

"""

from typing import Union
from pathlib import Path
import logging
import logging_config

from pipelines.horizontal_thoughts_pipeline import (
    parellel_thoughts_generation_wt_openai_pipeline,
    parellel_thoughts_generation_wt_claude_pipeline,
)
from pipelines.vertical_thoughts_pipeline import (
    vertical_thought_wt_openai_pipeline,
    vertical_thought_wt_claude_pipeline,
)
from utils.generic_utils import insert_file_name_suffix

from config import (
    # THOUGHT_GENERATION_OPENAI_OUTPUT_DIR,
    # THOUGHT_GENERATION_CLAUDE_OUTPUT_DIR,
    rank_of_thoughts_openai_json_output_file,
    array_of_thoughts_openai_json_output_file,
    rank_of_thoughts_claude_output_file,
    array_of_thoughts_claude_json_output_file,
)


# Setup logger
logger = logging.getLogger(__name__)

ideas = [
    "embedded software development",
    "embedded software development in automotive",
    "embedded software development in aerospace",
]
# other ideas:
# "RTOS (Real-Time Operating System)"


def run_pipeline_1a():  # OpenAI pipeline
    """
    GPT/OpenAI version:
    Run horizontal thoughts pipeline to generate horizontal sub-topics,
    then run vertical thought pipeline to generate vertical sub topics for each.
    """

    logger.info("Starting pipeline 1a: generating array of thoughts with OpenAI.")

    idea = ideas[0]

    # Insert "idea" at the end of output json file
    suffix = f"_{idea.replace(' ', '_')}"  # replace " " w/t "_" add "_" in front

    rank_of_thoughts_file = insert_file_name_suffix(
        rank_of_thoughts_openai_json_output_file, suffix
    )
    logger.info(f"rank_of_thoughts_file: {rank_of_thoughts_file}")  # debugging

    array_of_thoughts_file = insert_file_name_suffix(
        array_of_thoughts_openai_json_output_file, suffix
    )

    logger.info(f"array_of_thoughts_file: {array_of_thoughts_file}")  # debugging

    parellel_thoughts_generation_wt_openai_pipeline(
        idea=idea, num_thoughts=10, json_file=rank_of_thoughts_file
    )
    vertical_thought_wt_openai_pipeline(
        input_json_file=rank_of_thoughts_file,
        output_json_file=array_of_thoughts_file,
    )

    logger.info(
        "Finished running pipeline 1a: generating array of thoughts with OpenAI."
    )


def run_pipeline_1b():  # Claude pipeline
    """
    Claude/Anthropic version:
    Run horizontal thoughts pipeline to generate horizontal sub-topics,
    then run vertical thought pipeline to generate vertical sub topics for each.
    """

    logger.info("Starting pipeline 1b: generating array of thoughts with Claude.")

    # main_topic = "RTOS (Real-Time Operating System)"
    idea = ideas[0]

    # Insert "idea" at the end of output json file
    suffix = f"_{idea.replace(' ', '_')}"  # replace " " w/t "_" add "_" in front

    rank_of_thoughts_file = insert_file_name_suffix(
        rank_of_thoughts_claude_output_file, suffix
    )
    array_of_thoughts_file = insert_file_name_suffix(
        array_of_thoughts_claude_json_output_file, suffix
    )

    parellel_thoughts_generation_wt_claude_pipeline(
        idea=idea,
        num_thoughts=10,
        json_file=rank_of_thoughts_file,
    )
    vertical_thought_wt_claude_pipeline(
        input_json_file=rank_of_thoughts_file,
        output_json_file=array_of_thoughts_file,
    )

    logger.info(
        "Finished running pipeline 1b: generating array of thoughts with Claude."
    )


def main():
    """
    Main function for testing and generating sub-thoughts using OpenAI and self-attention.
    """
    run_pipeline_1a()
    run_pipeline_1b()


if __name__ == "__main__":
    main()
