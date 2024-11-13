"""main.py"""

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
from config import (
    rank_of_sub_thoughts_output_0_json,
    array_of_sub_thoughts_output_0_json,
    rank_of_sub_thoughts_output_1_json,
    array_of_sub_thoughts_output_1_json,
)

logger = logging.getLogger(__name__)


def run_pipeline_1a():
    """
    GPT/OpenAI version:
    Run horizontal thoughts pipeline to generate horizontal sub-topics,
    then run vertical thought pipeline to generate vertical sub topics for each.
    """

    logger.info("Starting pipeline 1a: generating array of thoughts with OpenAI.")

    # main_topic = "RTOS (Real-Time Operating System)"
    main_topic = "embedded software development"
    rank_of_thoughts_wt_gpt = rank_of_sub_thoughts_output_0_json
    array_of_thoughts_wt_gpt = array_of_sub_thoughts_output_0_json
    parellel_thoughts_generation_wt_openai_pipeline(
        main_concept=main_topic, num_thoughts=10, json_file=rank_of_thoughts_wt_gpt
    )
    vertical_thought_wt_openai_pipeline(
        input_json_file=rank_of_thoughts_wt_gpt,
        output_json_file=array_of_thoughts_wt_gpt,
    )

    logger.info(
        "Finished running pipeline 1a: generating array of thoughts with OpenAI."
    )


def run_pipeline_1b():
    """
    Claude/Anthropic version:
    Run horizontal thoughts pipeline to generate horizontal sub-topics,
    then run vertical thought pipeline to generate vertical sub topics for each.
    """

    logger.info("Starting pipeline 1b: generating array of thoughts with Claude.")

    # main_topic = "RTOS (Real-Time Operating System)"
    main_topic = "embedded software development"
    rank_of_thoughts_wt_claude = rank_of_sub_thoughts_output_1_json
    array_of_thoughts_wt_claude = array_of_sub_thoughts_output_1_json
    parellel_thoughts_generation_wt_claude_pipeline(
        main_concept=main_topic,
        num_thoughts=10,
        json_file=rank_of_thoughts_wt_claude,
    )
    vertical_thought_wt_claude_pipeline(
        input_json_file=rank_of_thoughts_wt_claude,
        output_json_file=array_of_thoughts_wt_claude,
    )

    logger.info(
        "Finished running pipeline 1b: generating array of thoughts with Claude."
    )


def main():
    """
    Main function for testing and generating sub-thoughts using OpenAI and self-attention.
    """
    # run_pipeline_1a()
    run_pipeline_1b()


if __name__ == "__main__":
    main()
