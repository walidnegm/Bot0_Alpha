"""
TODO: interview dialog portion; to be integrated later into the central main.py file.
"""

from pathlib import Path
import asyncio
import json
from utils.generic_utils import read_from_json_file

from pipelines.interview_pipeline_async import (
    interview_pipeline_async,
)
from src.pipelines.thought_processing_pipeline import indexed_thought_reading_pipeline

from utils.generic_utils import pretty_print_json
from thought_generation.thought_reader import ThoughtReader


# pylint: disable=next-line
from config import INDEXED_MODELS_CLAUDE_DIR, INDEXED_MODELS_CLAUDE_DIR


# thought_file = array_of_thoughts_claude_json_output_file
thought_file = array_of_thoughts_openai_output_file

memory_file = Path(
    r"C:\github\Bot0_Release1\backend\input_output\memory\memory_testing.json"
)


async def main_async():
    asyncio.run(interview_pipeline_async(thought_file, memory_file))


def run_indexed_thought_reading_pipeline():

    openai_file_without_index = array_of_thoughts_openai_output_file
    openai_file_with_index = array_of_thoughts_with_index_openai_output_file
    claude_file_without_index = array_of_thoughts_claude_output_file
    claude_file_with_index = array_of_thoughts_with_index_claude_output_file

    # # Claude
    # indexed_thought_reading_pipeline(
    #     unindexed_model_file=claude_file_without_index,
    #     indexed_model_file=claude_file_with_index,
    #     thought_index=0,
    # )

    # OpenAI
    indexed_thought_reading_pipeline(
        unindexed_model_file=openai_file_without_index,
        indexed_model_file=openai_file_with_index,
        thought_index=0,
    )


if __name__ == "__main__":
    # main()
    run_indexed_thought_reading_pipeline()
