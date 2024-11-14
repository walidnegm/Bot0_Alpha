"""
Module docstring: TBA
"""

from pathlib import Path
import asyncio
import json
from utils.generic_utils import read_from_json_file

from pipelines.topic_conversation_pipeline_async import (
    run_topic_conversation_pipeline_async,
)
from pipelines.topic_conversation_pipeline import run_topic_conversation_pipeline
from pipelines.thought_reading_pipeline import thought_reading_with_index_pipeline
from thought_generation.thought_reader import read_thoughts
from utils.generic_utils import pretty_print_json
from thought_generation.thought_reader import ThoughtReader


# pylint: disable=next-line
from config import (
    rank_of_thoughts_openai_json_output_file,
    array_of_thoughts_openai_json_output_file,
    rank_of_thoughts_claude_output_file,
    array_of_thoughts_claude_json_output_file,
)

# thought_file = array_of_thoughts_claude_json_output_file
thought_file = array_of_thoughts_openai_json_output_file

memory_file = Path(
    r"C:\github\Bot0_Release1\backend\input_output\memory\memory_testing.json"
)


def main():
    run_topic_conversation_pipeline(thought_file, memory_file)


async def main_async():
    asyncio.run(run_topic_conversation_pipeline_async(thought_file, memory_file))


def run_indexed_thought_reading_pipeline():
    thought_reading_with_index_pipeline()


if __name__ == "__main__":
    # main()
    run_indexed_thought_reading_pipeline()
