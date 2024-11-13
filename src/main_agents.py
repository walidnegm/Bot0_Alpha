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


def testing():
    json_file = r"C:\github\Bot0_Release1\backend\input_output\thought_generation\openai_output\array_of_thoughts_output_openai_embedded_software_development.json"
    thought_reader = ThoughtReader(json_file)
    idea = thought_reader.get_idea()
    thoughts = thought_reader.get_thoughts_and_descriptions()

    print(f"idea: {idea}")

    print(f"thoughts: \n{thoughts}")


def main():
    run_topic_conversation_pipeline(thought_file, memory_file)


async def main_async():
    asyncio.run(run_topic_conversation_pipeline_async(thought_file, memory_file))


if __name__ == "__main__":
    # main()
    testing()
