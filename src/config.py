""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
...
)

"""

# config.py
from pathlib import Path
from utils.find_project_root import find_project_root

# Base/Root Directory
BASE_DIR = Path(find_project_root())


# Input/Output Directory
INPUT_OUTPUT_DIR = BASE_DIR / "backend" / "input_output"  # input/output data folder

THOUGHT_GENERATION_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "thought_generation"
THOUGHT_GENERATION_OPENAI_OUTPUT_DIR = (
    THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "openai_output"
)
THOUGHT_GENERATION_CLAUDE_OUTPUT_DIR = (
    THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "claude_output"
)

# JSON output file paths: OpenAI
rank_of_thoughts_openai_json_output_file = (
    THOUGHT_GENERATION_OPENAI_OUTPUT_DIR / "rank_of_thoughts_output_openai.json"
)
array_of_thoughts_openai_json_output_file = (
    THOUGHT_GENERATION_OPENAI_OUTPUT_DIR / "array_of_thoughts_output_openai.json"
)

# JSON output file paths: Claude
rank_of_thoughts_claude_output_file = (
    THOUGHT_GENERATION_CLAUDE_OUTPUT_DIR / "rank_of_thoughts_output_claude.json"
)
array_of_thoughts_claude_json_output_file = (
    THOUGHT_GENERATION_CLAUDE_OUTPUT_DIR / "array_of_thoughts_output_claude.json"
)
# rank_of_sub_thoughts_output_2_json = (
#     THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "parallel_sub_thoughts_output_2.json"
# )
# array_of_sub_thoughts_output_2_json = (
#     THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "array_of_thoughts_output_2.json"
# )

MEMORY_DIR = INPUT_OUTPUT_DIR / "memory"


# *LLM Models
# Anthropic (Claude) models
CLAUDE_OPUS = "claude-3-opus-20240229"
CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
CLAUDE_HAIKU = "claude-3-haiku-20240307"

# OpenAI models
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4"
GPT_4_TURBO = "gpt-4-turbo"
GPT_4_TURBO_32K = "gpt-4-turbo-32k"
GPT_4O = "gpt-4o"
