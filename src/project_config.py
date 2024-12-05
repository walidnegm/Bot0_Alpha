""" 
Data Input/Output dir/file configuration 
"""

# config.py
from pathlib import Path
from utils.find_project_root import find_project_root

# Base/Root Directory
BASE_DIR = Path(find_project_root())

# *Input/Output Directory
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder


# Thought generation direcotry
THOUGHT_GENERATION_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "thought_generation"

# *OpenAI directories: openai thought gen/ideas/models wt indexes, models wo indexes
OPENAI_THOUGHT_GENERATION_DIR = (
    THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "openai_thought_generation"
)
OPENAI_UNINDEXED_MODELS_DIR = OPENAI_THOUGHT_GENERATION_DIR / "models_without_indexes"
OPENAI_INDEXED_MODELS_DIR = OPENAI_THOUGHT_GENERATION_DIR / "models_with_indexes"

# *Claude directories: claude thought gen/ideas/models wt indexes, models wo indexes
CLAUDE_THOUGHT_GENERATION_DIR = (
    THOUGHT_GENERATION_INPUT_OUTPUT_DIR / "claude_thought_generation"
)
CLAUDE_UNINDEXED_MODELS_DIR = CLAUDE_IDEAS_DIR = (
    CLAUDE_THOUGHT_GENERATION_DIR / "models_without_indexes"
)
CLAUDE_INDEXED_MODELS_DIR = CLAUDE_IDEAS_DIR = (
    CLAUDE_THOUGHT_GENERATION_DIR / "models_with_indexes"
)


# *States data direcotry (for state managements, sessions, user, etc.)
INTERVIEW_STATES_DIR = INPUT_OUTPUT_DIR / "interview_states"
INTERVIEW_STATES_FILE = INTERVIEW_STATES_DIR / "interview_states_data.json"

# ! JSON output file names for for ideas, thoughts, and sub_thoughts
# ! These names are "root" names: after importing them, you need to insert suffix in them
# ! to form the actual file names
# JSON output file paths: OpenAI
RANK_OF_THOUGHTS_FILE_NAME_ROOT = "rank_of_thoughts_output"
ARRAY_OF_THOUGHTS_FILE_NAME_ROOT = "array_of_thoughts_output"


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
