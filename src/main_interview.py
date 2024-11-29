from pathlib import Path
import asyncio
import logging
import logging_config

from pipelines.interview_pipeline_async import interview_pipeline_async
from project_config import (
    CLAUDE_INDEXED_MODELS_DIR,
    OPENAI_INDEXED_MODELS_DIR,
    MEMORY_DIR,
)

logger = logging.getLogger(__name__)

# *Data source:
source_file_list = [
    "array_of_thoughts_output_with_index_embedded_software_development_claude.json",
    "array_of_thoughts_output_with_index_embedded_software_development_in_aerospace_claude.json",
    "array_of_thoughts_output_with_index_embedded_software_development_in_automotive_claude.json",
]


def run_interview_pipeline():
    data_file = CLAUDE_INDEXED_MODELS_DIR / source_file_list[0]
    memory_file = MEMORY_DIR / "chat_memory.json"
    user_id = "user_1"
    asyncio.run(
        interview_pipeline_async(
            thought_data_file=data_file, user_id=user_id, memory_file=memory_file
        )
    )
    print("finished pipeline")


if __name__ == "__main__":
    run_interview_pipeline()
