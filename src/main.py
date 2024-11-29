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


def main():
    pass


if __name__ == "__main__":
    main()
