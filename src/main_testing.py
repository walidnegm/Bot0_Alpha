from pathlib import Path

from pipelines.thought_processing_pipeline import indexed_thought_processing_pipeline
from utils.get_file_names import get_file_names

from config import CLAUDE_INDEXED_MODELS_DIR, OPENAI_INDEXED_MODELS_DIR


idea_file_name = (
    "array_of_thoughts_output_with_index_embedded_software_development_openai.json"
)
idea_file = Path(OPENAI_INDEXED_MODELS_DIR / idea_file_name)

sub_thoughts = indexed_thought_processing_pipeline(idea_file)
print(sub_thoughts)
