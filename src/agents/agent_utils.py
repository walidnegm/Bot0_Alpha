# transformation_utils.py

from typing import List
from models.thought_models import IdeaJSONModel  # Import the original IdeaJSONModel
from models.indexed_thought_models import (
    IndexedIdeaJSONModel,
    IndexedThoughtJSONModel,
    IndexedSubThoughtJSONModel,
)


def transform_to_indexed_idea_model(idea_model: IdeaJSONModel) -> IndexedIdeaJSONModel:
    """
    Transforms an IdeaJSONModel instance to an IndexedIdeaModel with indices added.

    Args:
        idea_model (IdeaJSONModel): The original IdeaJSONModel instance.

    Returns:
        IndexedIdeaModel: A new IndexedIdeaModel with thought and sub-thought indices added.
    """
    # Create indexed thoughts and sub-thoughts
    indexed_thoughts = []
    for thought_index, thought in enumerate(idea_model.thoughts or []):
        indexed_sub_thoughts = [
            IndexedSubThoughtJSONModel(
                sub_thought_index=sub_index,
                name=sub_thought.name,
                description=sub_thought.description,
                importance=sub_thought.importance,
                connection_to_next=sub_thought.connection_to_next,
            )
            for sub_index, sub_thought in enumerate(thought.sub_thoughts or [])
        ]
        indexed_thoughts.append(
            IndexedThoughtJSONModel(
                thought_index=thought_index,
                thought=thought.thought,
                description=thought.description,
                sub_thoughts=indexed_sub_thoughts,
            )
        )

    # Create and return the IndexedIdeaModel
    return IndexedIdeaJSONModel(idea=idea_model.idea, thoughts=indexed_thoughts)
