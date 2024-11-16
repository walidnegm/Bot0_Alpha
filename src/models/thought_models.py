"""
json_response_models.py

This module contains Pydantic models for validating and processing structured data related to 
concepts and sub-concepts, known as "thoughts" and "sub-thoughts." These models enable the 
use of Pydantic's validation, serialization, and transformation features, making it easier 
to process and validate batches of thought data.

Classes:
    - Level 1: IdeaJSONModel, representing the top-level idea or overarching theme 
    containing multiple thoughts.
    - Level 2: ThoughtJSONModel, representing a main thought with a list of associated sub-thoughts.
    - Level 3: SubThoughtJSONModel, representing a single sub-thought within a main thought.


Functions:
    - validate_thought_batch: Validates a batch of thought data dictionaries, logging any 
      validation errors for individual items and returning the valid thoughts.
"""

from typing import Any, Dict, Optional, List, Union
from pydantic import BaseModel, Field, ValidationError
import logging
import logging_config
import uuid

# Set up logger
logger = logging.getLogger(__name__)


# TODO: Do not delete the following function
# *Eventually, we need to Use Unique Identifiers and Include Metadata
"""
Assign a unique ID to each sub-thought upon creation: This can be a simple 
incrementing number, a UUID, or any other scheme that guarantees uniqueness.

Include Metadata: Along with the ID, consider include timestamps, authorship information, 
or other metadata that aids in logging and memory.

For the prototype, we can use looping index (idx) only; however, it has limitations such as:
Data Mutability: 
    If the list of sub-thoughts changes—due to insertion, deletion, or \
    reordering—the indices can become misaligned with the actual data, leading to errors \
        in tracking and referencing.
Asynchronous Processing: 
    In systems with asynchronous operations, multiple processes \
    might access or modify the sub-thoughts simultaneously, causing the loop index to \
        become unreliable.
Scalability Issues: 
    As the system grows in complexity, managing state and history using \
    loop indices can become cumbersome and error-prone.
Persistence and Storage: 
    When storing conversation history or logs, having a unique identifier \
    for each sub-thought is crucial for retrieval, analysis, and debugging.
"""
# class SubThoughtJSONModel(BaseModel):
# id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the sub-thought")
# name: str = Field(..., description="The name of the sub-thought")
# description: str = Field(..., description="A description of the sub-thought")
# importance: Optional[str] = Field(
#     None, description="The importance of the sub-thought, optional"
# )
# connection_to_next: Optional[str] = Field(
#     None,
#     description="Description of how this sub-thought connects to the next, optional",
# )


class SubThoughtJSONModel(BaseModel):
    """
    Represents a sub-thought within a broader main thought,
    with optional fields for indicating the importance of the sub-thought and
    how it connects to other sub-thoughts.

    Attributes:
        - name (str): The name of the sub-thought (required).
        - description (str): A description of the sub-thought (required).
        - importance (Optional[str]): The importance of the sub-thought,
        if applicable (optional).
        - connection_to_next (Optional[str]): Description of the connection
        between this sub-thought and the next, if applicable (optional).
    """

    name: str = Field(..., description="The name of the sub-thought")
    description: str = Field(..., description="A description of the sub-thought")
    importance: Optional[str] = Field(
        None, description="The importance of the sub-thought, optional"
    )
    connection_to_next: Optional[str] = Field(
        None,
        description="Description of how this sub-thought connects to the next, optional",
    )


class ThoughtJSONModel(BaseModel):
    """
    Represents a main thought associated with one or more sub-thoughts.
    This model ensures that the thought data follows the required structure and validates that
    each sub-thought meets the specified requirements.

    Attributes:
        - thought (str): The main thought or concept (required).
        - description (str): A description of the main thought (required).
        - sub_thoughts (Optional[List[SubThoughtJSONModel]]): An optional list of sub-thoughts
          associated with the main thought.
    """

    thought: str = Field(..., description="The main thought or concept")
    description: Optional[str] = Field(
        None, description="A description of the main thought"
    )  # *defines thought's description (in pydantic, if the definition
    # *immediately follows an attribute, then it defines the attribute.)
    sub_thoughts: Optional[List[SubThoughtJSONModel]] = Field(
        None,
        description="An optional list of sub-thoughts associated with the main thought",
    )

    class Config:
        "Remove None values"
        from_attributes = True

        # Exclude None values from JSON output
        json_encoders = {Optional: lambda v: v or None}

        # Exclude fields with None values in JSON output
        exclude_none = True


class ClusterJSONModel(BaseModel):
    name: str = Field(..., description="Name of the cluster")
    description: Optional[str] = Field(None, description="Description of the cluster")
    thoughts: List[str] = Field(
        ..., description="List of thought names within the cluster"
    )

    class Config:
        "Remove None values"
        from_attribute = True
        # Exclude None values from JSON output
        json_encoders = {Optional: lambda v: v or None}


class IdeaClusterJSONModel(BaseModel):
    idea: str = Field(..., description="The overarching theme or idea")
    clusters: List[ClusterJSONModel] = Field(..., description="List of clusters")


class IdeaJSONModel(BaseModel):
    idea: str = Field(..., description="The overarching theme or idea.")
    thoughts: Optional[List[ThoughtJSONModel]] = Field(
        None, description="A list of individual thoughts without clustering."
    )
    # clusters: Optional[List[ClusterJSONModel]] = Field(
    #     None, description="A list of clusters, each containing grouped thoughts."
    # )

    class Config:
        "Remove None values"
        from_attribute = True
        # Exclude None values from JSON output
        json_encoders = {Optional: lambda v: v or None}


def validate_thought_batch(thought_data_batch: List[dict]) -> List[ThoughtJSONModel]:
    """
    Validates a batch of thought data dictionaries, logging any validation errors for
    individual items and returning a list of valid Thought instances.

    Args:
        thought_data_batch (List[dict]): A list of dictionaries, each representing a thought.

    Returns:
        List[ThoughtJSONModel]: A list of validated Thought instances.

    Example:
        >>> thought_data_batch = [
                {
                    "thought": "Artificial Intelligence",
                    "sub_thoughts": [
                        {"name": "Machine Learning", 
                        "description": "AI subfield for pattern recognition"},
                        {"name": "Neural Networks", 
                        "description": "Inspired by human brain structure", \
                            "importance": "high"}
                    ]
                },
                {
                    "thought": "Climate Change",
                    "sub_thoughts": [
                        {"name": "Greenhouse Gases", 
                        "description": "Gases that trap heat in the atmosphere"}
                    ]
                }
            ]
        >>> validated_thoughts = validate_thought_batch(thought_data_batch)
        >>> for thought in validated_thoughts:
                print(thought.json(indent=4))
    """
    validated_thoughts = []
    for idx, data in enumerate(thought_data_batch):
        try:
            thought = ThoughtJSONModel(**data)
            validated_thoughts.append(thought)
        except ValidationError as e:
            logger.error(f"Validation error in batch item at index {idx}: {e}")
    return validated_thoughts


# TODO: placeholder for now; model for getting evaluation from LLM later
class EvalJSONModel(BaseModel):
    data: Union[
        Dict[str, Any], List[Dict[str, Any]]
    ]  # Allow both dict and list of dicts
