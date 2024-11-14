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

# indexed_idea_model.py


class IndexedSubThoughtJSONModel(BaseModel):
    sub_thought_index: int = Field(..., description="Index of this sub-thought")
    name: str = Field(..., description="The name of the sub-thought")
    description: str = Field(..., description="A description of the sub-thought")
    importance: Optional[str] = Field(None, description="Importance of the sub-thought")
    connection_to_next: Optional[str] = Field(
        None, description="Connection to next sub-thought"
    )


class IndexedThoughtJSONModel(BaseModel):
    thought_index: int = Field(..., description="Index of this main thought")
    thought: str = Field(..., description="The main thought or concept")
    description: Optional[str] = Field(
        None, description="Description of the main thought"
    )
    sub_thoughts: Optional[List[IndexedSubThoughtJSONModel]] = Field(
        None, description="List of indexed sub-thoughts"
    )


class IndexedIdeaJSONModel(BaseModel):
    idea: str = Field(..., description="The overarching theme or idea")
    thoughts: Optional[List[IndexedThoughtJSONModel]] = Field(
        None, description="List of indexed thoughts for the idea"
    )
