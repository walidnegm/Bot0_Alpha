"""
File: base_models.py
Last Updated on:

pydantic base models
"""

from pydantic import BaseModel, ConfigDict, ValidationError, constr
from typing import List, Union, Any, Optional, Dict
import pandas as pd
import json
import logging

# Define Pydantic Models for Different Expected Response Types


class TextResponse(BaseModel):
    content: str

    model_config = ConfigDict(
        frozen=True
    )  # Use ConfigDict instead of class-based Config


# Define a model for the sub-concepts
class SubConcept(BaseModel):
    name: str
    description: str
    details: Optional[Dict[str, Union[str, List[str]]]] = None


class JSONResponse(BaseModel):
    data: Union[
        Dict[str, Any], List[Dict[str, Any]]
    ]  # Allow both dict and list of dicts

    class Config:
        arbitrary_types_allowed = True


# class JSONResponse(BaseModel):
#     concept: str
#     sub_concepts: List[SubConcept]

#     class Config:
#         arbitrary_types_allowed = True  # Allow non-standard types like DataFrame


class TabularResponse(BaseModel):
    data: pd.DataFrame  # Pandas DataFrame for tabular data

    class Config:
        arbitrary_types_allowed = True  # Allow non-standard types like DataFrame


class CodeResponse(BaseModel):
    code: str

    class Config:
        arbitrary_types_allowed = True
